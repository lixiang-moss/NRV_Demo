#!/usr/bin/env python3
"""Receive decoded NRV events over TCP and run recurrent E2FAI inference."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import queue
import signal
import socket
import struct
import threading
import time

import cv2
import numpy as np
import torch

from nrv_e2fai import load_recurrent_model
from nrv_e2fai.visualization import event_polarity_rgb, flow_hsv_rgb


EVENT_DTYPE = np.dtype([
    ("x", "<u2"), ("y", "<u2"), ("timestamp_ns", "<u8"), ("polarity", "u1")
])
BRIDGE_HEADER = struct.Struct("!4sIIII")
BRIDGE_V2_METADATA = struct.Struct("!QQQQ")  # start/end ns, cumulative RAW missing/resets
COMPACT_EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("polarity", "u1")])
MAX_PACKET_SPAN_NS = 100_000_000  # Matches the adapter's host-arrival interpolation.
MAX_BATCH_EVENTS = 20_000_000
NUM_BINS = 15


def join_events(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=EVENT_DTYPE)
    if len(parts) == 1:
        return parts[0]
    # Structured concatenate copies fields individually; the packed wire records
    # can instead be copied once as contiguous bytes without changing any bits.
    return np.concatenate([
        np.ascontiguousarray(part).view(np.uint8) for part in parts
    ]).view(EVENT_DTYPE)


def recv_exact(connection: socket.socket, size: int, should_stop=lambda: False) -> bytearray | None:
    data = bytearray(size)
    view = memoryview(data)
    received = 0
    while received < size:
        try:
            count = connection.recv_into(view[received:])
        except socket.timeout:
            if should_stop():
                return None
            continue
        if count == 0:
            return None
        received += count
    # Each read owns its buffer; NumPy can retain it without another payload copy.
    return data


def unpack_compact_events(payload, count: int, start_ns: int, end_ns: int) -> np.ndarray:
    """Restore the existing synthetic packet timestamps, not sensor timestamps."""
    if not 0 <= count <= MAX_BATCH_EVENTS:
        raise ValueError("Invalid compact event count")
    if not 0 <= start_ns <= end_ns < (1 << 64) or end_ns - start_ns > MAX_PACKET_SPAN_NS:
        raise ValueError("Invalid compact packet time interval")
    if len(payload) != count * COMPACT_EVENT_DTYPE.itemsize:
        raise ValueError("Invalid compact event payload length")
    packed = np.frombuffer(payload, dtype=COMPACT_EVENT_DTYPE)
    events = np.empty(count, dtype=EVENT_DTYPE)
    for field in ("x", "y", "polarity"):
        events[field] = packed[field]
    if count > 1:
        # The bounded product fits uint64. Integer arithmetic preserves endpoints
        # exactly even for nanosecond epoch timestamps (which float64 cannot).
        timestamps = np.arange(count, dtype=np.uint64)
        timestamps *= np.uint64(end_ns - start_ns)
        timestamps //= np.uint64(count - 1)
        timestamps += np.uint64(start_ns)
        events["timestamp_ns"] = timestamps
    else:
        events["timestamp_ns"] = start_ns
    return events


class EventTcpReceiver:
    """Continuously drain the bridge socket while the main thread runs the GPU."""

    def __init__(self, server: socket.socket, should_stop) -> None:
        self.server = server
        self.should_stop = should_stop
        self.batches: queue.Queue = queue.Queue(maxsize=32)
        self.dropped_batches = 0
        self.received_batches = 0
        self.dequeued_batches = 0
        self.connections = 0
        self.queue_peak = 0
        self.received_events = 0
        self.received_bytes = 0
        self.protocols: set[str] = set()
        self.adapter_raw_missing_reported = 0
        self.adapter_raw_resets_reported = 0
        self.unpack_ms = 0.0
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name="event-bridge-reader", daemon=True)
        self.thread.start()

    def _put(self, item) -> bool:
        while not self.should_stop():
            try:
                self.batches.put(item, timeout=0.2)
                self.queue_peak = max(self.queue_peak, self.batches.qsize())
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            while not self.should_stop():
                try:
                    connection, address = self.server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                self.connections += 1
                generation = self.connections
                previous_sequence: int | None = None
                previous_upstream: tuple[int, int] | None = None
                print(f"Event bridge connected from {address[0]}:{address[1]}", flush=True)
                connection.settimeout(0.5)
                with connection:
                    while not self.should_stop():
                        header = recv_exact(
                            connection, BRIDGE_HEADER.size, self.should_stop
                        )
                        if header is None:
                            break
                        magic, width, height, count, sequence = BRIDGE_HEADER.unpack(header)
                        if (
                            magic not in (b"NRV1", b"NRV2")
                            or not 0 < width <= 10_000
                            or not 0 < height <= 10_000
                        ):
                            raise RuntimeError("Invalid NRV bridge header")
                        if count > MAX_BATCH_EVENTS:
                            raise RuntimeError(
                                f"Refusing implausible event batch of {count:,} events"
                            )
                        interval = None
                        upstream_gap = False
                        if magic == b"NRV2":
                            encoded_interval = recv_exact(connection, BRIDGE_V2_METADATA.size, self.should_stop)
                            if encoded_interval is None:
                                break
                            start_ns, end_ns, raw_missing, raw_resets = BRIDGE_V2_METADATA.unpack(encoded_interval)
                            interval = (start_ns, end_ns)
                            upstream = (raw_missing, raw_resets)
                            upstream_gap = previous_upstream is not None and upstream != previous_upstream
                            previous_upstream = upstream
                            self.adapter_raw_missing_reported = max(self.adapter_raw_missing_reported, raw_missing)
                            self.adapter_raw_resets_reported = max(self.adapter_raw_resets_reported, raw_resets)
                            if interval[1] < interval[0] or interval[1] - interval[0] > MAX_PACKET_SPAN_NS:
                                raise RuntimeError("Invalid compact packet time interval")
                        itemsize = COMPACT_EVENT_DTYPE.itemsize if interval is not None else EVENT_DTYPE.itemsize
                        payload = recv_exact(connection, count * itemsize, self.should_stop)
                        if payload is None:
                            break
                        sequence_gap = (
                            previous_sequence is not None
                            and sequence != (previous_sequence + 1) % (1 << 32)
                        )
                        if sequence_gap:
                            self.dropped_batches += (
                                sequence - previous_sequence - 1
                            ) % (1 << 32)
                        previous_sequence = sequence
                        unpack_tick = time.perf_counter()
                        events = (
                            unpack_compact_events(payload, count, *interval)
                            if interval is not None else np.frombuffer(payload, dtype=EVENT_DTYPE)
                        )
                        self.unpack_ms += 1000 * (time.perf_counter() - unpack_tick)
                        self.received_batches += 1
                        self.received_events += count
                        self.received_bytes += BRIDGE_HEADER.size + len(payload) + (
                            BRIDGE_V2_METADATA.size if interval is not None else 0
                        )
                        self.protocols.add(magic.decode("ascii"))
                        if not self._put(
                            (generation, width, height, events, sequence_gap or upstream_gap)
                        ):
                            return
                print("Event bridge disconnected; waiting for reconnection", flush=True)
        except Exception as error:
            self.error = error

    def get(self, timeout: float = 0.5):
        item = self.batches.get(timeout=timeout)
        self.dequeued_batches += 1
        return item

    def get_nowait(self):
        item = self.batches.get_nowait()
        self.dequeued_batches += 1
        return item

    def close(self) -> None:
        self.server.close()
        self.thread.join(timeout=2.0)


class EventWindowBuffer:
    def __init__(self, window_ns: int, max_gap_windows: int = 3) -> None:
        self.window_ns = window_ns
        self.max_gap_ns = max_gap_windows * window_ns
        self.parts: deque[np.ndarray] = deque()
        self.start_ns: int | None = None
        self.last_event_ns: int | None = None
        self.width = self.height = 0

    def reset(self) -> None:
        self.parts.clear()
        self.start_ns = None
        self.last_event_ns = None

    @property
    def events(self) -> np.ndarray:
        return join_events(list(self.parts))

    def push(
        self, events: np.ndarray, width: int, height: int
    ) -> tuple[list[np.ndarray], bool]:
        if not len(events):
            return [], False
        timestamps = events["timestamp_ns"]
        if np.any(timestamps[1:] < timestamps[:-1]):
            events = np.sort(events, order="timestamp_ns")
            timestamps = events["timestamp_ns"]

        discontinuity = False
        first_ns, last_ns = int(timestamps[0]), int(timestamps[-1])
        if (
            self.last_event_ns is not None
            and (first_ns < self.last_event_ns
                 or first_ns > self.last_event_ns + self.max_gap_ns)
        ) or (self.width and (width != self.width or height != self.height)):
            self.reset()
            discontinuity = True

        self.width, self.height = width, height
        if self.start_ns is None:
            self.start_ns = first_ns
        self.parts.append(events)
        self.last_event_ns = last_ns
        complete: list[np.ndarray] = []
        while last_ns >= self.start_ns + self.window_ns:
            end_ns = self.start_ns + self.window_ns
            window_parts = []
            while self.parts and int(self.parts[0]["timestamp_ns"][0]) < end_ns:
                part = self.parts.popleft()
                if int(part["timestamp_ns"][-1]) < end_ns:
                    window_parts.append(part)
                    continue
                cut = int(np.searchsorted(part["timestamp_ns"], end_ns, side="left"))
                window_parts.append(part[:cut])
                self.parts.appendleft(part[cut:])
                break
            if window_parts:
                complete.append(join_events(window_parts))
            self.start_ns = end_ns
        return complete, discontinuity


class RunningRange:
    def __init__(self, momentum: float = 0.9) -> None:
        self.momentum = momentum
        self.low: float | None = None
        self.high: float | None = None

    def render(self, values: np.ndarray) -> np.ndarray:
        finite = values[np.isfinite(values)]
        low, high = np.percentile(finite, (1, 99)) if finite.size else (0.0, 1.0)
        if high <= low:
            high = low + 1e-6
        if self.low is None:
            self.low, self.high = float(low), float(high)
        else:
            self.low = self.momentum * self.low + (1 - self.momentum) * float(low)
            self.high = self.momentum * self.high + (1 - self.momentum) * float(high)
        gray = np.clip((values - self.low) / max(self.high - self.low, 1e-6), 0, 1)
        gray = np.nan_to_num(gray)
        return np.repeat((gray[..., None] * 255).round().astype(np.uint8), 3, axis=2)


def voxelize(
    events: np.ndarray,
    source_width: int,
    source_height: int,
    device: torch.device,
    max_events: int,
    sensor_width: int,
    sensor_height: int,
    *,
    input_width: int | None = None,
    input_height: int | None = None,
) -> torch.Tensor:
    if len(events) < 2:
        raise ValueError("A voxel needs at least two events")
    if (source_width, source_height) != (sensor_width, sensor_height):
        raise ValueError(
            f"Camera reports {source_width}x{source_height}, but the configured "
            f"sensor is {sensor_width}x{sensor_height}"
        )
    input_width = sensor_width if input_width is None else input_width
    input_height = sensor_height if input_height is None else input_height
    original_count = len(events)
    if original_count > max_events:
        indices = np.linspace(0, original_count - 1, max_events, dtype=np.int64)
        events = events[indices]
    # PyTorch 2.1 cannot construct a tensor directly from NumPy uint16.
    x = torch.tensor(events["x"].astype(np.int32), device=device)
    y = torch.tensor(events["y"].astype(np.int32), device=device)
    valid_xy = (x >= 0) & (x < source_width) & (y >= 0) & (y < source_height)
    # Bin the full sensor field into the model grid; keep all event weights.
    # Integer mapping sends (959, 719) to (639, 479) for 960x720 -> 640x480.
    if (input_width, input_height) != (source_width, source_height):
        x = torch.div(x * input_width, source_width, rounding_mode="floor")
        y = torch.div(y * input_height, source_height, rounding_mode="floor")
    relative_ns = events["timestamp_ns"].copy() - events["timestamp_ns"][0]
    if relative_ns[-1] == 0:
        normalized_time = torch.linspace(
            0, NUM_BINS - 1, len(events), device=device, dtype=torch.float32
        )
    else:
        normalized_time = torch.tensor(
            relative_ns.astype(np.float32) / float(relative_ns[-1]), device=device
        ) * (NUM_BINS - 1)
    polarity = torch.tensor(events["polarity"].copy(), device=device, dtype=torch.float32)
    t0 = normalized_time.int()
    signed = 2 * polarity - 1
    voxel = torch.zeros(
        NUM_BINS * input_height * input_width, dtype=torch.float32, device=device
    )
    # Sensor x/y coordinates are integers, so only time needs interpolation.
    # Multiple events can hit the same voxel, hence scatter_add rather than assignment.
    pixel_index = input_width * y + x
    plane_size = input_height * input_width
    for ti in (t0, t0 + 1):
        valid = valid_xy & (ti >= 0) & (ti < NUM_BINS)
        index = (plane_size * ti + pixel_index).clamp(
            0, NUM_BINS * plane_size - 1
        ).long()
        weight = signed * (1 - (ti - normalized_time).abs())
        voxel.scatter_add_(0, index, weight * valid)
    voxel = voxel.view(NUM_BINS, input_height, input_width)
    if len(events) != original_count:
        voxel.mul_(original_count / len(events))
    return voxel.unsqueeze(0)


def label(panel: np.ndarray, title: str) -> np.ndarray:
    result = np.pad(panel, ((28, 0), (0, 0), (0, 0)), constant_values=0)
    cv2.putText(result, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return result


def montage(
    output: dict[str, torch.Tensor],
    image_range: RunningRange,
    flow_max_px: float,
    event_count: int,
    inference_ms: float,
) -> np.ndarray:
    log_image = output["log_image"][0, 0].detach().float().cpu().numpy()
    image_rgb = image_range.render(log_image)
    flow_rgb = flow_hsv_rgb(output["flow"][:, None], flow_max_px)[0, 0]
    height, width = log_image.shape
    panels = [
        label(image_rgb, f"E2FAI image {width}x{height} ({event_count:,} events)"),
        label(flow_rgb, f"E2FAI flow (max {flow_max_px:g} px)"),
    ]
    frame = np.concatenate(panels, axis=1)
    cv2.putText(frame, f"inference {inference_ms:.1f} ms", (frame.shape[1] - 210, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def self_check() -> None:
    window = EventWindowBuffer(100_000_000)
    events = np.empty(21, dtype=EVENT_DTYPE)
    events["x"] = np.arange(21) % 960
    events["y"] = np.arange(21) % 720
    events["timestamp_ns"] = np.arange(21, dtype=np.uint64) * 10_000_000
    events["polarity"] = np.arange(21) % 2
    chunks, reset = window.push(events, 960, 720)
    assert not reset and len(chunks) == 2
    voxel = voxelize(chunks[0], 960, 720, torch.device("cpu"), 1_000, 960, 720)
    assert voxel.shape == (1, NUM_BINS, 720, 960)
    assert torch.isfinite(voxel).all() and voxel.abs().sum() > 0
    assert event_polarity_rgb(voxel[:, None]).shape == (1, 1, 720, 960, 3)
    packed = BRIDGE_HEADER.pack(b"NRV1", 960, 720, len(events), 7)
    assert BRIDGE_HEADER.unpack(packed) == (b"NRV1", 960, 720, len(events), 7)
    print("NRV event bridge/window/voxel self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--window-ms", type=float, default=100.0)
    parser.add_argument("--sensor-width", type=int, default=960)
    parser.add_argument("--sensor-height", type=int, default=720)
    parser.add_argument("--input-width", type=int, help="model width (default: sensor width)")
    parser.add_argument("--input-height", type=int, help="model height (default: sensor height)")
    parser.add_argument("--image-checkpoint", type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flow-max-px", type=float, default=20.0)
    parser.add_argument("--max-events", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("output/e2fai_live"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--latest-batch",
        action="store_true",
        help=(
            "coalesce all currently pending event batches for low-latency inference "
            "instead of waiting for a strict fixed-duration window"
        ),
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check and (args.image_checkpoint is None or args.backbone is None):
        parser.error("--image-checkpoint and --backbone are required")
    if min(args.window_ms, args.flow_max_px, args.sensor_width, args.sensor_height) <= 0:
        parser.error("window, event limit, flow scale and sensor size must be positive")
    if args.max_events < 2:
        parser.error("max-events must be at least 2")
    if (args.input_width is None) != (args.input_height is None):
        parser.error("--input-width and --input-height must be specified together")
    if args.input_width is None:
        args.input_width, args.input_height = args.sensor_width, args.sensor_height
    if min(args.input_width, args.input_height) <= 0:
        parser.error("input width and height must be positive")
    if args.input_width > args.sensor_width or args.input_height > args.sensor_height:
        parser.error("input resolution cannot exceed the sensor resolution")
    if args.input_width * args.sensor_height != args.input_height * args.sensor_width:
        parser.error("input resolution must preserve the sensor aspect ratio")
    if args.input_width % 16 or args.input_height % 16:
        parser.error("input width and height must be divisible by the 16-pixel flow tile")
    return args


def pipeline_diagnostics(output_dir: Path) -> dict:
    """Keep missing diagnostics distinct from a measured zero loss count."""
    sources = {}
    for name, filename in (("raw_monitor", "summary.json"), ("adapter", "adapter_summary.json")):
        path = output_dir / filename
        sources[name] = json.loads(path.read_text()) if path.is_file() else None
    raw, adapter = sources["raw_monitor"], sources["adapter"]
    detected = bool(raw and raw.get("raw_sequence_gaps", 0)) or bool(adapter and any(
        adapter.get(key, 0) for key in (
            "raw_missing_packets", "raw_reordered_or_reset", "bridge_dropped_batches",
        )
    ))
    sources["upstream_loss_detected"] = detected if detected or (raw is not None and adapter is not None) else None
    return sources


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, metadata = load_recurrent_model(
        args.image_checkpoint, backbone_checkpoint=args.backbone, device=device,
        sensor_width=args.input_width, sensor_height=args.input_height,
    )
    torch.backends.cudnn.benchmark = False
    with torch.inference_mode():
        model.forward_step(torch.zeros(
            1, NUM_BINS, args.input_height, args.input_width, device=device
        ))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    server.settimeout(0.5)
    (args.output_dir / "ready").write_text(f"{args.host}:{args.port}\n")
    print(f"Ready for NRV decoded events on {args.host}:{args.port}", flush=True)

    buffer = EventWindowBuffer(round(args.window_ms * 1e6))
    state = None
    previous_connection: int | None = None
    image_range = RunningRange()
    voxel_times: list[float] = []
    inference_times: list[float] = []
    render_times: list[float] = []
    frame_times: list[float] = []
    frame_completed_at: list[float] = []
    input_spans_ms: list[float] = []
    assembly_ms = 0.0
    assembly_batches = 0
    processed_windows = coalesced_batches = 0
    voxel_input_events = voxel_kept_events = voxel_subsampled_windows = 0
    last_frame = None
    video = None
    started = time.monotonic()
    receiver = EventTcpReceiver(server, lambda: stopped)
    try:
        while not stopped:
            try:
                generation, width, height, events, sequence_gap = receiver.get()
            except queue.Empty:
                if receiver.error is not None:
                    raise RuntimeError("Event bridge reader failed") from receiver.error
                continue
            assembly_tick = time.perf_counter()
            if previous_connection is not None and generation != previous_connection:
                buffer.reset()
                state = None
            previous_connection = generation
            if args.latest_batch:
                parts = [events]
                while True:
                    try:
                        next_generation, next_width, next_height, next_events, next_gap = (
                            receiver.get_nowait()
                        )
                    except queue.Empty:
                        break
                    if (next_generation, next_width, next_height) != (
                        generation, width, height
                    ):
                        state = None
                    generation, width, height = (
                        next_generation, next_width, next_height
                    )
                    sequence_gap = sequence_gap or next_gap
                    parts.append(next_events)
                if len(parts) > 1:
                    coalesced_batches += len(parts) - 1
                    events = join_events(parts)
                if sequence_gap:
                    state = None
                if np.any(events["timestamp_ns"][1:] < events["timestamp_ns"][:-1]):
                    events = np.sort(events, order="timestamp_ns")
                windows = [events]
                buffer.reset()
            else:
                if sequence_gap:
                    buffer.reset()
                    state = None
                windows, discontinuity = buffer.push(events, width, height)
                if discontinuity:
                    state = None
            assembly_ms += 1000 * (time.perf_counter() - assembly_tick)
            assembly_batches += 1
            for event_window in windows:
                if len(event_window) < 2:
                    continue
                frame_tick = time.perf_counter()
                input_spans_ms.append(
                    (
                        int(event_window["timestamp_ns"][-1])
                        - int(event_window["timestamp_ns"][0])
                    ) / 1e6
                )
                voxel = voxelize(
                    event_window, width, height, device, args.max_events,
                    args.sensor_width, args.sensor_height,
                    input_width=args.input_width, input_height=args.input_height,
                )
                voxel_input_events += len(event_window)
                voxel_kept_events += min(len(event_window), args.max_events)
                voxel_subsampled_windows += int(len(event_window) > args.max_events)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                voxel_times.append(1000 * (time.perf_counter() - frame_tick))
                tick = time.perf_counter()
                with torch.inference_mode():
                    output, state = model.forward_step(voxel, state)
                if not all(torch.isfinite(output[key]).all() for key in ("log_image", "flow")):
                    print("Non-finite model output; resetting recurrent state", flush=True)
                    state = None
                    continue
                state = state.detach()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_ms = 1000 * (time.perf_counter() - tick)
                inference_times.append(inference_ms)
                processed_windows += 1
                render_tick = time.perf_counter()
                last_frame = montage(
                    output, image_range, args.flow_max_px,
                    len(event_window), inference_ms,
                )
                render_times.append(1000 * (time.perf_counter() - render_tick))
                if args.record:
                    if video is None:
                        height_px, width_px = last_frame.shape[:2]
                        video = cv2.VideoWriter(
                            str(args.output_dir / "e2fai_live.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            1000.0 / args.window_ms,
                            (width_px, height_px),
                        )
                        if not video.isOpened():
                            raise RuntimeError("Could not open MP4 writer")
                    video.write(last_frame)
                if not args.headless:
                    cv2.imshow("NRV E2FAI real-time", last_frame)
                    cv2.waitKey(1)
                completed_at = time.perf_counter()
                frame_times.append(1000 * (completed_at - frame_tick))
                frame_completed_at.append(completed_at)
    finally:
        stopped = True
        receiver.close()
        if video is not None:
            video.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if last_frame is not None:
            cv2.imwrite(str(args.output_dir / "last.png"), last_frame)
        diagnostics = pipeline_diagnostics(args.output_dir)
        if receiver.dropped_batches or receiver.adapter_raw_missing_reported or receiver.adapter_raw_resets_reported:
            diagnostics["upstream_loss_detected"] = True
        summary = {
            "processed_windows": processed_windows,
            "max_events_per_voxel": args.max_events,
            "voxel_input_events": voxel_input_events,
            "voxel_kept_events": voxel_kept_events,
            "voxel_subsampled_windows": voxel_subsampled_windows,
            "bridge_received_batches": receiver.received_batches,
            "bridge_received_events": receiver.received_events,
            "bridge_received_bytes": receiver.received_bytes,
            "bridge_protocols": sorted(receiver.protocols),
            "mean_bridge_unpack_ms": receiver.unpack_ms / receiver.received_batches if receiver.received_batches else None,
            "mean_input_assembly_ms_per_batch": assembly_ms / assembly_batches if assembly_batches else None,
            "bridge_unprocessed_batches_at_shutdown": receiver.received_batches - receiver.dequeued_batches,
            "buffered_window_events_at_shutdown": sum(len(part) for part in buffer.parts),
            "adapter_raw_missing_reported": receiver.adapter_raw_missing_reported,
            "adapter_raw_resets_reported": receiver.adapter_raw_resets_reported,
            "bridge_coalesced_batches": coalesced_batches,
            "bridge_queue_peak": receiver.queue_peak,
            "sender_dropped_batches_observed": receiver.dropped_batches,
            "input_mode": "coalesced_latest" if args.latest_batch else "fixed_window",
            "window_ms": args.window_ms,
            "sensor_resolution": [args.sensor_width, args.sensor_height],
            "model_input": [NUM_BINS, args.input_height, args.input_width],
            "coordinate_mapping": "floor_bin",
            "flow_units": "model_input_pixels_per_window",
            "flow_to_sensor_scale_xy": [
                args.sensor_width / args.input_width,
                args.sensor_height / args.input_height,
            ],
            "mean_input_span_ms": float(np.mean(input_spans_ms)) if input_spans_ms else None,
            "mean_voxel_ms": float(np.mean(voxel_times)) if voxel_times else None,
            "mean_inference_ms": float(np.mean(inference_times)) if inference_times else None,
            "p95_inference_ms": float(np.percentile(inference_times, 95)) if inference_times else None,
            "mean_render_ms": float(np.mean(render_times)) if render_times else None,
            # Excludes input wait/assembly; includes voxel, model, render, record and GUI.
            "mean_frame_processing_ms": float(np.mean(frame_times)) if frame_times else None,
            "p95_frame_processing_ms": float(np.percentile(frame_times, 95)) if frame_times else None,
            # Measure delivered frame intervals, not configured DURATION or startup time.
            "output_fps": (
                (len(frame_completed_at) - 1) / (frame_completed_at[-1] - frame_completed_at[0])
                if len(frame_completed_at) > 1 else None
            ),
            "p95_frame_interval_ms": (
                float(np.percentile(np.diff(frame_completed_at) * 1000, 95))
                if len(frame_completed_at) > 1 else None
            ),
            "wall_seconds": time.monotonic() - started,
            "model": metadata,
            "pipeline": diagnostics,
        }
        (args.output_dir / "e2fai_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        if diagnostics["upstream_loss_detected"]:
            print("WARNING: upstream event loss was observed; inspect pipeline.adapter and bridge counters.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
