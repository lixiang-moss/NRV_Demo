#!/usr/bin/env python3
"""Receive decoded NRV events over TCP and run recurrent E2FAI inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import socket
import struct
import time

import cv2
import numpy as np
import torch

from learning_everything.inference import load_recurrent_model
from learning_everything.visualization import event_polarity_rgb, flow_hsv_rgb


EVENT_DTYPE = np.dtype([
    ("x", "<u2"), ("y", "<u2"), ("timestamp_ns", "<u8"), ("polarity", "u1")
])
BRIDGE_HEADER = struct.Struct("!4sIIII")
MAX_BATCH_EVENTS = 20_000_000
NUM_BINS = 15


def recv_exact(connection: socket.socket, size: int, should_stop=lambda: False) -> bytes | None:
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
    return bytes(data)


class EventWindowBuffer:
    def __init__(self, window_ns: int, max_gap_windows: int = 3) -> None:
        self.window_ns = window_ns
        self.max_gap_ns = max_gap_windows * window_ns
        self.events = np.empty(0, dtype=EVENT_DTYPE)
        self.start_ns: int | None = None
        self.width = self.height = 0

    def reset(self) -> None:
        self.events = np.empty(0, dtype=EVENT_DTYPE)
        self.start_ns = None

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
            self.start_ns is not None
            and (first_ns < int(self.events["timestamp_ns"][-1])
                 or first_ns > int(self.events["timestamp_ns"][-1]) + self.max_gap_ns)
        ) or (self.width and (width != self.width or height != self.height)):
            self.reset()
            discontinuity = True

        self.width, self.height = width, height
        if self.start_ns is None:
            self.start_ns = first_ns
        self.events = np.concatenate((self.events, events))
        complete: list[np.ndarray] = []
        while last_ns >= self.start_ns + self.window_ns:
            end_ns = self.start_ns + self.window_ns
            cut = int(np.searchsorted(self.events["timestamp_ns"], end_ns, side="left"))
            if cut:
                complete.append(self.events[:cut])
            self.events = self.events[cut:]
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
) -> torch.Tensor:
    if len(events) < 2:
        raise ValueError("A voxel needs at least two events")
    if (source_width, source_height) != (sensor_width, sensor_height):
        raise ValueError(
            f"Camera reports {source_width}x{source_height}, but the model was "
            f"created for {sensor_width}x{sensor_height}"
        )
    original_count = len(events)
    if original_count > max_events:
        indices = np.linspace(0, original_count - 1, max_events, dtype=np.int64)
        events = events[indices]
    x = torch.tensor(events["x"].copy(), device=device, dtype=torch.float32)
    y = torch.tensor(events["y"].copy(), device=device, dtype=torch.float32)
    relative_ns = events["timestamp_ns"].copy() - events["timestamp_ns"][0]
    if relative_ns[-1] == 0:
        raise ValueError("A voxel needs two distinct event timestamps")
    normalized_time = torch.tensor(
        relative_ns.astype(np.float32) / float(relative_ns[-1]), device=device
    ) * (NUM_BINS - 1)
    polarity = torch.tensor(events["polarity"].copy(), device=device, dtype=torch.float32)
    x0, y0, t0 = x.int(), y.int(), normalized_time.int()
    signed = 2 * polarity - 1
    voxel = torch.zeros(
        NUM_BINS * sensor_height * sensor_width, dtype=torch.float32, device=device
    )
    for xi in (x0, x0 + 1):
        for yi in (y0, y0 + 1):
            for ti in (t0, t0 + 1):
                valid = (
                    (xi >= 0) & (xi < sensor_width)
                    & (yi >= 0) & (yi < sensor_height)
                    & (ti >= 0) & (ti < NUM_BINS)
                )
                index = (
                    sensor_height * sensor_width * ti + sensor_width * yi + xi
                ).clamp(0, NUM_BINS * sensor_height * sensor_width - 1).long()
                weight = (
                    signed * (1 - (xi - x).abs()) * (1 - (yi - y).abs())
                    * (1 - (ti - normalized_time).abs())
                )
                voxel.scatter_add_(0, index, weight * valid)
    voxel = voxel.view(NUM_BINS, sensor_height, sensor_width)
    if len(events) != original_count:
        voxel.mul_(original_count / len(events))
    return voxel.unsqueeze(0)


def label(panel: np.ndarray, title: str) -> np.ndarray:
    result = np.pad(panel, ((28, 0), (0, 0), (0, 0)), constant_values=0)
    cv2.putText(result, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return result


def montage(
    voxel: torch.Tensor,
    output: dict[str, torch.Tensor],
    image_range: RunningRange,
    flow_max_px: float,
    event_count: int,
    inference_ms: float,
) -> np.ndarray:
    events_rgb = event_polarity_rgb(voxel[:, None])[0, 0]
    log_image = output["log_image"][0, 0].detach().float().cpu().numpy()
    image_rgb = image_range.render(log_image)
    flow_rgb = flow_hsv_rgb(output["flow"][:, None], flow_max_px)[0][0, 0]
    panels = [
        label(events_rgb, f"Events ({event_count:,} / window)"),
        label(image_rgb, "E2FAI + recurrent image residual"),
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
    parser.add_argument("--image-checkpoint", type=Path)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flow-max-px", type=float, default=20.0)
    parser.add_argument("--max-events", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("output/e2fai_live"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check and (args.image_checkpoint is None or args.backbone is None):
        parser.error("--image-checkpoint and --backbone are required")
    if min(args.window_ms, args.flow_max_px, args.sensor_width, args.sensor_height) <= 0:
        parser.error("window, event limit, flow scale and sensor size must be positive")
    if args.max_events < 2:
        parser.error("max-events must be at least 2")
    if args.sensor_width % 16 or args.sensor_height % 16:
        parser.error("sensor width and height must be divisible by the 16-pixel flow tile")
    return args


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
        sensor_width=args.sensor_width, sensor_height=args.sensor_height,
    )
    torch.backends.cudnn.benchmark = False
    with torch.inference_mode():
        model.forward_step(torch.zeros(
            1, NUM_BINS, args.sensor_height, args.sensor_width, device=device
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
    previous_sequence: int | None = None
    image_range = RunningRange()
    inference_times: list[float] = []
    processed_windows = dropped_batches = 0
    last_frame = None
    video = None
    started = time.monotonic()
    try:
        while not stopped:
            try:
                connection, address = server.accept()
            except socket.timeout:
                continue
            print(f"Event bridge connected from {address[0]}:{address[1]}", flush=True)
            connection.settimeout(0.5)
            with connection:
                while not stopped:
                    header = recv_exact(connection, BRIDGE_HEADER.size, lambda: stopped)
                    if header is None:
                        break
                    magic, width, height, count, sequence = BRIDGE_HEADER.unpack(header)
                    if magic != b"NRV1" or not 0 < width <= 10_000 or not 0 < height <= 10_000:
                        raise RuntimeError("Invalid NRV bridge header")
                    if count > MAX_BATCH_EVENTS:
                        raise RuntimeError(f"Refusing implausible event batch of {count:,} events")
                    payload = recv_exact(
                        connection, count * EVENT_DTYPE.itemsize, lambda: stopped
                    )
                    if payload is None:
                        break
                    events = np.frombuffer(payload, dtype=EVENT_DTYPE).copy()
                    if previous_sequence is not None and sequence != (previous_sequence + 1) % (1 << 32):
                        dropped_batches += (sequence - previous_sequence - 1) % (1 << 32)
                        buffer.reset()
                        state = None
                    previous_sequence = sequence
                    windows, discontinuity = buffer.push(events, width, height)
                    if discontinuity:
                        state = None
                    for event_window in windows:
                        if len(event_window) < 2:
                            continue
                        voxel = voxelize(
                            event_window, width, height, device, args.max_events,
                            args.sensor_width, args.sensor_height,
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
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
                        last_frame = montage(
                            voxel, output, image_range, args.flow_max_px,
                            len(event_window), inference_ms,
                        )
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
            buffer.reset()
            state = None
            previous_sequence = None
            print("Event bridge disconnected; waiting for reconnection", flush=True)
    finally:
        server.close()
        if video is not None:
            video.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if last_frame is not None:
            cv2.imwrite(str(args.output_dir / "last.png"), last_frame)
        summary = {
            "processed_windows": processed_windows,
            "sender_dropped_batches_observed": dropped_batches,
            "window_ms": args.window_ms,
            "model_input": [NUM_BINS, args.sensor_height, args.sensor_width],
            "mean_inference_ms": float(np.mean(inference_times)) if inference_times else None,
            "p95_inference_ms": float(np.percentile(inference_times, 95)) if inference_times else None,
            "wall_seconds": time.monotonic() - started,
            "model": metadata,
        }
        (args.output_dir / "e2fai_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
