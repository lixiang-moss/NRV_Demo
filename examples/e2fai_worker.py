#!/usr/bin/env python3
"""Persistent host GPU worker for ordered, sensor-timestamped ROS events.

Only the model is carried from E2FAI. This local protocol, ordered assembly and
session lifecycle are independent of its research transport and timestamp code.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import queue
import signal
import socket
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT / "ros_ws/src/nrv_demo/scripts"))

import numpy as np
import torch

from e2fai_protocol import EVENT_DTYPE, ProtocolError, recv_packet, send_packet
from nrv_e2fai import load_recurrent_model
from nrv_e2fai.preprocessing import EventWindowBuffer, NUM_BINS, RunningRange, voxelize
from nrv_e2fai.visualization import flow_hsv_rgb
from performance import PerfRecorder

DEFAULT_QUEUE_BATCHES = 64
DEFAULT_QUEUE_BYTES = 1024 * 1024 * 1024


class BoundedEventQueue:
    """FIFO with explicit overload failure; no drop-oldest or catch-up policy."""
    def __init__(self, max_batches=DEFAULT_QUEUE_BATCHES, max_bytes=DEFAULT_QUEUE_BYTES):
        self.max_batches, self.max_bytes = max_batches, max_bytes
        self.items = deque()
        self.bytes = 0
        self.peak_batches = self.peak_bytes = 0
        self.lock = threading.Lock()

    def put(self, item, size):
        with self.lock:
            if len(self.items) >= self.max_batches or self.bytes + size > self.max_bytes:
                raise OverflowError(
                    "Host event FIFO exceeded {} batches / {:g} MiB; session paused".format(
                        self.max_batches, self.max_bytes / (1024 * 1024)))
            self.items.append((item, size))
            self.bytes += size
            self.peak_batches = max(self.peak_batches, len(self.items))
            self.peak_bytes = max(self.peak_bytes, self.bytes)

    def get(self):
        with self.lock:
            if not self.items:
                return None
            item, size = self.items.popleft()
            self.bytes -= size
            return item

    def clear(self):
        with self.lock:
            batches = len(self.items)
            events = sum(len(item[0]) for item, _ in self.items)
            self.items.clear()
            self.bytes = 0
            return batches, events

    def snapshot(self):
        with self.lock:
            items = [item for item, _ in self.items]
            event_parts = [item[0] for item in items if len(item[0])]
            return dict(batches=len(items), bytes=self.bytes,
                        oldest_received_ns=items[0][2] if items else None,
                        oldest_callback_ns=items[0][1]["callback_ns"] if items else None,
                        source_span_ns=(int(event_parts[-1]["timestamp_ns"][-1])
                                        - int(event_parts[0]["timestamp_ns"][0])) if event_parts else 0)


def observe_queue(perf, fifo):
    if not perf.enabled:
        return
    snapshot = fifo.snapshot()
    now = time.monotonic_ns()
    perf.observe("host_fifo_batches", snapshot["batches"], unit="batches")
    perf.observe("host_fifo_bytes", snapshot["bytes"], unit="bytes")
    perf.observe("host_fifo_source_span", snapshot["source_span_ns"] / 1e6)
    for name, source in (("host_fifo_oldest_wait", "oldest_received_ns"),
                         ("host_fifo_oldest_callback_age", "oldest_callback_ns")):
        perf.observe(name, (now - snapshot[source]) / 1e6 if snapshot[source] is not None else 0.0)


def validate_events(metadata, payload):
    required = ("batch_seq", "ros_header_seq", "header_stamp_ns", "width", "height",
                "count", "callback_ns")
    session = metadata.get("session_id")
    if not isinstance(session, str) or not 0 < len(session) <= 128:
        raise ProtocolError("events requires a nonempty session_id of at most 128 characters")
    if any(type(metadata.get(key)) is not int or metadata[key] < 0 for key in required):
        raise ProtocolError("events requires nonnegative integer counters and timestamps")
    if metadata["ros_header_seq"] >= (1 << 32):
        raise ProtocolError("ROS header sequence must be uint32")
    if "ros_sequence_valid" in metadata and type(metadata["ros_sequence_valid"]) is not bool:
        raise ProtocolError("ros_sequence_valid must be a boolean")
    if not 0 < metadata["width"] <= 65535 or not 0 < metadata["height"] <= 65535:
        raise ProtocolError("Invalid event resolution")
    if len(payload) != metadata["count"] * EVENT_DTYPE.itemsize:
        raise ProtocolError("Event count does not match payload length")
    events = np.frombuffer(payload, dtype=EVENT_DTYPE)
    if np.any(events["polarity"] > 1):
        raise ProtocolError("Event polarity must be 0 or 1")
    return events


class SessionReceiver:
    def __init__(self, connection, perf, max_batches=DEFAULT_QUEUE_BATCHES, max_bytes=DEFAULT_QUEUE_BYTES):
        self.connection, self.perf = connection, perf
        self.queue = BoundedEventQueue(max_batches, max_bytes)
        self.send_lock = threading.Lock()
        self.done = threading.Event()
        self.failed = threading.Event()
        self.error = None
        self.session_id = None
        self.received_batches = self.received_events = 0
        self.rejected_batches = self.rejected_events = 0
        self.thread = threading.Thread(target=self._run, name="e2fai-event-reader", daemon=True)
        self.thread.start()

    def send(self, kind, metadata, payload=b""):
        with self.send_lock:
            if kind == "result" and (self.failed.is_set() or self.done.is_set()):
                return False
            send_packet(self.connection, kind, metadata, payload)
            return True

    def fail(self, code, message):
        if self.failed.is_set():
            return
        self.error = {"session_id": self.session_id, "code": code, "message": str(message)}
        self.failed.set()
        try:
            self.send("error", self.error)
        except (OSError, ValueError):
            pass
        try:
            self.connection.shutdown(socket.SHUT_RD)
        except OSError:
            pass
        self.done.set()

    def _run(self):
        try:
            while not self.done.is_set():
                packet = recv_packet(self.connection)
                if packet is None:
                    break
                kind, metadata, payload = packet
                if kind != "events":
                    raise ProtocolError("Host worker accepts events packets only")
                tick = self.perf.tick()
                events = validate_events(metadata, payload)
                if self.session_id is None:
                    self.session_id = metadata["session_id"]
                elif self.session_id != metadata["session_id"]:
                    raise ProtocolError("A TCP connection cannot mix acquisition sessions")
                self.perf.elapsed("bridge_unpack", tick, events=len(events),
                                  batch_seq=metadata["batch_seq"])
                received_ns = time.monotonic_ns() if self.perf.enabled else 0
                self.received_batches += 1
                self.received_events += len(events)
                if self.perf.enabled:
                    self.perf.observe("callback_to_host_receive", (received_ns - metadata["callback_ns"]) / 1e6,
                                      events=len(events), batch_seq=metadata["batch_seq"])
                self.queue.put((events, metadata, received_ns), len(payload))
                observe_queue(self.perf, self.queue)
        except OverflowError as error:
            self.rejected_batches += 1
            self.rejected_events += len(events)
            self.fail("queue_overflow", error)
        except (OSError, EOFError, ValueError) as error:
            if not self.done.is_set():
                self.fail("protocol_or_connection_error", error)
        finally:
            self.done.set()

    def close(self):
        self.done.set()
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.connection.close()
        self.thread.join(timeout=2)
        return self.queue.clear()


def snapshot_output(output):
    """Finish CUDA copies on the inference thread; result jobs own CPU arrays."""
    return (output["log_image"][0, 0].detach().float().cpu().numpy(),
            output["flow"][0].detach().float().cpu().numpy())


def render_result(snapshot, image_range, flow_max_px):
    log_image, flow_chw = snapshot
    gray = image_range.render(log_image)
    color = flow_hsv_rgb(flow_chw[None, None], flow_max_px)[0, 0]
    flow = np.ascontiguousarray(flow_chw.transpose(1, 2, 0), dtype="<f4")
    # Protocol payload has exactly H*W*(1 + 3 + 8) bytes and no image compression.
    return gray.tobytes() + color.tobytes() + flow.tobytes()


class ResultProcessor:
    """One CPU result consumer; the FIFO holds at most one waiting result."""
    def __init__(self, receiver, perf, flow_max_px, mode, should_stop):
        self.receiver, self.perf = receiver, perf
        self.flow_max_px, self.mode, self.should_stop = flow_max_px, mode, should_stop
        self.queue = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.image_range = RunningRange()
        self.generation = None
        self.result_count = self.queue_peak = 0
        self.cancelled_results = self.cancelled_events = 0
        self.thread = None
        if mode == "thread":
            self.thread = threading.Thread(target=self._run, name="e2fai-results", daemon=True)
            self.thread.start()
        elif mode != "inline":
            raise ValueError("Result mode must be inline or thread")

    def _cancelled(self):
        return (self.stopping.is_set() or self.receiver.done.is_set()
                or self.receiver.failed.is_set() or self.should_stop())

    def submit(self, snapshot, metadata, events, copy_ms=0):
        job = dict(snapshot=snapshot, metadata=dict(metadata), events=events,
                   copy_ms=copy_ms, enqueued_ns=0)
        tick = self.perf.tick()
        if self.mode == "inline":
            return self._process(job)
        while not self._cancelled():
            try:
                # Only the successful nonblocking put supplies the timestamp;
                # earlier full-queue waits are measured separately.
                job["enqueued_ns"] = self.perf.tick()
                self.queue.put_nowait(job)
                self.queue_peak = 1
                self.perf.elapsed("result_handoff_wait", tick, events=events,
                                  window_id=metadata["window_id"])
                return True
            except queue.Full:
                self.stopping.wait(0.01)
        return False

    def _process(self, job):
        if self._cancelled():
            self.cancelled_results += 1
            self.cancelled_events += job["events"]
            return False
        metadata, events = job["metadata"], job["events"]
        window_id = metadata["window_id"]
        if job["enqueued_ns"]:
            self.perf.elapsed("result_queue_residence", job["enqueued_ns"],
                              events=events, window_id=window_id)
        if metadata["reset_count"] != self.generation:
            self.image_range = RunningRange()
            self.generation = metadata["reset_count"]
        tick = self.perf.tick()
        payload = render_result(job["snapshot"], self.image_range, self.flow_max_px)
        if self.perf.enabled:
            render_ms = (time.perf_counter_ns() - tick) / 1e6
            self.perf.observe("result_cpu_render_and_pack", render_ms,
                              events=events, window_id=window_id)
            # Keep execution cost comparable with the old combined operation:
            # CPU copies + rendering/packing, excluding result-queue waiting.
            self.perf.observe("result_render_and_pack", job["copy_ms"] + render_ms,
                              events=events, window_id=window_id)
        if self._cancelled():
            self.cancelled_results += 1
            self.cancelled_events += events
            return False
        metadata.update(completed_ns=time.monotonic_ns(),
                        flow_units="model_input_pixels_per_window",
                        timestamp_source="group_aer_corrected_sensor_clock_ros_anchor")
        tick = self.perf.tick()
        try:
            sent = self.receiver.send("result", metadata, payload)
        except (OSError, ValueError):
            self.cancelled_results += 1
            self.cancelled_events += events
            raise
        if not sent:
            self.cancelled_results += 1
            self.cancelled_events += events
            return False
        self.perf.elapsed("result_send", tick, events=events, window_id=window_id)
        if self.perf.enabled:
            self.perf.observe("callback_to_result_send",
                              (time.monotonic_ns() - metadata["source_callback_ns"]) / 1e6,
                              events=events, window_id=window_id)
            self.perf.observe("window_complete", 1, unit="window", events=events,
                              **{key: metadata[key] for key in (
                                  "session_id", "window_id", "source_batch_first", "source_batch_last",
                                  "window_start_ns", "window_end_ns", "event_first_ns", "event_last_ns",
                                  "width", "height", "reset_count", "reset_reason", "source_callback_ns",
                                  "completed_ns")})
        self.result_count += 1
        return True

    def _run(self):
        try:
            while not self._cancelled():
                try:
                    job = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if not self._process(job):
                    break
        except (OSError, ValueError, RuntimeError) as error:
            if not self._cancelled():
                self.receiver.fail("inference_or_send_error", error)
        finally:
            self.stopping.set()

    def close(self):
        """The session owner closes its socket first, releasing blocked sends."""
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError("Result thread did not stop after connection shutdown")
        while True:
            try:
                job = self.queue.get_nowait()
            except queue.Empty:
                break
            self.cancelled_results += 1
            self.cancelled_events += job["events"]
        self.image_range = RunningRange()
        return dict(pending_results_cleared_on_disconnect=self.cancelled_results,
                    pending_result_events_cleared_on_disconnect=self.cancelled_events,
                    result_queue_peak=self.queue_peak)


def run_session(connection, model, device, args, perf, should_stop):
    receiver = SessionReceiver(connection, perf, max_batches=args.queue_batches)
    windows = EventWindowBuffer(round(args.window_ms * 1e6))
    state = None
    state_generation = None
    results = ResultProcessor(receiver, perf, args.flow_max_px,
                              getattr(args, "result_mode", "thread"), should_stop)
    inference_count = input_events = small_windows = small_window_events = 0
    pending_window_events = 0
    summary = {}
    try:
        while not should_stop() and not receiver.done.is_set():
            item = receiver.queue.get()
            if item is None:
                receiver.done.wait(0.01)
                continue
            events, metadata, received_ns = item
            observe_queue(perf, receiver.queue)
            if (metadata["width"], metadata["height"]) != (args.sensor_width, args.sensor_height):
                receiver.fail("resolution_mismatch", "Event resolution does not match model configuration")
                break
            if perf.enabled:
                perf.observe("host_fifo_wait", (time.monotonic_ns() - received_ns) / 1e6,
                             events=len(events), batch_seq=metadata["batch_seq"])
            tick = perf.tick()
            previous_ros_gaps = windows.ros_sequence_gap_incidents
            assembled = windows.push(events, metadata)
            perf.increment("ros_sequence_gap_incidents",
                           windows.ros_sequence_gap_incidents - previous_ros_gaps)
            pending_window_events = sum(len(window.events) for window in assembled)
            perf.elapsed("window_assembly", tick, events=len(events), batch_seq=metadata["batch_seq"])
            for window in assembled:
                if receiver.done.is_set() or should_stop():
                    break
                if len(window.events) < 2:
                    small_windows += 1
                    small_window_events += len(window.events)
                    pending_window_events -= len(window.events)
                    continue
                generation = window.metadata["reset_count"]
                if generation != state_generation:
                    state = None
                    state_generation = generation
                tick = perf.tick()
                window_tick = tick
                voxel = voxelize(window.events, args.sensor_width, args.sensor_height, device,
                                 args.sensor_width, args.sensor_height)
                if perf.enabled and device.type == "cuda":
                    torch.cuda.synchronize(device)
                perf.elapsed("voxel_build", tick, events=len(window.events),
                             window_id=window.metadata["window_id"])
                gpu_start = gpu_end = None
                if perf.enabled and device.type == "cuda":
                    gpu_start = torch.cuda.Event(enable_timing=True)
                    gpu_end = torch.cuda.Event(enable_timing=True)
                    gpu_start.record()
                tick = perf.tick()
                with torch.inference_mode():
                    output, state = model.forward_step(voxel, state)
                perf.elapsed("model_submit", tick, events=len(window.events),
                             window_id=window.metadata["window_id"])
                if gpu_end is not None:
                    gpu_end.record()
                tick = perf.tick()
                finite = all(torch.isfinite(output[key]).all() for key in ("log_image", "flow"))
                perf.elapsed("output_ready_and_finite_check", tick, events=len(window.events),
                             window_id=window.metadata["window_id"])
                if gpu_end is not None:
                    # The finite checks already require CPU access; synchronization
                    # is performed only with measurements enabled to read CUDA events.
                    gpu_end.synchronize()
                    perf.observe("model_cuda_forward", gpu_start.elapsed_time(gpu_end),
                                 events=len(window.events), window_id=window.metadata["window_id"])
                if not finite:
                    receiver.fail("nonfinite_output", "Model produced nonfinite output; session paused")
                    break
                state = state.detach()
                inference_count += 1
                input_events += len(window.events)
                pending_window_events -= len(window.events)
                tick = perf.tick()
                snapshot = snapshot_output(output)
                copy_ms = (time.perf_counter_ns() - tick) / 1e6 if perf.enabled else 0
                perf.observe("result_cpu_copy", copy_ms, events=len(window.events),
                             window_id=window.metadata["window_id"])
                perf.elapsed("inference_to_cpu_snapshot", window_tick, events=len(window.events),
                             window_id=window.metadata["window_id"])
                if not results.submit(snapshot, window.metadata, len(window.events), copy_ms):
                    break
    except OverflowError as error:
        receiver.fail("window_buffer_overflow", error)
    except (OSError, ValueError, RuntimeError) as error:
        receiver.fail("inference_or_send_error", error)
    finally:
        discarded_batches, discarded_events = receiver.close()
        summary.update(results.close())
        summary.update(session_id=receiver.session_id, results=results.result_count,
                       result_mode=results.mode,
                       inferred_windows=inference_count,
                       inferred_events=input_events, received_batches=receiver.received_batches,
                       received_events=receiver.received_events,
                       overload_rejected_batches=receiver.rejected_batches,
                       overload_rejected_events=receiver.rejected_events,
                       pending_batches_cleared_on_disconnect=discarded_batches,
                       pending_events_cleared_on_disconnect=discarded_events,
                       partial_window_events_at_disconnect=windows.buffered_events,
                       partial_events_cleared_on_time_reset=windows.discarded_partial_events,
                       complete_window_events_cleared_on_disconnect=pending_window_events,
                       reset_count=windows.reset_count, last_reset_reason=windows.reset_reason,
                       ros_sequence_tracking=windows.ros_sequence_tracking,
                       ros_sequence_gap_incidents=windows.ros_sequence_gap_incidents,
                       ros_sequence_forward_missing=windows.ros_sequence_forward_missing,
                       ros_sequence_unconfirmed_batches=windows.ros_sequence_unknown_batches,
                       fewer_than_two_event_windows=small_windows,
                       fewer_than_two_event_window_events=small_window_events,
                       empty_windows=windows.empty_windows,
                       queue_peak_batches=receiver.queue.peak_batches,
                       queue_peak_bytes=receiver.queue.peak_bytes, error=receiver.error)
        # Session locals own all event chunks, display normalization and GRU state.
        state = None
        windows.parts.clear()
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--window-ms", type=float, default=250.0)
    parser.add_argument("--result-mode", choices=("inline", "thread"), default="thread",
                        help="CPU rendering/packing and TCP sending mode (default: thread)")
    parser.add_argument("--queue-batches", type=int, default=DEFAULT_QUEUE_BATCHES,
                        help="Host FIFO batch limit; byte limit is 1 GiB (1024 MiB)")
    parser.add_argument("--sensor-width", type=int, default=960)
    parser.add_argument("--sensor-height", type=int, default=720)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--image-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flow-max-px", type=float, default=None,
                        help="Preview saturation in pixels/window (default: 200 pixels/s times window length)")
    parser.add_argument("--perf-enabled", action="store_true")
    parser.add_argument("--perf-interval", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.flow_max_px is None:
        args.flow_max_px = 200.0 * args.window_ms / 1000.0
    if args.host != "127.0.0.1":
        parser.error("The integration is local-only; --host must be 127.0.0.1")
    if (not 0 < args.port < 65536 or args.window_ms <= 0 or args.flow_max_px <= 0
            or args.queue_batches < 1
            or min(args.sensor_width, args.sensor_height) <= 0
            or args.sensor_width % 16 or args.sensor_height % 16):
        parser.error("Port, timing, queue batch limit, flow scale or native dimensions are invalid")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ready = args.output_dir / "ready"
    ready.unlink(missing_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, model_metadata = load_recurrent_model(
        args.image_checkpoint, backbone_checkpoint=args.backbone, device=device,
        sensor_height=args.sensor_height, sensor_width=args.sensor_width)
    torch.backends.cudnn.benchmark = False
    with torch.inference_mode():
        model.forward_step(torch.zeros(1, NUM_BINS, args.sensor_height, args.sensor_width, device=device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stop = threading.Event()
    active = [None]
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    server.settimeout(0.25)
    perf = PerfRecorder(args.output_dir, "e2fai_worker", enabled=args.perf_enabled,
                        interval_s=args.perf_interval)

    def request_stop(_signum, _frame):
        stop.set()
        if active[0] is not None:
            try:
                active[0].shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    ready.write_text("{}:{}\n".format(args.host, args.port))
    print("E2FAI model ready on {}:{}".format(args.host, args.port), flush=True)
    sessions = 0
    try:
        while not stop.is_set():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            connection.settimeout(1.0)
            active[0] = connection
            summary = run_session(connection, model, device, args, perf, stop.is_set)
            active[0] = None
            sessions += 1
            summary.update(model=model_metadata, window_ms=args.window_ms,
                           queue_max_batches=args.queue_batches, queue_max_bytes=DEFAULT_QUEUE_BYTES,
                           resolution=[args.sensor_width, args.sensor_height],
                           event_subsampling_enabled=False, perf_enabled=args.perf_enabled,
                           timestamp_source="group_aer_corrected_sensor_clock_ros_anchor")
            token = hashlib.sha256(str(summary["session_id"]).encode()).hexdigest()[:12]
            filename = args.output_dir / "e2fai_session_{:03d}_{}.json".format(sessions, token)
            filename.write_text(json.dumps(summary, indent=2) + "\n")
            print("E2FAI session ended: {} results; error={}".format(summary["results"], summary["error"]), flush=True)
    finally:
        server.close()
        perf.close()
        ready.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
