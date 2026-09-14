"""Model inputs assembled from unchanged, ordered ROS event timestamps."""
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

NUM_BINS = 15


@dataclass
class EventWindow:
    events: np.ndarray
    metadata: dict


@dataclass
class RoutedWindow:
    """Metadata for one incrementally completed half-open event window."""
    metadata: dict


class EventWindowBuffer:
    """200 ms half-open windows; discontinuities reset, never sort or retime.

    Partial windows discarded at a discontinuity are counted explicitly. The
    model resets when a subsequent window carries a changed reset_count.
    """
    def __init__(self, window_ns=200_000_000, max_gap_ns=300_000_000, max_buffer_bytes=256 * 1024 * 1024):
        if window_ns <= 0 or max_gap_ns <= 0:
            raise ValueError("Window and gap limits must be positive")
        self.window_ns = int(window_ns)
        self.max_gap_ns = int(max_gap_ns)
        self.max_buffer_bytes = max_buffer_bytes
        self.parts = deque()
        self.start_ns = None
        self.last_event_ns = None
        self.shape = None
        self.previous_batch = None
        self.previous_ros_sequence = None
        self.ros_sequence_tracking = "unconfirmed"
        self.ros_sequence_nonzero_seen = False
        self.ros_sequence_gap_incidents = 0
        self.ros_sequence_forward_missing = 0
        self.ros_sequence_unknown_batches = 0
        self.window_id = 0
        self.reset_count = 0
        self.reset_reason = "session_start"
        self.discarded_partial_events = 0
        self.empty_windows = 0

    @property
    def buffered_events(self):
        return sum(len(events) for events, _ in self.parts)

    def reset(self, reason):
        self.discarded_partial_events += self.buffered_events
        self.parts.clear()
        self.start_ns = None
        self.last_event_ns = None
        self.reset_count += 1
        self.reset_reason = reason

    def _observe_ros_sequence(self, metadata):
        # A bridge must first establish that its publisher supplies ROS message
        # sequence numbers. This is a publication counter, never a RAW packet ID.
        if metadata.get("ros_sequence_valid") is not True:
            self.previous_ros_sequence = None
            self.ros_sequence_tracking = "unconfirmed"
            self.ros_sequence_unknown_batches += 1
            return False
        sequence = metadata["ros_header_seq"]
        previous = self.previous_ros_sequence
        self.previous_ros_sequence = sequence
        if not self.ros_sequence_nonzero_seen and sequence == 0:
            self.ros_sequence_tracking = "unconfirmed_constant_zero"
            self.ros_sequence_unknown_batches += 1
            return False
        self.ros_sequence_nonzero_seen = True
        self.ros_sequence_tracking = "active"
        if previous is None or sequence == (previous + 1) % (1 << 32):
            return False
        self.ros_sequence_gap_incidents += 1
        distance = (sequence - previous) % (1 << 32)
        if 1 < distance < (1 << 31):
            self.ros_sequence_forward_missing += distance - 1
        return True

    def push(self, events, metadata):
        shape = (metadata["width"], metadata["height"])
        batch = metadata["batch_seq"]
        ros_sequence_gap = self._observe_ros_sequence(metadata)
        if self.shape is not None and shape != self.shape:
            self.reset("resolution_changed")
        elif ros_sequence_gap:
            self.reset("ros_sequence_gap")
        elif self.previous_batch is not None and batch != self.previous_batch + 1:
            self.reset("batch_sequence_gap")
        self.shape, self.previous_batch = shape, batch
        if not len(events):
            return []
        timestamps = events["timestamp_ns"]
        # Compare uint64 values directly; subtraction before comparison can wrap.
        previous = timestamps[:-1]
        current = timestamps[1:]
        backwards = current < previous
        forward = (~backwards) & ((current - previous) > self.max_gap_ns)
        cuts = np.flatnonzero(backwards | forward) + 1
        windows = []
        start = 0
        for stop in list(cuts) + [len(events)]:
            segment = events[start:stop]
            first_ns = int(segment["timestamp_ns"][0])
            if self.last_event_ns is not None:
                if first_ns < self.last_event_ns:
                    self.reset("event_time_reversed")
                elif first_ns - self.last_event_ns > self.max_gap_ns:
                    self.reset("event_time_gap")
            windows.extend(self._append_segment(segment, metadata))
            start = stop
        return windows

    def _append_segment(self, events, metadata):
        if (self.buffered_events + len(events)) * events.dtype.itemsize > self.max_buffer_bytes:
            raise OverflowError("Event window buffer exceeded its byte limit")
        if self.start_ns is None:
            self.start_ns = int(events["timestamp_ns"][0])
        self.parts.append((events, metadata))
        self.last_event_ns = int(events["timestamp_ns"][-1])
        complete = []
        while self.last_event_ns >= self.start_ns + self.window_ns:
            end_ns = self.start_ns + self.window_ns
            pieces, sources = [], []
            while self.parts and int(self.parts[0][0]["timestamp_ns"][0]) < end_ns:
                part, source = self.parts.popleft()
                cut = int(np.searchsorted(part["timestamp_ns"], end_ns, side="left"))
                if cut:
                    pieces.append(part[:cut])
                    sources.append(source)
                if cut < len(part):
                    self.parts.appendleft((part[cut:], source))
                    break
            if pieces:
                joined = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
                self.window_id += 1
                complete.append(EventWindow(joined, {
                    "session_id": metadata["session_id"],
                    "window_id": self.window_id,
                    "source_batch_first": sources[0]["batch_seq"],
                    "source_batch_last": sources[-1]["batch_seq"],
                    "source_ros_header_first": sources[0]["ros_header_seq"],
                    "source_ros_header_last": sources[-1]["ros_header_seq"],
                    "window_start_ns": self.start_ns,
                    "window_end_ns": end_ns,
                    "event_first_ns": int(joined["timestamp_ns"][0]),
                    "event_last_ns": int(joined["timestamp_ns"][-1]),
                    "event_count": len(joined),
                    "width": metadata["width"], "height": metadata["height"],
                    "reset_reason": self.reset_reason,
                    "reset_count": self.reset_count,
                    "source_callback_ns": sources[-1]["callback_ns"],
                }))
            else:
                self.empty_windows += 1
            self.start_ns = end_ns
        return complete


def voxelize(events, source_width, source_height, device, sensor_width, sensor_height):
    """E2FAI's full-event native-resolution 15-bin voxel construction.

    This normalizes original integer timestamp differences into model bins; it
    does not modify source timestamps or synthesize timestamps from packet heads.
    """
    if len(events) < 2:
        raise ValueError("A voxel needs at least two events")
    if (source_width, source_height) != (sensor_width, sensor_height):
        raise ValueError("Camera resolution does not match configured sensor")
    if np.any(events["timestamp_ns"][1:] < events["timestamp_ns"][:-1]):
        raise ValueError("Voxel events must retain monotonic source order")
    x = torch.tensor(events["x"].astype(np.int32), device=device)
    y = torch.tensor(events["y"].astype(np.int32), device=device)
    valid_xy = (x >= 0) & (x < source_width) & (y >= 0) & (y < source_height)
    relative_ns = events["timestamp_ns"].copy() - events["timestamp_ns"][0]
    if relative_ns[-1] == 0:
        normalized_time = torch.linspace(0, NUM_BINS - 1, len(events),
                                         device=device, dtype=torch.float32)
    else:
        normalized_time = torch.tensor(
            relative_ns.astype(np.float32) / float(relative_ns[-1]), device=device
        ) * (NUM_BINS - 1)
    polarity = torch.tensor(events["polarity"].copy(), device=device, dtype=torch.float32)
    t0 = normalized_time.int()
    signed = 2 * polarity - 1
    plane_size = sensor_height * sensor_width
    voxel = torch.zeros(NUM_BINS * plane_size, dtype=torch.float32, device=device)
    pixel_index = sensor_width * y + x
    for ti in (t0, t0 + 1):
        valid = valid_xy & (ti >= 0) & (ti < NUM_BINS)
        index = (plane_size * ti + pixel_index).clamp(0, NUM_BINS * plane_size - 1).long()
        weight = signed * (1 - (ti - normalized_time).abs())
        voxel.scatter_add_(0, index, weight * valid)
    return voxel.view(1, NUM_BINS, sensor_height, sensor_width)


def accumulate_fixed_window(voxel, x, y, timestamp_ns, polarity,
                            window_start_ns, window_ns, sensor_width, sensor_height):
    """Accumulate one already-routed event chunk into a fixed-window voxel."""
    if voxel.ndim != 1 or voxel.numel() != NUM_BINS * sensor_width * sensor_height:
        raise ValueError("Fixed-window voxel storage has the wrong shape")
    valid_xy = (x >= 0) & (x < sensor_width) & (y >= 0) & (y < sensor_height)
    relative_ns = timestamp_ns.to(torch.int64) - int(window_start_ns)
    valid_time = (relative_ns >= 0) & (relative_ns < int(window_ns))
    normalized_time = relative_ns.to(torch.float32) * (
        float(NUM_BINS - 1) / float(window_ns))
    t0 = torch.floor(normalized_time).to(torch.int64)
    signed = polarity.to(torch.float32).mul(2).sub(1)
    plane_size = sensor_height * sensor_width
    pixel_index = sensor_width * y.to(torch.int64) + x.to(torch.int64)
    for ti in (t0, t0 + 1):
        valid = valid_xy & valid_time & (ti >= 0) & (ti < NUM_BINS)
        index = (plane_size * ti + pixel_index).clamp(
            0, NUM_BINS * plane_size - 1).to(torch.int64)
        weight = signed * (1 - (ti.to(torch.float32) - normalized_time).abs())
        voxel.scatter_add_(0, index, weight * valid)


def voxelize_fixed_window(events, source_width, source_height, device,
                           sensor_width, sensor_height, window_start_ns, window_end_ns):
    """One-shot correctness reference for fixed-window incremental voxelization."""
    if window_end_ns <= window_start_ns:
        raise ValueError("Fixed voxel window must have a positive duration")
    if (source_width, source_height) != (sensor_width, sensor_height):
        raise ValueError("Camera resolution does not match configured sensor")
    if np.any(events["timestamp_ns"][1:] < events["timestamp_ns"][:-1]):
        raise ValueError("Voxel events must retain monotonic source order")
    if len(events) and (int(events["timestamp_ns"][0]) < window_start_ns
                        or int(events["timestamp_ns"][-1]) >= window_end_ns):
        raise ValueError("Fixed-window events must lie inside the half-open window")
    plane_size = sensor_height * sensor_width
    voxel = torch.zeros(NUM_BINS * plane_size, dtype=torch.float32, device=device)
    if len(events):
        x = torch.as_tensor(np.ascontiguousarray(events["x"], dtype=np.int32), device=device)
        y = torch.as_tensor(np.ascontiguousarray(events["y"], dtype=np.int32), device=device)
        timestamp_ns = torch.as_tensor(
            np.ascontiguousarray(events["timestamp_ns"], dtype=np.int64), device=device)
        polarity = torch.as_tensor(
            np.ascontiguousarray(events["polarity"], dtype=np.uint8), device=device)
        accumulate_fixed_window(
            voxel, x, y, timestamp_ns, polarity, window_start_ns,
            window_end_ns - window_start_ns, sensor_width, sensor_height)
    return voxel.view(1, NUM_BINS, sensor_height, sensor_width)


class IncrementalWindowRouter:
    """Route ordered event batches into fixed windows without retaining full windows."""

    def __init__(self, window_ns, max_gap_ns=300_000_000):
        if window_ns <= 0 or max_gap_ns <= 0:
            raise ValueError("Window and gap limits must be positive")
        self.window_ns = int(window_ns)
        self.max_gap_ns = int(max_gap_ns)
        self.start_ns = None
        self.last_event_ns = None
        self.shape = None
        self.previous_batch = None
        self.previous_ros_sequence = None
        self.ros_sequence_tracking = "unconfirmed"
        self.ros_sequence_nonzero_seen = False
        self.ros_sequence_gap_incidents = 0
        self.ros_sequence_forward_missing = 0
        self.ros_sequence_unknown_batches = 0
        self.window_id = 0
        self.reset_count = 0
        self.reset_reason = "session_start"
        self.discarded_partial_events = 0
        self.empty_windows = 0
        self._clear_window_metadata()

    @property
    def buffered_events(self):
        return self.event_count

    def _clear_window_metadata(self):
        self.event_count = 0
        self.event_first_ns = None
        self.event_last_ns = None
        self.source_batch_first = None
        self.source_batch_last = None
        self.source_ros_header_first = None
        self.source_ros_header_last = None
        self.source_callback_ns = None
        self.input_complete_ns = None
        self.session_id = None
        self.width = self.height = None

    def reset(self, reason, discard_current):
        if self.event_count:
            self.discarded_partial_events += self.event_count
            discard_current(self.event_count)
        self.start_ns = None
        self.last_event_ns = None
        self._clear_window_metadata()
        self.reset_count += 1
        self.reset_reason = reason

    def _observe_ros_sequence(self, metadata):
        if metadata.get("ros_sequence_valid") is not True:
            self.previous_ros_sequence = None
            self.ros_sequence_tracking = "unconfirmed"
            self.ros_sequence_unknown_batches += 1
            return False
        sequence = metadata["ros_header_seq"]
        previous = self.previous_ros_sequence
        self.previous_ros_sequence = sequence
        if not self.ros_sequence_nonzero_seen and sequence == 0:
            self.ros_sequence_tracking = "unconfirmed_constant_zero"
            self.ros_sequence_unknown_batches += 1
            return False
        self.ros_sequence_nonzero_seen = True
        self.ros_sequence_tracking = "active"
        if previous is None or sequence == (previous + 1) % (1 << 32):
            return False
        self.ros_sequence_gap_incidents += 1
        distance = (sequence - previous) % (1 << 32)
        if 1 < distance < (1 << 31):
            self.ros_sequence_forward_missing += distance - 1
        return True

    def push(self, events, metadata, append_chunk, finish_window, discard_current):
        shape = (metadata["width"], metadata["height"])
        batch = metadata["batch_seq"]
        ros_sequence_gap = self._observe_ros_sequence(metadata)
        if self.shape is not None and shape != self.shape:
            self.reset("resolution_changed", discard_current)
        elif ros_sequence_gap:
            self.reset("ros_sequence_gap", discard_current)
        elif self.previous_batch is not None and batch != self.previous_batch + 1:
            self.reset("batch_sequence_gap", discard_current)
        self.shape, self.previous_batch = shape, batch
        if not len(events):
            return
        timestamps = events["timestamp_ns"]
        previous = timestamps[:-1]
        current = timestamps[1:]
        backwards = current < previous
        forward = (~backwards) & ((current - previous) > self.max_gap_ns)
        cuts = np.flatnonzero(backwards | forward) + 1
        offset = 0
        for stop in list(cuts) + [len(events)]:
            segment = events[offset:stop]
            first_ns = int(segment["timestamp_ns"][0])
            if self.last_event_ns is not None:
                if first_ns < self.last_event_ns:
                    self.reset("event_time_reversed", discard_current)
                elif first_ns - self.last_event_ns > self.max_gap_ns:
                    self.reset("event_time_gap", discard_current)
            self._append_segment(segment, metadata, append_chunk, finish_window)
            offset = stop

    def _append_segment(self, events, metadata, append_chunk, finish_window):
        if self.start_ns is None:
            self.start_ns = int(events["timestamp_ns"][0])
        offset = 0
        while offset < len(events):
            end_ns = self.start_ns + self.window_ns
            timestamp = int(events["timestamp_ns"][offset])
            while timestamp >= end_ns:
                if self.event_count:
                    finish_window(RoutedWindow(self._completed_metadata(end_ns)))
                else:
                    self.empty_windows += 1
                self.start_ns = end_ns
                end_ns = self.start_ns + self.window_ns
            cut = int(np.searchsorted(events["timestamp_ns"], end_ns,
                                      side="left", sorter=None))
            cut = max(cut, offset)
            if cut == offset:
                continue
            piece = events[offset:cut]
            if not self.event_count:
                self.session_id = metadata["session_id"]
                self.event_first_ns = int(piece["timestamp_ns"][0])
                self.source_batch_first = metadata["batch_seq"]
                self.source_ros_header_first = metadata["ros_header_seq"]
                self.width, self.height = metadata["width"], metadata["height"]
            self.event_count += len(piece)
            self.event_last_ns = int(piece["timestamp_ns"][-1])
            self.source_batch_last = metadata["batch_seq"]
            self.source_ros_header_last = metadata["ros_header_seq"]
            self.source_callback_ns = metadata["callback_ns"]
            self.input_complete_ns = metadata.get("input_complete_ns")
            append_chunk(piece, self.start_ns, end_ns)
            offset = cut
        self.last_event_ns = int(events["timestamp_ns"][-1])

    def _completed_metadata(self, end_ns):
        self.window_id += 1
        result = {
            "session_id": self.session_id,
            "window_id": self.window_id,
            "source_batch_first": self.source_batch_first,
            "source_batch_last": self.source_batch_last,
            "source_ros_header_first": self.source_ros_header_first,
            "source_ros_header_last": self.source_ros_header_last,
            "window_start_ns": self.start_ns,
            "window_end_ns": end_ns,
            "event_first_ns": self.event_first_ns,
            "event_last_ns": self.event_last_ns,
            "event_count": self.event_count,
            "width": self.width,
            "height": self.height,
            "reset_reason": self.reset_reason,
            "reset_count": self.reset_count,
            "source_callback_ns": self.source_callback_ns,
            "input_complete_ns": self.input_complete_ns,
        }
        self._clear_window_metadata()
        return result


class RunningRange:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.low = self.high = None

    def render(self, values):
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
        return (np.nan_to_num(gray) * 255).round().astype(np.uint8)
