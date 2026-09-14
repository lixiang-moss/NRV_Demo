"""Fixed-window incremental voxel construction with a three-slot GPU pool."""
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time

import numpy as np
import torch

from .preprocessing import NUM_BINS, accumulate_fixed_window


@dataclass
class _StagingSlot:
    cpu_x: torch.Tensor
    cpu_y: torch.Tensor
    cpu_timestamp: torch.Tensor
    cpu_polarity: torch.Tensor
    gpu_x: torch.Tensor
    gpu_y: torch.Tensor
    gpu_timestamp: torch.Tensor
    gpu_polarity: torch.Tensor
    done_event: object = None
    in_use: bool = False
    count: int = 0
    first_wait_ns: int = 0
    window_id: int = 0
    timing_start: object = None
    timing_end: object = None


@dataclass
class VoxelSlot:
    index: int
    voxel: torch.Tensor
    state: str = "FREE"
    metadata: dict = None
    generation: int = 0
    event_count: int = 0
    window_start_ns: int = 0
    window_end_ns: int = 0
    fill_done_event: object = None
    inference_done_event: object = None
    voxel_start_event: object = None
    model_start_event: object = None
    close_tick_ns: int = 0
    model_submit_tick_ns: int = 0


class IncrementalVoxelizer:
    """Own pinned microbatches, CUDA streams/events and three reusable voxels."""

    def __init__(self, width, height, window_ns, device, perf,
                 pool_size=3, staging_slots=2, microbatch_max_events=65536,
                 microbatch_max_wait_ms=2.0, should_stop=lambda: False):
        if pool_size != 3 or staging_slots < 2 or microbatch_max_events < 1:
            raise ValueError("Incremental voxelizer requires three voxels and at least two staging slots")
        self.width, self.height = int(width), int(height)
        self.window_ns = int(window_ns)
        self.device = torch.device(device)
        self.perf = perf
        self.max_events = int(microbatch_max_events)
        self.max_wait_ns = round(float(microbatch_max_wait_ms) * 1e6)
        self.should_stop = should_stop
        self.cuda = self.device.type == "cuda"
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.ready_slots = deque()
        self.slots = []
        self.staging = []
        self.current = None
        self.current_staging = None
        self.next_staging = 0
        self.closed = False
        self.microbatch_count = 0
        self.pool_exhaustions = 0
        self.pool_peak_slots = 0
        self.invalidated_windows = 0
        self.voxel_stream = torch.cuda.Stream(device=self.device) if self.cuda else None
        self.inference_stream = torch.cuda.Stream(device=self.device) if self.cuda else None
        shape = (1, NUM_BINS, self.height, self.width)
        for index in range(pool_size):
            voxel = torch.zeros(shape, dtype=torch.float32, device=self.device)
            fill_done = (torch.cuda.Event(enable_timing=self.perf.enabled)
                         if self.cuda else None)
            self.slots.append(VoxelSlot(index=index, voxel=voxel,
                                        fill_done_event=fill_done))
        for _ in range(staging_slots):
            options = dict(pin_memory=self.cuda)
            cpu_x = torch.empty(self.max_events, dtype=torch.int32, **options)
            cpu_y = torch.empty(self.max_events, dtype=torch.int32, **options)
            cpu_timestamp = torch.empty(self.max_events, dtype=torch.int64, **options)
            cpu_polarity = torch.empty(self.max_events, dtype=torch.uint8, **options)
            self.staging.append(_StagingSlot(
                cpu_x, cpu_y, cpu_timestamp, cpu_polarity,
                torch.empty_like(cpu_x, device=self.device),
                torch.empty_like(cpu_y, device=self.device),
                torch.empty_like(cpu_timestamp, device=self.device),
                torch.empty_like(cpu_polarity, device=self.device),
                torch.cuda.Event() if self.cuda else None))
        if self.cuda:
            with torch.cuda.device(self.device), torch.cuda.stream(self.voxel_stream):
                for slot in self.slots:
                    slot.voxel.zero_()
            self.voxel_stream.synchronize()

    def _occupied(self):
        return sum(slot.state != "FREE" for slot in self.slots)

    def _update_peak(self):
        self.pool_peak_slots = max(self.pool_peak_slots, self._occupied())

    def _acquire_slot(self, generation, window_start_ns, window_end_ns):
        exhausted_recorded = False
        with self.condition:
            while not self.closed and not self.should_stop():
                slot = next((value for value in self.slots if value.state == "FREE"), None)
                if slot is not None:
                    slot.state = "FILLING"
                    slot.metadata = None
                    slot.generation = int(generation)
                    slot.event_count = 0
                    slot.window_start_ns = int(window_start_ns)
                    slot.window_end_ns = int(window_end_ns)
                    slot.voxel_start_event = None
                    slot.model_start_event = None
                    slot.close_tick_ns = 0
                    slot.model_submit_tick_ns = 0
                    self._update_peak()
                    return slot
                if not exhausted_recorded:
                    self.pool_exhaustions += 1
                    exhausted_recorded = True
                    if self.perf.enabled:
                        self.perf.increment("voxel_pool_exhaustions")
                self.condition.wait(0.01)
        return None

    def _finish_staging_measurement(self, stage):
        if not (self.cuda and stage.timing_start is not None):
            return
        self.perf.observe("microbatch_gpu", stage.timing_start.elapsed_time(stage.timing_end),
                          events=stage.count, window_id=stage.window_id)
        stage.timing_start = stage.timing_end = None

    def _claim_staging(self):
        stage = self.staging[self.next_staging]
        self.next_staging = (self.next_staging + 1) % len(self.staging)
        if stage.in_use:
            if self.cuda:
                stage.done_event.synchronize()
            self._finish_staging_measurement(stage)
            stage.in_use = False
        stage.count = 0
        stage.first_wait_ns = 0
        stage.window_id = 0
        self.current_staging = stage
        return stage

    def append(self, events, window_start_ns, window_end_ns, generation):
        if self.closed:
            return False
        if self.current is None:
            self.current = self._acquire_slot(generation, window_start_ns, window_end_ns)
            if self.current is None:
                return False
        slot = self.current
        if (slot.generation != generation or slot.window_start_ns != window_start_ns
                or slot.window_end_ns != window_end_ns):
            raise RuntimeError("Incremental event chunk does not match the active voxel window")
        offset = 0
        while offset < len(events):
            stage = self.current_staging or self._claim_staging()
            available = self.max_events - stage.count
            count = min(available, len(events) - offset)
            target = slice(stage.count, stage.count + count)
            source = slice(offset, offset + count)
            tick = self.perf.tick()
            np.copyto(stage.cpu_x[target].numpy(), events["x"][source], casting="unsafe")
            np.copyto(stage.cpu_y[target].numpy(), events["y"][source], casting="unsafe")
            np.copyto(stage.cpu_timestamp[target].numpy(), events["timestamp_ns"][source], casting="unsafe")
            np.copyto(stage.cpu_polarity[target].numpy(), events["polarity"][source], casting="unsafe")
            if not stage.count:
                stage.first_wait_ns = time.monotonic_ns()
            stage.count += count
            slot.event_count += count
            if self.perf.enabled:
                self.perf.elapsed("microbatch_stage", tick, events=count)
            offset += count
            if stage.count == self.max_events:
                self.flush()
        return True

    def flush_due(self):
        stage = self.current_staging
        if stage is not None and stage.count and time.monotonic_ns() - stage.first_wait_ns >= self.max_wait_ns:
            self.flush()

    def seconds_until_flush(self):
        stage = self.current_staging
        if stage is None or not stage.count:
            return 0.01
        remaining = self.max_wait_ns - (time.monotonic_ns() - stage.first_wait_ns)
        return max(0.0, min(0.01, remaining / 1e9))

    def flush(self):
        stage = self.current_staging
        slot = self.current
        if stage is None or not stage.count:
            return
        if slot is None:
            raise RuntimeError("A pending microbatch has no active voxel")
        count = stage.count
        stage.window_id = slot.metadata["window_id"] if slot.metadata else 0
        if self.perf.enabled:
            self.perf.observe("microbatch_events", count, unit="events", events=count)
        flat = slot.voxel.view(-1)
        if self.cuda:
            timing_start = torch.cuda.Event(enable_timing=True) if self.perf.enabled else None
            timing_end = torch.cuda.Event(enable_timing=True) if self.perf.enabled else None
            with torch.cuda.device(self.device), torch.cuda.stream(self.voxel_stream):
                if slot.voxel_start_event is None and self.perf.enabled:
                    slot.voxel_start_event = torch.cuda.Event(enable_timing=True)
                    slot.voxel_start_event.record(self.voxel_stream)
                if timing_start is not None:
                    timing_start.record(self.voxel_stream)
                stage.gpu_x[:count].copy_(stage.cpu_x[:count], non_blocking=True)
                stage.gpu_y[:count].copy_(stage.cpu_y[:count], non_blocking=True)
                stage.gpu_timestamp[:count].copy_(stage.cpu_timestamp[:count], non_blocking=True)
                stage.gpu_polarity[:count].copy_(stage.cpu_polarity[:count], non_blocking=True)
                accumulate_fixed_window(
                    flat, stage.gpu_x[:count], stage.gpu_y[:count],
                    stage.gpu_timestamp[:count], stage.gpu_polarity[:count],
                    slot.window_start_ns, self.window_ns, self.width, self.height)
                if timing_end is not None:
                    timing_end.record(self.voxel_stream)
                stage.done_event.record(self.voxel_stream)
            stage.timing_start, stage.timing_end = timing_start, timing_end
        else:
            accumulate_fixed_window(
                flat, stage.gpu_x[:count].copy_(stage.cpu_x[:count]),
                stage.gpu_y[:count].copy_(stage.cpu_y[:count]),
                stage.gpu_timestamp[:count].copy_(stage.cpu_timestamp[:count]),
                stage.gpu_polarity[:count].copy_(stage.cpu_polarity[:count]),
                slot.window_start_ns, self.window_ns, self.width, self.height)
        stage.in_use = True
        self.microbatch_count += 1
        self.current_staging = None

    def finish_window(self, metadata):
        slot = self.current
        if slot is None:
            raise RuntimeError("Cannot finish a window without an active voxel")
        slot.close_tick_ns = self.perf.tick()
        self.flush()
        if slot.event_count != metadata["event_count"]:
            raise RuntimeError("Incremental voxel event count does not match router metadata")
        slot.metadata = dict(metadata)
        if self.cuda:
            with torch.cuda.device(self.device), torch.cuda.stream(self.voxel_stream):
                slot.fill_done_event.record(self.voxel_stream)
        with self.condition:
            slot.state = "READY"
            self.current = None
            self.ready_slots.append(slot)
            self.condition.notify_all()
        return slot

    def _enqueue_clear(self, slot, wait_event=None):
        slot.state = "CLEARING"
        if self.cuda:
            with torch.cuda.device(self.device), torch.cuda.stream(self.voxel_stream):
                if wait_event is not None:
                    self.voxel_stream.wait_event(wait_event)
                slot.voxel.zero_()
        else:
            slot.voxel.zero_()
        slot.metadata = None
        slot.event_count = 0
        slot.voxel_start_event = None
        slot.model_start_event = None
        slot.state = "FREE"

    def abort_current(self, _event_count=0):
        slot = self.current
        if slot is None:
            return
        if self.current_staging is not None:
            self.current_staging.count = 0
            self.current_staging.first_wait_ns = 0
            self.current_staging = None
        with self.condition:
            self._enqueue_clear(slot)
            self.current = None
            self.invalidated_windows += 1
            self.condition.notify_all()

    def invalidate_generation(self, generation):
        with self.condition:
            if self.current is not None and self.current.generation != generation:
                self.abort_current(self.current.event_count)
            retained = deque()
            while self.ready_slots:
                slot = self.ready_slots.popleft()
                if slot.generation == generation:
                    retained.append(slot)
                else:
                    self._enqueue_clear(slot)
                    self.invalidated_windows += 1
            self.ready_slots.extend(retained)
            self.condition.notify_all()

    def acquire_ready(self, generation, timeout=0.01):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.closed and not self.should_stop():
                while self.ready_slots:
                    slot = self.ready_slots.popleft()
                    if slot.generation != generation:
                        self._enqueue_clear(slot)
                        self.invalidated_windows += 1
                        continue
                    slot.state = "INFERENCING"
                    return slot
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
        return None

    @contextmanager
    def inference_scope(self, slot):
        if self.cuda:
            with torch.cuda.device(self.device), torch.cuda.stream(self.inference_stream):
                self.inference_stream.wait_event(slot.fill_done_event)
                slot.model_submit_tick_ns = self.perf.tick()
                if self.perf.enabled:
                    slot.model_start_event = torch.cuda.Event(enable_timing=True)
                    slot.model_start_event.record(self.inference_stream)
                try:
                    yield
                finally:
                    slot.inference_done_event = torch.cuda.Event()
                    slot.inference_done_event.record(self.inference_stream)
        else:
            slot.model_submit_tick_ns = self.perf.tick()
            try:
                yield
            finally:
                slot.inference_done_event = None

    def observe_inference_start(self, slot):
        if not self.perf.enabled:
            return
        self.perf.observe(
            "window_close_to_inference_submit",
            (slot.model_submit_tick_ns - slot.close_tick_ns) / 1e6,
            events=slot.event_count, window_id=slot.metadata["window_id"])
        if self.cuda and slot.model_start_event is not None:
            slot.model_start_event.synchronize()
            self.perf.observe(
                "window_close_to_inference_gpu",
                slot.fill_done_event.elapsed_time(slot.model_start_event),
                events=slot.event_count, window_id=slot.metadata["window_id"])
            if slot.voxel_start_event is not None:
                self.perf.observe(
                    "incremental_voxel_gpu", slot.voxel_start_event.elapsed_time(slot.fill_done_event),
                    events=slot.event_count, window_id=slot.metadata["window_id"])

    def release(self, slot):
        with self.condition:
            self._enqueue_clear(slot, slot.inference_done_event)
            self.condition.notify_all()

    def close(self):
        with self.condition:
            if self.closed:
                return self.summary()
            self.closed = True
            if self.current is not None:
                self.abort_current(self.current.event_count)
            while self.ready_slots:
                self._enqueue_clear(self.ready_slots.popleft())
            self.condition.notify_all()
        if self.cuda:
            self.voxel_stream.synchronize()
            self.inference_stream.synchronize()
        for stage in self.staging:
            if stage.in_use:
                if self.cuda:
                    stage.done_event.synchronize()
                self._finish_staging_measurement(stage)
                stage.in_use = False
        return self.summary()

    def summary(self):
        return {
            "voxel_pool_slots": len(self.slots),
            "voxel_pool_peak_slots": self.pool_peak_slots,
            "voxel_pool_exhaustions": self.pool_exhaustions,
            "voxel_invalidated_windows": self.invalidated_windows,
            "microbatch_count": self.microbatch_count,
            "microbatch_max_events": self.max_events,
            "microbatch_max_wait_ms": self.max_wait_ns / 1e6,
        }
