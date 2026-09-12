"""Optional bounded, periodic batch measurements; no ROS or NumPy dependency."""

from collections import deque
import json
import math
from pathlib import Path
import re
import threading
import time


def _percentile(ordered, fraction):
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class PerfRecorder:
    """Keep complete totals and recent bounded samples per reporting interval.

    Disabled methods deliberately return before clocks, allocation, or locks.
    When capacity is exceeded, percentiles describe only retained recent samples.
    Disk I/O and percentile sorting happen on the reporting thread, outside the
    sampling lock. Components should use a fixed, finite set of metric names.
    """

    def __init__(self, output_dir, component, enabled=False, interval_s=5, capacity=4096):
        self.enabled = bool(enabled)
        self.error = None
        if not self.enabled:
            return
        self.interval_s = float(interval_s)
        self.capacity = int(capacity)
        if not math.isfinite(self.interval_s) or self.interval_s <= 0 or self.capacity < 1:
            raise ValueError("performance interval and capacity must be positive")
        self.component = re.sub(r"[^A-Za-z0-9_.-]", "_", str(component))
        self.path = Path(output_dir) / ("performance_" + self.component + ".jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._metrics = {}
        self._counters = {}
        self._closed = False
        self._start_ns = time.perf_counter_ns()
        self._thread = threading.Thread(target=self._run, name="perf-" + self.component, daemon=True)
        self._thread.start()

    def tick(self):
        return time.perf_counter_ns() if self.enabled else 0

    def elapsed(self, name, start_ns, events=0, **metadata):
        if not self.enabled:
            return
        self.observe(name, (time.perf_counter_ns() - start_ns) / 1e6,
                     events=events, **metadata)

    def observe(self, name, value, unit="ms", events=0, **metadata):
        if not self.enabled:
            return
        value = float(value)
        if not math.isfinite(value):
            self.increment("invalid_measurements")
            return
        sample_ns = time.perf_counter_ns()
        event_count = int(events)
        with self._lock:
            if self._closed:
                return
            metric = self._metrics.get(name)
            if metric is None:
                metric = dict(unit=unit, samples=deque(maxlen=self.capacity), count=0,
                              events=0, total=0.0, minimum=value, maximum=value,
                              metadata_last={})
                self._metrics[name] = metric
            elif metric["unit"] != unit:
                raise ValueError("A performance metric cannot change units: " + name)
            sample_metadata = dict(list(metadata.items())[:16])
            metric["samples"].append((value, event_count, sample_ns, sample_metadata))
            metric["count"] += 1
            metric["events"] += event_count
            metric["total"] += value
            metric["minimum"] = min(metric["minimum"], value)
            metric["maximum"] = max(metric["maximum"], value)
            # Metadata shares the same bounded batch/window sample lifetime.
            metric["metadata_last"] = sample_metadata

    def increment(self, name, n=1):
        if not self.enabled:
            return
        with self._lock:
            if not self._closed:
                self._counters[name] = self._counters.get(name, 0) + n

    def _flush(self, final=False):
        end_ns = time.perf_counter_ns()
        with self._lock:
            metrics, self._metrics = self._metrics, {}
            counters, self._counters = self._counters, {}
            start_ns, self._start_ns = self._start_ns, end_ns
        result = dict(schema_version=1, component=self.component,
                      clock="perf_counter_ns", interval_start_ns=start_ns,
                      interval_end_ns=end_ns, interval_s=(end_ns - start_ns) / 1e9,
                      final=final, capacity_per_metric=self.capacity,
                      metrics={}, counters=counters)
        for name, metric in metrics.items():
            samples = metric["samples"]
            values = sorted(sample[0] for sample in samples)
            result["metrics"][name] = dict(
                unit=metric["unit"], count=metric["count"], events=metric["events"],
                sample_count=len(values), sample_events=sum(s[1] for s in samples),
                overflow=metric["count"] - len(values), sum=metric["total"],
                mean=metric["total"] / metric["count"], min=metric["minimum"],
                max=metric["maximum"], p50=_percentile(values, .5),
                p95=_percentile(values, .95), p99=_percentile(values, .99),
                percentile_scope="recent_bounded_samples", metadata_last=metric["metadata_last"],
                samples=[[sample[0], sample[1], sample[2]] for sample in samples],
                sample_metadata=[sample[3] for sample in samples])
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        except (OSError, ValueError, TypeError) as error:
            self.error = str(error)

    def _run(self):
        while not self._stop.wait(self.interval_s):
            self._flush()

    def close(self):
        if not self.enabled:
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self._thread.join()
        self._flush(final=True)
