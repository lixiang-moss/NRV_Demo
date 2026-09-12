#!/usr/bin/env python3
"""Sample Linux process/thread, memory and NVIDIA GPU resources without psutil."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time


PROC = Path("/proc")
TICKS = os.sysconf("SC_CLK_TCK")
PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")


def read_stat(path):
    raw = path.read_text()
    end = raw.rfind(")")
    fields = raw[end + 2:].split()
    return dict(name=raw[raw.find("(") + 1:end], ppid=int(fields[1]),
                ticks=int(fields[11]) + int(fields[12]), start_ticks=int(fields[19]),
                rss_bytes=int(fields[21]) * PAGE_BYTES)


def read_processes(roots, patterns):
    all_processes = {}
    selected = set(roots)
    for path in PROC.iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        pid = int(path.name)
        try:
            stat = read_stat(path / "stat")
            all_processes[pid] = stat
            if patterns:
                command = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                if any(pattern.search(command) for pattern in patterns):
                    selected.add(pid)
        except (OSError, ValueError, IndexError):
            continue
    while True:
        children = {pid for pid, stat in all_processes.items() if stat["ppid"] in selected}
        if children.issubset(selected):
            break
        selected.update(children)
    return {pid: all_processes[pid] for pid in selected if pid in all_processes}


def memory_status():
    values = {}
    for line in (PROC / "meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0]) * 1024
    return dict(memory_total_bytes=values.get("MemTotal"),
                memory_available_bytes=values.get("MemAvailable"),
                swap_total_bytes=values.get("SwapTotal"), swap_free_bytes=values.get("SwapFree"))


def _gpu_command(query):
    result = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=2)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    return csv.reader(result.stdout.splitlines(), skipinitialspace=True)


def _number(value, scale=1):
    try:
        return float(value) * scale
    except ValueError:
        return None


def gpu_status():
    result = dict(devices=[], processes=[], error=None)
    try:
        for row in _gpu_command("--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total"):
            if len(row) == 5:
                result["devices"].append(dict(index=int(row[0]), uuid=row[1],
                    utilization_percent=_number(row[2]), memory_used_bytes=_number(row[3], 1024 ** 2),
                    memory_total_bytes=_number(row[4], 1024 ** 2)))
        for row in _gpu_command("--query-compute-apps=pid,gpu_uuid,used_gpu_memory"):
            if len(row) == 3:
                result["processes"].append(dict(pid=int(row[0]), gpu_uuid=row[1],
                                               memory_used_bytes=_number(row[2], 1024 ** 2)))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        result["error"] = str(error)
    return result


def sample(roots, patterns, previous, now_ns, gpu=True):
    rows = []
    current = {}
    processes = read_processes(roots, patterns)
    for pid, stat in processes.items():
        tasks = [(pid, "process", stat)]
        try:
            for path in (PROC / str(pid) / "task").iterdir():
                try:
                    tasks.append((int(path.name), "thread", read_stat(path / "stat")))
                except (OSError, ValueError, IndexError):
                    continue
        except OSError:
            pass
        for tid, kind, item in tasks:
            key = (pid, tid, kind, item["start_ticks"])
            old = previous.get(key)
            cpu = None
            if old is not None and now_ns > old[1] and item["ticks"] >= old[0]:
                cpu = 100 * (item["ticks"] - old[0]) / TICKS / ((now_ns - old[1]) / 1e9)
            current[key] = (item["ticks"], now_ns)
            rows.append(dict(type=kind, pid=pid, tid=tid, name=item["name"],
                             cpu_percent=cpu, rss_bytes=item["rss_bytes"]))
    try:
        memory = memory_status()
    except (OSError, ValueError) as error:
        memory = dict(error=str(error))
    result = dict(processes=rows, system=memory,
                  gpu=gpu_status() if gpu else dict(devices=[], processes=[], error="disabled"))
    return result, current


CSV_FIELDS = ["elapsed_s", "monotonic_ns", "unix_ns", "type", "pid", "tid", "name", "cpu_percent",
              "rss_bytes", "index", "uuid", "gpu_uuid", "utilization_percent",
              "memory_used_bytes", "memory_total_bytes", "memory_available_bytes",
              "swap_total_bytes", "swap_free_bytes", "collection_ms", "error"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, action="append", default=[])
    parser.add_argument("--match", action="append", default=[], help="command-line regex; repeatable")
    parser.add_argument("--duration", type=float, default=0, help="seconds; 0 until interrupted")
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()
    if args.duration < 0 or args.interval <= 0:
        parser.error("duration must be nonnegative and interval positive")
    patterns = [re.compile(value) for value in args.match]
    if not args.pid and not patterns:
        patterns = [re.compile(r"(?:e2fai|nrv_capture|nrv_gui|nodelet|roslaunch)")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    previous = {}
    started = time.monotonic_ns()
    with (args.output_dir / "resources.jsonl").open("w") as json_file, \
            (args.output_dir / "resources.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, CSV_FIELDS)
        writer.writeheader()
        while not stopped.is_set():
            tick = time.monotonic_ns()
            elapsed = (tick - started) / 1e9
            if args.duration and elapsed >= args.duration:
                break
            result, previous = sample(args.pid, patterns, previous, tick, not args.no_gpu)
            result.update(elapsed_s=elapsed, monotonic_ns=tick, unix_ns=time.time_ns(),
                          collection_ms=(time.monotonic_ns() - tick) / 1e6,
                          cpu_percent_basis="100_percent_per_logical_cpu",
                          thread_rss_shared_with_process=True)
            json_file.write(json.dumps(result, allow_nan=False) + "\n")
            common = dict(elapsed_s=elapsed, monotonic_ns=tick, unix_ns=result["unix_ns"])
            rows = result["processes"] + [dict(type="system", collection_ms=result["collection_ms"],
                                                **result["system"])]
            rows += [dict(type="gpu", **item) for item in result["gpu"]["devices"]]
            rows += [dict(type="gpu_process", **item) for item in result["gpu"]["processes"]]
            if result["gpu"]["error"]:
                rows.append(dict(type="gpu_error", error=result["gpu"]["error"]))
            for row in rows:
                writer.writerow(dict(common, **row))
            json_file.flush()
            csv_file.flush()
            stopped.wait(max(0, args.interval - (time.monotonic_ns() - tick) / 1e9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
