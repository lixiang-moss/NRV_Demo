#!/usr/bin/env python3
"""Exercise the real host receiver/window/voxel/GPU/render path without a camera.

Run from the repo root with the E2FAI Python environment and PYTHONPATH=runtime.
The default is 20 million spatially distributed events/second for ten seconds.
This is headless and retains the demo's default one-million-event voxel limit.
"""

import argparse
import json
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "ready").exists():
        parser.error("use a fresh output directory")
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    count, packets, period = 200_000, 1000, 0.01
    events = np.empty(count, dtype=[("x", "<u2"), ("y", "<u2"), ("polarity", "u1")])
    index = np.arange(count, dtype=np.uint32)
    events["x"], events["y"], events["polarity"] = index % 960, (index // 960) % 720, index % 2
    payload = events.tobytes()
    log = (args.output_dir / "host.log").open("w")
    process = subprocess.Popen([
        sys.executable, str(root / "examples/e2fai_realtime.py"), "--headless",
        "--port", str(port), "--input-width", "640", "--input-height", "480",
        "--image-checkpoint", str(root / "checkpoints/image_residual_epoch043.pt"),
        "--backbone", str(root / "checkpoints/e2fai_backbone.ckpt"),
        "--output-dir", str(args.output_dir),
    ], stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 60
        while not (args.output_dir / "ready").exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("E2FAI did not start; inspect host.log")
            time.sleep(0.1)
        with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
            started = time.monotonic()
            start_ns = time.time_ns()
            for sequence in range(packets):
                delay = started + sequence * period - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                first = start_ns + sequence * 10_000_000
                connection.sendall(struct.pack("!4sIIIIQQQQ", b"NRV2", 960, 720, count,
                                               sequence, first, first + 10_000_000, 0, 0) + payload)
            offered_rate = (packets - 1) * count / (time.monotonic() - started)
        # Bound drain time; any remaining backlog is a test failure, not hidden.
        time.sleep(2)
    finally:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
    result = json.loads((args.output_dir / "e2fai_summary.json").read_text())
    passed = (process.returncode == 0 and result["bridge_received_batches"] == packets
              and result["bridge_received_events"] == packets * count
              and result["bridge_unprocessed_batches_at_shutdown"] == 0
              and result["sender_dropped_batches_observed"] == 0
              and result["processed_windows"] == 100
              and offered_rate >= 19_000_000 and result["output_fps"] >= 9.5)
    report = {"passed": passed, "offered_event_rate": offered_rate,
              "published_batches": packets, "published_events": packets * count,
              "headless": True, "summary": result}
    (args.output_dir / "host_stress_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
