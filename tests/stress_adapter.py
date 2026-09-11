#!/usr/bin/env python3
"""Deterministic grouped-AER -> ROS adapter -> TCP test, inside the ROS image.

No camera, GPU, event sampling, or changes to the user's ROS master are required.
Run with the repository mounted at /workspace and a writable --output-dir.
The drain client deliberately isolates upstream throughput from model throughput.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import threading
import time
import xmlrpc.client

import numpy as np


def recv_exact(connection, size):
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        count = connection.recv_into(view[offset:])
        if not count:
            raise EOFError("Incomplete bridge packet")
        offset += count
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("NRV1", "NRV2"), default="NRV2")
    parser.add_argument("--events-per-packet", type=int, default=400_000)
    parser.add_argument("--packet-hz", type=float, default=100)
    parser.add_argument("--packets", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.events_per_packet <= 0 or args.events_per_packet % 16 or args.packet_hz <= 0 or args.packets <= 0:
        parser.error("positive sizes/rates required; events-per-packet must be a multiple of 16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        ros_port = probe.getsockname()[1]
    os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:{}/".format(ros_port)
    os.environ["ROS_HOSTNAME"] = "127.0.0.1"
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(15)
    bridge_port = server.getsockname()[1]
    stopped = threading.Event()
    stats = {"received_batches": 0, "received_events": 0, "received_bytes": 0,
             "sequence_missing": 0, "payload_errors": 0, "reader_error": None}

    def drain():
        previous = None
        try:
            connection, _ = server.accept()
            with connection:
                connection.settimeout(15)
                while not stopped.is_set():
                    header = recv_exact(connection, 20)
                    magic, width, height, count, sequence = struct.unpack("!4sIIII", header)
                    if magic.decode() != args.protocol or (width, height) != (960, 720):
                        raise ValueError("Unexpected protocol or geometry")
                    if previous is not None:
                        stats["sequence_missing"] += (sequence - previous - 1) % (1 << 32)
                    previous = sequence
                    if magic == b"NRV2":
                        start, end, _, _ = struct.unpack("!QQQQ", recv_exact(connection, 32))
                        if not 0 <= end - start <= 100_000_000:
                            raise ValueError("Invalid timestamp interval")
                        dtype = np.dtype([("x", "<u2"), ("y", "<u2"), ("polarity", "u1")])
                        extra = 32
                    else:
                        dtype = np.dtype([("x", "<u2"), ("y", "<u2"), ("ts", "<u8"), ("polarity", "u1")])
                        extra = 0
                    payload = recv_exact(connection, count * dtype.itemsize)
                    events = np.frombuffer(payload, dtype=dtype)
                    valid = (count == args.events_per_packet and events[0]["y"] == 0
                             and events[-1]["y"] == 15 and events[0]["polarity"] == 1
                             and events[-1]["polarity"] == 0 and events[0]["x"] == events[-1]["x"]
                             and events[0]["x"] < 960)
                    stats["payload_errors"] += int(not valid)
                    stats["received_batches"] += 1
                    stats["received_events"] += count
                    stats["received_bytes"] += 20 + extra + len(payload)
        except (EOFError, OSError) as error:
            if not stopped.is_set():
                stats["reader_error"] = str(error)
        except Exception as error:
            stats["reader_error"] = repr(error)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    log = (args.output_dir / "ros.log").open("w")
    process = subprocess.Popen([
        "roslaunch", str(Path(__file__).with_name("adapter_stress.launch")),
        "port:={}".format(bridge_port), "compact:={}".format(str(args.protocol == "NRV2").lower()),
        "summary_path:={}".format(args.output_dir / "adapter_summary.json"),
    ], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                xmlrpc.client.ServerProxy(os.environ["ROS_MASTER_URI"]).getUri("stress_probe")
                break
            except OSError:
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("ROS master did not start; inspect ros.log")
                time.sleep(0.05)
        import rospy
        from event_camera_msgs.msg import EventPacket

        rospy.init_node("adapter_stress_source", disable_signals=True)
        # Synchronous publication avoids an additional dropping publisher queue.
        publisher = rospy.Publisher("/nrv_stress/events", EventPacket, queue_size=None)
        deadline = time.monotonic() + 15
        while publisher.get_num_connections() == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError("Adapter did not subscribe; inspect ros.log")
            time.sleep(0.05)
        # Frame end + reference timestamp + column + repeated two-row-group words.
        # Each group emits 8 ON events at y=0..7 and 8 OFF events at y=8..15.
        prefix = bytes.fromhex("0c00000008000000")
        groups = bytes.fromhex("8402ffff") * (args.events_per_packet // 16)
        packet = EventPacket(width=960, height=720, encoding="group_aer", is_bigendian=True)
        started = time.monotonic()
        for sequence in range(args.packets):
            delay = started + sequence / args.packet_hz - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            x = sequence % 960
            packet.events = prefix + bytes((4, 0, x >> 8, x & 255)) + groups
            packet.seq = sequence
            packet.header.stamp = rospy.Time.now()
            publisher.publish(packet)
        elapsed = time.monotonic() - started
        stats["offered_event_rate"] = (args.packets - 1) * args.events_per_packet / elapsed
        deadline = time.monotonic() + 5
        while stats["received_batches"] < args.packets and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        stopped.set()
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        server.close()
        reader.join(timeout=2)
        log.close()
    adapter = json.loads((args.output_dir / "adapter_summary.json").read_text())
    stats.update(protocol=args.protocol, published_batches=args.packets,
                 published_events=args.packets * args.events_per_packet, adapter=adapter)
    stats["passed"] = (stats["received_batches"] == args.packets
                       and stats["received_events"] == stats["published_events"]
                       and not stats["payload_errors"] and not stats["reader_error"]
                       and not adapter["raw_missing_packets"]
                       and not adapter["bridge_dropped_batches"]
                       and stats["offered_event_rate"] >= 0.9 * args.packet_hz * args.events_per_packet)
    (args.output_dir / "stress_summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0 if stats["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
