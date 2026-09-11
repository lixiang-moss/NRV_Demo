#!/usr/bin/env python3
"""NRV ROS1 Python handoff: encoded RAW reception and event-centroid images."""

import json
from pathlib import Path
import queue
import socket
import struct
import threading
import time

import cv2
import numpy as np
import rospy
from dvs_msgs.msg import EventArray
from event_camera_msgs.msg import EventPacket
from sensor_msgs.msg import Image


EVENT_DTYPE = np.dtype([
    ("x", "<u2"), ("y", "<u2"), ("timestamp_ns", "<u8"), ("polarity", "u1")
])
BRIDGE_HEADER = struct.Struct("!4sIIII")


class EventTcpSender:
    """Non-blocking latest-batch bridge from ROS to the host GPU process."""

    def __init__(self, host, port):
        self.host, self.port = host, port
        self.queue = queue.Queue(maxsize=2)
        self.stopping = threading.Event()
        self.sent_batches = self.dropped_batches = self.reconnects = 0
        self.sequence = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, width, height, x, y, timestamp_ns, polarity):
        events = np.empty(len(x), dtype=EVENT_DTYPE)
        events["x"], events["y"] = x, y
        events["timestamp_ns"], events["polarity"] = timestamp_ns, polarity
        item = (width, height, events, self.sequence)
        self.sequence = (self.sequence + 1) % (1 << 32)
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            self.queue.get_nowait()  # Keep latency bounded by discarding the oldest batch.
            self.queue.put_nowait(item)
            self.dropped_batches += 1

    def _run(self):
        connection = None
        while not self.stopping.is_set():
            try:
                item = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if connection is None:
                try:
                    connection = socket.create_connection((self.host, self.port), timeout=1.0)
                    connection.settimeout(1.0)
                    self.reconnects += 1
                except OSError:
                    self.dropped_batches += 1
                    time.sleep(0.2)
                    continue
            width, height, events, sequence = item
            try:
                connection.sendall(BRIDGE_HEADER.pack(
                    b"NRV1", width, height, len(events), sequence
                ))
                connection.sendall(memoryview(events).cast("B"))
                self.sent_batches += 1
            except OSError:
                connection.close()
                connection = None
                self.dropped_batches += 1
        if connection is not None:
            connection.close()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=2.0)


def process_raw_packet(packet):
    """Algorithm integration point for the original encoded payload.

    packet.events is uint8[] (bytes in rospy). Keep packet.encoding, seq,
    time_base, header, width and height with it if passing it downstream.
    This example only counts bytes; it does not decode or modify the payload.
    """
    payload = memoryview(packet.events)
    return payload.nbytes


class NrvPythonDemo:
    def __init__(self):
        self.decode_events = rospy.get_param("~decode_events", True)
        self.duration = float(rospy.get_param("~duration", 0.0))
        self.image_fps = float(rospy.get_param("~image_fps", 10.0))
        self.output_dir = Path(rospy.get_param("~output_dir"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.stats = dict(raw_packets=0, raw_bytes=0, raw_sequence_gaps=0,
                          event_batches=0, decoded_events=0, rendered_images=0,
                          algorithm_images=0, encoding=None, width=0, height=0)
        self.last_seq = None
        self.canvas = None
        self.last_output = None
        self.frame_id = ""
        self.event_stamp = rospy.Time()
        bridge_port = int(rospy.get_param("~bridge_port", 0))
        self.bridge = EventTcpSender(rospy.get_param("~bridge_host", "127.0.0.1"), bridge_port) if bridge_port else None
        self.window_count = self.window_x = self.window_y = self.window_on = 0
        self.subscribers = [rospy.Subscriber(
            rospy.get_param("~raw_topic", "/delta_driver/events"), EventPacket,
            self.on_raw, queue_size=100, buff_size=16 * 1024 * 1024)]
        if self.decode_events:
            self.publisher = rospy.Publisher("~image", Image, queue_size=1)
            self.subscribers.extend([
                rospy.Subscriber(rospy.get_param("~events_topic", "/dvs/events"),
                                 EventArray, self.on_events, queue_size=10,
                                 buff_size=64 * 1024 * 1024),
                rospy.Subscriber(rospy.get_param("~rendered_topic", "/delta_renderer/image"),
                                 Image, self.on_rendered, queue_size=1,
                                 buff_size=16 * 1024 * 1024),
            ])

    def on_raw(self, packet):
        byte_count = process_raw_packet(packet)  # Replace this function for a RAW algorithm.
        with self.lock:
            if self.last_seq is not None and packet.seq != (self.last_seq + 1) % (1 << 64):
                self.stats["raw_sequence_gaps"] += 1
            self.last_seq = packet.seq
            self.stats["raw_packets"] += 1
            self.stats["raw_bytes"] += byte_count
            self.stats.update(encoding=packet.encoding, width=packet.width,
                              height=packet.height, last_raw_seq=packet.seq,
                              last_raw_time_base=packet.time_base)

    def on_events(self, message):
        """Example algorithm input: every event has x, y, ts and polarity."""
        count = len(message.events)
        if not count:
            return
        x = np.fromiter((e.x for e in message.events), dtype=np.intp, count=count)
        y = np.fromiter((e.y for e in message.events), dtype=np.intp, count=count)
        on = np.fromiter((e.polarity for e in message.events), dtype=bool, count=count)
        if self.bridge is not None:
            timestamp_ns = np.fromiter((e.ts.to_nsec() for e in message.events), dtype=np.uint64, count=count)
            self.bridge.submit(message.width, message.height, x, y, timestamp_ns, on)
        with self.lock:
            if self.canvas is None:
                self.canvas = np.zeros((message.height, message.width, 3), dtype=np.uint8)
            # ON = green, OFF = blue, stored in OpenCV BGR order.
            self.canvas[y[~on], x[~on]] = (255, 100, 40)
            self.canvas[y[on], x[on]] = (80, 220, 80)
            # Simple algorithm: centroid of ALL events in this display window.
            self.window_count += count
            self.window_x += int(x.sum())
            self.window_y += int(y.sum())
            self.window_on += int(on.sum())
            self.stats["event_batches"] += 1
            self.stats["decoded_events"] += count
            self.event_stamp = message.header.stamp
            self.frame_id = message.header.frame_id

    def on_rendered(self, _message):
        with self.lock:
            self.stats["rendered_images"] += 1

    def publish_algorithm_image(self):
        with self.lock:
            if self.canvas is None:
                return
            frame = self.canvas.copy()
            self.canvas.fill(0)
            count, sx, sy, on = self.window_count, self.window_x, self.window_y, self.window_on
            self.window_count = self.window_x = self.window_y = self.window_on = 0
            stamp, frame_id = self.event_stamp, self.frame_id
        centroid = None
        if count:
            centroid = [sx / count, sy / count]
            center = (round(centroid[0]), round(centroid[1]))
            cv2.drawMarker(frame, center, (0, 255, 255), cv2.MARKER_CROSS, 35, 2)
            cv2.circle(frame, center, 20, (0, 255, 255), 2)
        # Labels have a black backing so they remain legible over active pixels.
        lines = ["Python: event activity centroid", "window events: {}  ON: {}".format(count, on),
                 "centroid: ({:.1f}, {:.1f})".format(*centroid) if centroid else "no events in this window"]
        for index, label in enumerate(lines):
            baseline = 25 + 25 * index
            cv2.putText(frame, label, (10, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(frame, label, (10, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        output = Image()
        output.header.stamp = stamp
        output.header.frame_id = frame_id
        output.height, output.width = frame.shape[:2]
        output.encoding = "bgr8"
        output.is_bigendian = False
        output.step = output.width * 3
        output.data = frame.tobytes()
        self.publisher.publish(output)
        with self.lock:
            self.stats["algorithm_images"] += 1
            self.stats["last_window_events"] = count
            self.stats["last_centroid"] = centroid
            self.last_output = frame

    def run(self):
        rospy.loginfo("NRV Python demo: RAW receiver active; decoded algorithm=%s; duration=%s s (0=Ctrl+C)",
                      self.decode_events, self.duration)
        next_log = self.started
        previous_bytes = previous_events = 0
        previous_log = self.started
        while not rospy.is_shutdown():
            now = time.monotonic()
            if self.duration > 0 and now - self.started >= self.duration:
                break
            if self.decode_events:
                self.publish_algorithm_image()
            if now >= next_log:
                with self.lock:
                    stats = dict(self.stats)
                elapsed = max(now - previous_log, 0.001)
                rospy.loginfo("RAW packets=%d bytes=%d %.2f MB/s gaps=%d | events=%d %.0f ev/s | rendered=%d algorithm=%d",
                              stats["raw_packets"], stats["raw_bytes"],
                              (stats["raw_bytes"] - previous_bytes) / elapsed / 1e6,
                              stats["raw_sequence_gaps"], stats["decoded_events"],
                              (stats["decoded_events"] - previous_events) / elapsed,
                              stats["rendered_images"], stats["algorithm_images"])
                previous_bytes, previous_events = stats["raw_bytes"], stats["decoded_events"]
                previous_log, next_log = now, now + 1.0
            time.sleep(1.0 / self.image_fps)
        for subscriber in self.subscribers:
            subscriber.unregister()
        if self.bridge is not None:
            self.bridge.close()
        with self.lock:
            result = dict(self.stats)
            last_output = self.last_output
        checks = dict(raw_received=result["raw_packets"] > 0 and result["raw_bytes"] > 0,
                      raw_sequence_contiguous=result["raw_sequence_gaps"] == 0)
        if self.decode_events:
            checks.update(events_received=result["decoded_events"] > 0,
                          renderer_received=result["rendered_images"] > 0,
                          algorithm_published=result["algorithm_images"] > 0)
        if self.bridge is not None:
            result.update(bridge_sent_batches=self.bridge.sent_batches,
                          bridge_dropped_batches=self.bridge.dropped_batches,
                          bridge_connections=self.bridge.reconnects)
            checks["event_bridge_sent"] = self.bridge.sent_batches > 0
        result.update(status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                      elapsed_seconds=round(time.monotonic() - self.started, 3),
                      decode_events=self.decode_events)
        (self.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        if last_output is not None:
            cv2.imwrite(str(self.output_dir / "algorithm_last.png"), last_output)
        rospy.loginfo("%s: %s", result["status"], json.dumps(checks))
        return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    rospy.init_node("nrv_python_demo")
    raise SystemExit(NrvPythonDemo().run())
