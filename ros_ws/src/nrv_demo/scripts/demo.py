#!/usr/bin/env python3
"""NRV ROS1 Python handoff: encoded RAW reception and event-centroid images."""

import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import rospy
from dvs_msgs.msg import EventArray
from event_camera_msgs.msg import EventPacket
from sensor_msgs.msg import Image
from std_msgs.msg import String
from noise_filter import NoiseFilter


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
        self.filter = NoiseFilter()
        self.filter_config = dict(background=False, window_ms=5.0, refractory=False, interval_ms=1.0)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1)
        self.stats["filtered_events"] = 0
        self.last_seq = None
        self.canvas = None
        self.last_output = None
        self.frame_id = ""
        self.event_stamp = rospy.Time()
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
        timestamps = np.fromiter((e.ts.to_sec() for e in message.events), dtype=float, count=count)
        keep = self.filter.apply(x, y, timestamps, (message.height, message.width), self.filter_config)
        x, y, on = x[keep], y[keep], on[keep]
        with self.lock:
            self.stats["decoded_events"] += count
            count = len(x)
            self.stats["filtered_events"] += count
            if self.canvas is None:
                self.canvas = np.zeros((message.height, message.width, 3), dtype=np.uint8)
            # ON = green, OFF = blue, stored in OpenCV BGR order.
            self.canvas[y[~on], x[~on]] = (255, 100, 40)
            self.canvas[y[on], x[on]] = (80, 220, 80)
            # Centroid of events retained by the software filters.
            self.window_count += count
            self.window_x += int(x.sum())
            self.window_y += int(y.sum())
            self.window_on += int(on.sum())
            self.stats["event_batches"] += 1
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
            self.filter_config = rospy.get_param("/nrv_noise", self.filter_config)
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
                stats["events_per_second"] = (stats["decoded_events"] - previous_events) / elapsed
                self.status_pub.publish(String(data=json.dumps(stats)))
                previous_bytes, previous_events = stats["raw_bytes"], stats["decoded_events"]
                previous_log, next_log = now, now + 1.0
            time.sleep(1.0 / self.image_fps)
        for subscriber in self.subscribers:
            subscriber.unregister()
        with self.lock:
            result = dict(self.stats)
            last_output = self.last_output
        checks = dict(raw_received=result["raw_packets"] > 0 and result["raw_bytes"] > 0,
                      raw_sequence_contiguous=result["raw_sequence_gaps"] == 0)
        if self.decode_events:
            checks.update(events_received=result["decoded_events"] > 0,
                          renderer_received=result["rendered_images"] > 0,
                          algorithm_published=result["algorithm_images"] > 0)
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
