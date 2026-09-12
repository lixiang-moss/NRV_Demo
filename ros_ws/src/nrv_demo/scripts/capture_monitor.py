#!/usr/bin/env python3
"""RAW packet counts and acquisition duration; no image or event algorithm."""
import json
from pathlib import Path
import threading
import time

import rospy
from event_camera_msgs.msg import EventPacket
from std_msgs.msg import String


class CaptureMonitor:
    def __init__(self):
        self.duration = float(rospy.get_param('~duration', 0.0))
        self.output = Path(rospy.get_param('~output_dir'))
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.stats = dict(raw_packets=0, raw_bytes=0, raw_sequence_gaps=0,
                          encoding=None, width=0, height=0)
        self.last_seq = None
        self.publisher = rospy.Publisher('~status', String, queue_size=1)
        self.subscriber = rospy.Subscriber('/delta_driver/events', EventPacket, self.on_raw,
                                           queue_size=100, buff_size=16 * 1024 * 1024)

    def on_raw(self, packet):
        with self.lock:
            if self.last_seq is not None and packet.seq != (self.last_seq + 1) % (1 << 64):
                self.stats['raw_sequence_gaps'] += 1
            self.last_seq = packet.seq
            self.stats['raw_packets'] += 1
            self.stats['raw_bytes'] += len(packet.events)
            self.stats.update(encoding=packet.encoding, width=packet.width, height=packet.height,
                              last_raw_seq=packet.seq, last_raw_time_base=packet.time_base)

    def run(self):
        previous_time, previous_bytes = self.started, 0
        next_status = self.started
        try:
            while not rospy.is_shutdown():
                now = time.monotonic()
                if self.duration > 0 and now - self.started >= self.duration:
                    break
                if now >= next_status:
                    with self.lock:
                        status = dict(self.stats)
                    status['bytes_per_second'] = (status['raw_bytes'] - previous_bytes) / max(now - previous_time, .001)
                    self.publisher.publish(String(data=json.dumps(status)))
                    previous_time, previous_bytes = now, status['raw_bytes']
                    next_status = now + 1.0
                time.sleep(.05)
        finally:
            self.subscriber.unregister()
            with self.lock:
                result = dict(self.stats)
            checks = dict(raw_received=result['raw_packets'] > 0 and result['raw_bytes'] > 0,
                          raw_sequence_contiguous=result['raw_sequence_gaps'] == 0)
            result.update(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                          elapsed_seconds=round(time.monotonic() - self.started, 3))
            (self.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    rospy.init_node('nrv_capture')
    raise SystemExit(CaptureMonitor().run())
