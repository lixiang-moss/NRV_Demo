#!/usr/bin/env python3
"""Minimal partner algorithm entry point. Run while the demo/driver is active."""
import rospy
from event_camera_msgs.msg import EventPacket


def on_raw(msg):
    payload = memoryview(msg.events)
    # Replace this with your algorithm. Keep the metadata with the encoded bytes:
    # algorithm.push(payload, encoding=msg.encoding, seq=msg.seq,
    #                time_base=msg.time_base, stamp=msg.header.stamp,
    #                width=msg.width, height=msg.height,
    #                is_bigendian=msg.is_bigendian)
    rospy.loginfo_throttle(1.0, "seq=%d encoding=%s payload=%d bytes",
                          msg.seq, msg.encoding, payload.nbytes)


if __name__ == "__main__":
    rospy.init_node("partner_raw_receiver")
    subscriber = rospy.Subscriber("/delta_driver/events", EventPacket, on_raw,
                                  queue_size=100, buff_size=16 * 1024 * 1024)
    rospy.spin()
