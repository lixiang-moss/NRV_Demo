"""Opt-in real ROS serialization test, isolated from camera and GPU workloads.

Run NRV_TEST_ROS_SEQUENCE=1 python -m unittest discover -s tests
    -p 'test_e2fai_ros_sequence.py' -v
Requires the project's nrv-demo:noetic image and Docker. No host ROS port,
camera device, GPU or workspace mount is used by the probe container.
"""
import json
import os
import subprocess
import unittest


PROBE = r'''
import json
import pathlib
import shlex
import subprocess
import tempfile
import time
import xmlrpc.client

import rospy
from dvs_msgs.msg import EventArray

code = r"""
#include <ros/ros.h>
#include <dvs_msgs/EventArray.h>
int main(int argc, char **argv) {
  ros::init(argc, argv, "seq_cpp_probe");
  ros::NodeHandle nh;
  auto publisher = nh.advertise<dvs_msgs::EventArray>("/seq_probe", 10);
  for (int wait = 0; wait < 100 && !publisher.getNumSubscribers(); ++wait) {
    ros::spinOnce();
    ros::WallDuration(0.02).sleep();
  }
  ros::Rate rate(20);
  for (int index = 0; index < 60 && ros::ok(); ++index) {
    dvs_msgs::EventArray message;
    message.header.seq = 0;  // Deliberately never assign a publication counter.
    message.header.stamp = ros::Time::now();
    message.width = 960;
    message.height = 720;
    publisher.publish(message);
    rate.sleep();
  }
}
"""
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    (root / 'seq.cpp').write_text(code)
    flags = shlex.split(subprocess.check_output(
        ['pkg-config', '--cflags', '--libs', 'roscpp'], universal_newlines=True))
    subprocess.run(['g++', '-std=c++14', '-I/opt/nrv_demo_ws/devel/include',
                    str(root / 'seq.cpp'), '-o', str(root / 'seq')] + flags, check=True)
    core = subprocess.Popen(['roscore'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                response = xmlrpc.client.ServerProxy('http://127.0.0.1:11311').getPid('/seq_probe_test')
                if response[0] == 1:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError('Isolated roscore failed to start')
        rospy.init_node('seq_python_observer', disable_signals=True)
        received = []
        subscriber = rospy.Subscriber('/seq_probe', EventArray,
                                       lambda message: received.append(int(message.header.seq)),
                                       queue_size=100)
        subprocess.run([str(root / 'seq')], check=True, timeout=10)
        time.sleep(0.2)
        assert len(received) >= 20, received
        assert all(second == (first + 1) % (1 << 32)
                   for first, second in zip(received, received[1:])), received
        assert any(received), received
        print(json.dumps({'probe': 'roscpp_default_header_seq_to_rospy',
                          'assigned_header_seq': 0, 'received_sequences': received}), flush=True)
        subscriber.unregister()
    finally:
        rospy.signal_shutdown('probe completed')
        core.terminate()
        try:
            core.wait(timeout=10)
        except subprocess.TimeoutExpired:
            core.kill()
            core.wait()
'''


@unittest.skipUnless(os.environ.get("NRV_TEST_ROS_SEQUENCE") == "1",
                     "Set NRV_TEST_ROS_SEQUENCE=1 for isolated real roscpp-to-rospy serialization")
class RealRosSequenceTests(unittest.TestCase):
    def test_roscpp_overwrites_zero_header_sequence_on_serialized_delivery(self):
        result = subprocess.run([
            "docker", "run", "--rm", "-i", "--network", "none",
            "-e", "ROS_MASTER_URI=http://127.0.0.1:11311", "-e", "ROS_HOSTNAME=127.0.0.1",
            os.environ.get("NRV_ROS_TEST_IMAGE", "nrv-demo:noetic"), "python3", "-"
        ], input=PROBE, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        evidence = next(json.loads(line) for line in result.stdout.splitlines()
                        if line.startswith('{"probe":'))
        self.assertEqual(evidence["assigned_header_seq"], 0)
        self.assertGreaterEqual(len(evidence["received_sequences"]), 20)
        print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    unittest.main()
