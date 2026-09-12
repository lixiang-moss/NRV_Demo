"""Isolated-container integration: real ROS/Qt/roslaunch, no camera or GPU required.

Run alone in a container, not against an existing workstation ROS master.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import xmlrpc.client

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import rospy
from PyQt5 import QtCore, QtTest, QtWidgets
from nrv_demo.msg import E2faiResult
from sensor_msgs.msg import Image

SCRIPTS = Path(__file__).resolve().parents[1] / 'ros_ws/src/nrv_demo/scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('nrv_live_gui', SCRIPTS / 'gui.py')
gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gui)


class LiveGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master = subprocess.Popen(['roscore'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                with xmlrpc.client.ServerProxy('http://localhost:11311') as server:
                    server.getPid('/gui_test')
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError('Test ROS master did not start')
        rospy.init_node('nrv_gui_test', disable_signals=True)

    @classmethod
    def tearDownClass(cls):
        rospy.signal_shutdown('test finished')
        cls.master.terminate()
        cls.master.wait(timeout=15)

    def pump_until(self, predicate, seconds=15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not predicate():
            QtTest.QTest.qWait(25)
        self.assertTrue(predicate())

    def test_real_ros_results_and_acquisition_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            sensor = Path(directory) / 'sensor.txt'
            sensor.write_text('20:0167=1C\n20:0168=1B\n')
            params = dict(output_dir=directory, profile_path=directory + '/profile.json',
                          sensor_setting_path=str(sensor), auto_start=False, start_driver=False,
                          e2fai_enabled=True, e2fai_port=28765, duration=0,
                          serial_number='', device_index='0', message_threshold_time_ms=10, image_fps=10)
            for name, value in params.items():
                rospy.set_param('~' + name, value)
            window = gui.NoiseWindow()
            publisher = rospy.Publisher('/nrv_e2fai/result', E2faiResult, queue_size=1)
            ticks = []
            responsiveness = QtCore.QTimer()
            responsiveness.timeout.connect(lambda: ticks.append(time.monotonic()))
            responsiveness.start(20)
            try:
                window.show()
                window.start()
                self.pump_until(lambda: window.process.state() == QtCore.QProcess.Running)
                self.pump_until(lambda: publisher.get_num_connections() > 0)
                self.pump_until(lambda: window.latest_model_status is not None)
                first = window.session_id
                result = E2faiResult(session_id='stale', window_id=1)
                image = Image(height=1, width=1, encoding='mono8', step=1, data=b'\x80')
                result.gray = image
                result.flow_preview = Image(height=1, width=1, encoding='rgb8', step=3, data=b'\xff\x00\x00')
                publisher.publish(result)
                QtTest.QTest.qWait(200)
                self.assertIsNone(window.views['e2fai_gray'].pixmap())
                result.session_id = first
                publisher.publish(result)
                self.pump_until(lambda: window.views['e2fai_gray'].pixmap() is not None)
                before = len(ticks)
                window.stop()
                self.assertIsNone(window.session_id)
                self.pump_until(lambda: window.process.state() == QtCore.QProcess.NotRunning, 25)
                self.assertGreater(len(ticks), before)
                window.start()
                self.pump_until(lambda: window.process.state() == QtCore.QProcess.Running)
                self.assertNotEqual(first, window.session_id)
                publisher.publish(result)  # In-flight result from previous acquisition.
                QtTest.QTest.qWait(200)
                self.assertIsNone(window.views['e2fai_gray'].pixmap())
                before = len(ticks)
                window.close()
                self.pump_until(lambda: window.process.state() == QtCore.QProcess.NotRunning, 25)
                self.assertGreater(len(ticks), before)
                self.assertTrue(window.closing)
            finally:
                responsiveness.stop()
                publisher.unregister()
                window.close()
                if window.process.state() != QtCore.QProcess.NotRunning:
                    self.pump_until(lambda: window.process.state() == QtCore.QProcess.NotRunning, 25)


if __name__ == '__main__':
    unittest.main()
