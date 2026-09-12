"""Qt offscreen GUI behavior; ROS messages/process are isolated test doubles."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    from PyQt5 import QtCore, QtTest, QtWidgets
except ImportError:
    QtWidgets = None

SCRIPTS = Path(__file__).resolve().parents[1] / 'ros_ws/src/nrv_demo/scripts'
sys.path.insert(0, str(SCRIPTS))


@unittest.skipIf(QtWidgets is None, 'PyQt5 unavailable; run in the ROS image')
class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name)
        sensor = self.output / 'sensor.txt'
        sensor.write_text('20:0167=1C\n20:0168=1B\n')
        self.params = {'~output_dir': str(self.output), '~profile_path': str(self.output / 'profile.json'),
                       '~sensor_setting_path': str(sensor), '~auto_start': False,
                       '~duration': 0, '~serial_number': '', '~device_index': '0',
                       '~message_threshold_time_ms': 10, '~image_fps': 10}
        rospy = types.ModuleType('rospy')
        rospy.get_param = lambda key, default=None: self.params.get(key, default)
        rospy.set_param = lambda key, value: self.params.update({key: value})
        rospy.Subscriber = lambda *args, **kwargs: types.SimpleNamespace(unregister=lambda: None)
        rospy.is_shutdown = lambda: False
        modules = {'rospy': rospy}
        for package, name in [('sensor_msgs', 'Image'), ('std_msgs', 'String'), ('nrv_demo', 'E2faiResult')]:
            modules[package] = types.ModuleType(package)
            modules[package + '.msg'] = types.ModuleType(package + '.msg')
            setattr(modules[package + '.msg'], name, object)
        spec = importlib.util.spec_from_file_location('testable_gui', SCRIPTS / 'gui.py')
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(self.module)
        self.window = self.module.NoiseWindow()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.temporary.cleanup()

    def test_one_to_three_views_and_immediate_persistence(self):
        self.assertEqual(self.window.view_grid.count(), 3)
        old_bias = self.window.active_bias['0167']
        self.window.bias['0167'].setValue(old_bias + 1)
        for key in ('e2fai_flow', 'e2fai_gray'):
            self.window.view_actions[key].setChecked(False)
        self.assertEqual(self.window.visible_views(), ['raw'])
        self.assertEqual(self.window.view_grid.count(), 1)
        self.assertFalse(self.window.view_actions['raw'].isEnabled())
        saved = json.loads((self.output / 'profile.json').read_text())
        self.assertEqual(saved['views'], ['raw'])
        self.assertEqual(saved['bias']['0167'], old_bias)
        self.assertNotIn('filters', saved)
        self.assertTrue(self.window.apply_button.isEnabled())
        for count, key in enumerate(('e2fai_gray', 'e2fai_flow'), 2):
            self.window.view_actions[key].setChecked(True)
            self.assertEqual(self.window.view_grid.count(), count)

    def test_old_profile_restores_three_views_and_keeps_apply_semantics(self):
        self.window.set_profile(dict(bias={'0167': 12, '0168': 13},
                                     filters=dict(background=True, refractory=True, window_ms=8., interval_ms=2.)))
        self.window.update_pending()
        self.assertEqual(len(self.window.visible_views()), 3)
        self.assertTrue(self.window.apply_button.isEnabled())
        self.assertNotEqual(self.window.active_bias['0167'], 12)
        self.window.apply_parameters()
        self.assertEqual(self.window.active_bias, {'0167': 12, '0168': 13})
        saved = json.loads((self.output / 'profile.json').read_text())
        self.assertEqual(saved['bias']['0167'], 12)
        self.window.set_profile(dict(bias={'0167': 12, '0168': 13}, views=['algorithm']))
        self.assertEqual(len(self.window.visible_views()), 3)
        self.window.set_profile(dict(bias={'0167': 12, '0168': 13}, views=['raw', 'algorithm']))
        self.assertEqual(self.window.visible_views(), ['raw'])

    def test_no_events_and_missing_worker_do_not_block_qt(self):
        called = []
        QtCore.QTimer.singleShot(10, lambda: called.append(True))
        QtTest.QTest.qWait(130)
        self.assertEqual(called, [True])
        self.window.settings_button.click()
        self.assertFalse(self.window.settings_panel.isHidden())

    def test_only_current_session_result_enters_latest_cache(self):
        self.window.session_id = 'current'
        message = types.SimpleNamespace(session_id='old', window_id=1, source_callback_ns=0, window_end_ns=1,
                                        gray=object(), flow_preview=object())
        self.window.on_model_result(message)
        self.assertFalse(self.window.frames)
        message.session_id = 'current'
        self.window.on_model_result(message)
        self.assertEqual(set(self.window.frames), {'e2fai_gray', 'e2fai_flow'})
        self.window.stop()
        self.assertIsNone(self.window.session_id)
        self.assertFalse(self.window.frames)
        self.window.on_model_result(message)
        self.assertFalse(self.window.frames)

    def test_raw_latest_overwrites_are_counted(self):
        with patch.object(self.window.perf, 'increment') as increment:
            self.window.on_image(object(), 'raw')
            self.window.on_image(object(), 'raw')
            self.assertEqual([call.args[0] for call in increment.call_args_list],
                             ['latest_frame_overwrites_raw'])
        self.window.frames.clear()

    def test_arriving_images_do_not_resize_panels_but_window_can_resize(self):
        from PyQt5 import QtGui
        self.window.show()
        self.app.processEvents()
        original = {key: view.geometry() for key, view in self.window.views.items()}
        for key in ('raw', 'e2fai_gray', 'e2fai_flow', 'raw'):
            view = self.window.views[key]
            pixmap = QtGui.QPixmap(960, 720)
            pixmap.fill(QtCore.Qt.black)
            view.setPixmap(pixmap.scaled(view.contentsRect().size(), QtCore.Qt.KeepAspectRatio))
            self.app.processEvents()
            self.assertEqual({k: v.geometry() for k, v in self.window.views.items()}, original)
        self.window.resize(1620, 920)
        self.app.processEvents()
        self.assertNotEqual(self.window.views['raw'].geometry(), original['raw'])

    def test_natural_finish_clears_pending_old_results_and_model_status(self):
        self.window.session_id = 'finished'
        self.window.latest_model_status = {'state': 'connected'}
        self.window.frames['e2fai_gray'] = (object(), 0, {})
        self.window.frames['e2fai_flow'] = (object(), 0, {})
        self.window.finished()
        self.assertIsNone(self.window.session_id)
        self.assertIsNone(self.window.latest_model_status)
        self.assertFalse(self.window.frames)
        self.assertIn('stopped', self.window.model_status.text())

    def test_nonobject_json_profile_uses_existing_load_error_dialog(self):
        malformed = self.output / 'bad.json'
        malformed.write_text('[]')
        original_bias = self.window.bias_values()
        with patch.object(self.module.Q.QFileDialog, 'getOpenFileName', return_value=(str(malformed), '')), \
                patch.object(self.module.Q.QMessageBox, 'warning') as warning:
            self.window.load_profile()
            warning.assert_called_once()
        self.assertEqual(self.window.bias_values(), original_bias)

    def test_start_generates_new_session_and_async_stop(self):
        class Process:
            def __init__(self):
                self.current = QtCore.QProcess.NotRunning
                self.calls = []

            def state(self):
                return self.current

            def start(self, command, arguments):
                self.calls.append((command, arguments))
                self.current = QtCore.QProcess.Running

            def processId(self):
                return 123456

        process = Process()
        self.window.process = process
        self.window.start()
        first = self.window.session_id
        with patch.object(self.module.os, 'kill') as kill:
            self.window.stop()
            kill.assert_called_once()
        self.assertIsNone(self.window.session_id)
        process.current = QtCore.QProcess.NotRunning
        self.window.finished()
        self.window.start()
        self.assertNotEqual(first, self.window.session_id)
        self.assertTrue(any(a.startswith('session_id:=') for a in process.calls[1][1]))
        with patch.object(self.module.os, 'kill'):
            self.window.close()
        self.assertTrue(self.window.closing)
        process.current = QtCore.QProcess.NotRunning
        self.window.finished()
        self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
