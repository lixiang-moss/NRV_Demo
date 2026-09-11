#!/usr/bin/env python3
"""Single-window NRV controls; roslaunch owns and restarts the entire pipeline."""
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from PyQt5 import QtCore, QtGui, QtWidgets as Q


class NoiseWindow(Q.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('NRV Demo — Camera & Noise Controls')
        self.resize(1420, 820)
        self.lock = threading.Lock()
        self.frames = {}
        self.latest_status = None
        self.last_frame = 0
        self.restart_pending = False
        self.closing = False
        self.stopping = False
        self.output = Path(rospy.get_param('~output_dir'))
        self.output.mkdir(parents=True, exist_ok=True)
        self.sensor_text = Path(rospy.get_param('~sensor_setting_path')).read_text()
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_log)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(lambda _: self.state.setText(self.process.errorString()))
        root = Q.QWidget()
        self.setCentralWidget(root)
        layout = Q.QVBoxLayout(root)
        bar = Q.QHBoxLayout()
        self.start_button = Q.QPushButton('Start')
        self.stop_button = Q.QPushButton('Stop')
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.state = Q.QLabel('Stopped')
        for widget in (self.start_button, self.stop_button, self.state):
            bar.addWidget(widget)
        bar.addStretch()
        self.settings_button = Q.QPushButton('Settings')
        self.settings_button.setCheckable(True)
        self.settings_button.setToolTip('Show or hide camera and noise parameters')
        bar.addWidget(self.settings_button)
        layout.addLayout(bar)
        body = Q.QHBoxLayout()
        layout.addLayout(body, 1)
        controls = Q.QWidget()
        self.settings_panel = controls
        controls.hide()
        self.settings_button.toggled.connect(controls.setVisible)
        controls.setMaximumWidth(340)
        panel = Q.QVBoxLayout(controls)
        hardware = Q.QGroupBox('Camera bias')
        form = Q.QFormLayout(hardware)
        self.bias = {}
        for address, label in [('0167', 'ON (0x0167)'), ('0168', 'OFF (0x0168)')]:
            value = int(re.search(r'20:' + address + r'=([0-9a-fA-F]+)', self.sensor_text, re.I)[1], 16)
            control = Q.QSpinBox()
            control.setRange(0, 63)
            control.setValue(value & 63)
            control.setToolTip('Decimal register code, not a physical threshold. A higher code does not necessarily reduce noise.')
            self.bias[address] = control
            form.addRow(label, control)
        note = Q.QLabel('Register codes (decimal). Change one step at a time and compare the images.')
        note.setWordWrap(True)
        form.addRow(note)
        apply_button = Q.QPushButton('Apply && restart')
        apply_button.clicked.connect(self.apply_bias)
        form.addRow(apply_button)
        panel.addWidget(hardware)
        software = Q.QGroupBox('Live software filters')
        form = Q.QFormLayout(software)
        self.background = Q.QCheckBox('Neighbour filter')
        self.window = Q.QDoubleSpinBox()
        self.window.setRange(0.1, 100)
        self.window.setValue(5)
        self.window.setSuffix(' ms')
        self.refractory = Q.QCheckBox('Pixel interval filter')
        self.interval = Q.QDoubleSpinBox()
        self.interval.setRange(0.1, 100)
        self.interval.setValue(1)
        self.interval.setSuffix(' ms')
        form.addRow(self.background)
        form.addRow('Neighbour window', self.window)
        form.addRow(self.refractory)
        form.addRow('Minimum interval', self.interval)
        for control in (self.background, self.refractory):
            control.toggled.connect(self.update_filters)
        for control in (self.window, self.interval):
            control.valueChanged.connect(self.update_filters)
        note = Q.QLabel('Left: original. Right: filtered events + centroid.\nSoftware filters leave the RAW topic unchanged.')
        note.setWordWrap(True)
        form.addRow(note)
        panel.addWidget(software)
        save = Q.QPushButton('Save profile')
        load = Q.QPushButton('Load profile')
        save.clicked.connect(self.save_profile)
        load.clicked.connect(self.load_profile)
        panel.addWidget(save)
        panel.addWidget(load)
        panel.addStretch()
        body.addWidget(controls)
        self.views = {}
        for key, title in [('raw', 'Original rendering'), ('algorithm', 'Filtered + centroid')]:
            column = Q.QVBoxLayout()
            column.addWidget(Q.QLabel(title))
            view = Q.QLabel('Waiting for camera data')
            view.setAlignment(QtCore.Qt.AlignCenter)
            view.setMinimumSize(320, 240)
            view.setStyleSheet('background:#111827;color:#cbd5e1;border:1px solid #334155;')
            column.addWidget(view, 1)
            body.addLayout(column, 1)
            self.views[key] = view
        self.metrics = Q.QLabel('Events/s: —   Retained: —   RAW gaps: —')
        layout.addWidget(self.metrics)
        self.log = Q.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        self.log.document().setMaximumBlockCount(150)
        layout.addWidget(self.log)
        self.subscribers = [
            rospy.Subscriber('/delta_renderer/image', Image, self.on_image, callback_args='raw', queue_size=1),
            rospy.Subscriber('/nrv_python_demo/image', Image, self.on_image, callback_args='algorithm', queue_size=1),
            rospy.Subscriber('/nrv_python_demo/status', String, self.on_status, queue_size=1),
        ]
        self.update_filters()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)
        QtCore.QTimer.singleShot(0, self.start)

    def update_filters(self):
        rospy.set_param('/nrv_noise', dict(background=self.background.isChecked(),
                        window_ms=self.window.value(), refractory=self.refractory.isChecked(),
                        interval_ms=self.interval.value()))

    def write_settings(self):
        text = self.sensor_text
        for address, widget in self.bias.items():
            # Change only the threshold's low six bits; keep the source setting's other bits.
            pattern = r'(20:' + address + r'=)([0-9a-fA-F]+)'
            value = widget.value()
            text = re.sub(pattern, lambda m: m[1] + format((int(m[2], 16) & ~63) | value, '02X'), text, flags=re.I)
        path = self.output / 'sensor_settings.txt'
        path.write_text(text)
        return path

    def start(self):
        if self.process.state() != QtCore.QProcess.NotRunning:
            return
        self.stopping = False
        settings = self.write_settings()
        args = ['nrv_demo', 'demo.launch', 'show_gui:=false', 'sensor_setting_path:=' + str(settings)]
        for name in ('serial_number', 'device_index', 'duration', 'output_dir', 'message_threshold_time_ms', 'image_fps'):
            args.append(name + ':=' + str(rospy.get_param('~' + name)))
        with self.lock:
            self.frames.clear()
            self.latest_status = None
            self.last_frame = 0
        for view in self.views.values():
            view.clear()
            view.setText('Waiting for camera data')
        self.metrics.setText('Events/s: —   Retained: —   RAW gaps: —')
        self.process.start('roslaunch', args)
        self.state.setText('Starting')
        self.start_button.setEnabled(False)

    def stop(self):
        self.restart_pending = False
        self.interrupt()

    def interrupt(self):
        if self.process.state() == QtCore.QProcess.Running:
            self.stopping = True
            self.state.setText('Stopping')
            os.kill(int(self.process.processId()), signal.SIGINT)

    def apply_bias(self):
        if self.process.state() == QtCore.QProcess.NotRunning:
            self.start()
        else:
            self.restart_pending = True
            self.interrupt()

    def finished(self, *_):
        self.state.setText('Stopped')
        self.start_button.setEnabled(True)
        if self.restart_pending and not self.closing:
            self.restart_pending = False
            QtCore.QTimer.singleShot(0, self.start)

    def read_log(self):
        text = bytes(self.process.readAllStandardOutput()).decode(errors='replace')
        text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
        text = re.sub(r'\x1b\][^\x07]*\x07', '', text)
        self.log.appendPlainText(text.rstrip())

    def on_image(self, message, key):
        with self.lock:
            self.frames[key] = message
            self.last_frame = time.monotonic()

    def on_status(self, message):
        with self.lock:
            self.latest_status = json.loads(message.data)

    def refresh(self):
        if rospy.is_shutdown():
            self.close()
            return
        with self.lock:
            frames, self.frames = self.frames, {}
            status = self.latest_status
            last = self.last_frame
        for key, message in frames.items():
            formats = {'bgr8': QtGui.QImage.Format_RGB888, 'rgb8': QtGui.QImage.Format_RGB888,
                       'mono8': QtGui.QImage.Format_Grayscale8}
            if message.encoding not in formats:
                self.log.appendPlainText('Unsupported image encoding: ' + message.encoding)
                continue
            image = QtGui.QImage(bytes(message.data), message.width, message.height,
                                 message.step, formats[message.encoding]).copy()
            if message.encoding == 'bgr8':
                image = image.rgbSwapped()
            view = self.views[key]
            view.setPixmap(QtGui.QPixmap.fromImage(image).scaled(view.size(), QtCore.Qt.KeepAspectRatio,
                                                                QtCore.Qt.SmoothTransformation))
        if self.process.state() == QtCore.QProcess.Running and not self.stopping:
            self.state.setText('Streaming' if last and time.monotonic() - last < 2 else
                               'Waiting for data (check the camera and log below)')
        if status:
            total = status['decoded_events']
            retained = 100 * status['filtered_events'] / total if total else 0
            self.metrics.setText('Events/s: {:,.0f}   Retained (session): {:.1f}%   RAW gaps: {}'.format(
                status['events_per_second'], retained, status['raw_sequence_gaps']))

    def save_profile(self):
        path, _ = Q.QFileDialog.getSaveFileName(self, 'Save profile', str(self.output / 'noise_profile.json'), 'JSON (*.json)')
        if path:
            Path(path).write_text(json.dumps(dict(bias={a: w.value() for a, w in self.bias.items()},
                                               filters=rospy.get_param('/nrv_noise')), indent=2) + '\n')

    def load_profile(self):
        path, _ = Q.QFileDialog.getOpenFileName(self, 'Load profile', '/output', 'JSON (*.json)')
        if path:
            try:
                config = json.loads(Path(path).read_text())
                for address, widget in self.bias.items():
                    widget.setValue(config['bias'][address])
                filters = config['filters']
                self.background.setChecked(filters['background'])
                self.window.setValue(filters['window_ms'])
                self.refractory.setChecked(filters['refractory'])
                self.interval.setValue(filters['interval_ms'])
                self.update_filters()
                self.log.appendPlainText('Profile loaded. Click Apply & restart for camera bias.')
            except (OSError, ValueError, KeyError, TypeError) as error:
                Q.QMessageBox.warning(self, 'Cannot load profile', str(error))

    def closeEvent(self, event):
        self.closing = True
        self.stop()
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.process.waitForFinished(20000)
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.closing = False
            event.ignore()
            return
        event.accept()


if __name__ == '__main__':
    rospy.init_node('nrv_gui', disable_signals=True)
    app = Q.QApplication(sys.argv)
    window = NoiseWindow()
    signal.signal(signal.SIGINT, lambda *_: window.close())
    signal.signal(signal.SIGTERM, lambda *_: window.close())
    window.show()
    app.exec_()
    rospy.signal_shutdown('GUI closed')
