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
import uuid

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from nrv_demo.msg import E2faiResult
from PyQt5 import QtCore, QtGui, QtWidgets as Q
from performance import PerfRecorder


VIEW_SOURCES = [('raw', 'Original rendering'),
                ('e2fai_gray', 'E2FAI reconstruction'), ('e2fai_flow', 'E2FAI optical flow')]
WINDOW_OPTIONS_MS = (50, 100, 150, 200, 250)
RESOLUTION_OPTIONS = ((960, 720), (640, 480), (384, 288))


class NoiseWindow(Q.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('NRV Demo — E2FAI')
        self.resize(1420, 820)
        self.lock = threading.Lock()
        self.frames = {}
        self.raw_color_table = [QtGui.qRgb(0, 0, 0)] * 256
        self.raw_color_table[0] = QtGui.qRgb(255, 0, 0)
        self.raw_color_table[255] = QtGui.qRgb(0, 0, 255)
        self.latest_status = None
        self.last_frame = 0
        self.restart_pending = False
        self.closing = False
        self.stopping = False
        self.session_id = None
        self.latest_model_status = None
        self.e2fai_enabled = rospy.get_param('~e2fai_enabled', True)
        self.output = Path(rospy.get_param('~output_dir'))
        self.output.mkdir(parents=True, exist_ok=True)
        self.perf = PerfRecorder(self.output, 'gui', enabled=rospy.get_param('~perf_enabled', False),
                                 interval_s=rospy.get_param('~perf_interval_s', 5.0),
                                 capacity=rospy.get_param('~perf_capacity', 4096))
        self.profile_path = Path(rospy.get_param('~profile_path', '/output/last_applied_parameters.json'))
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
        self.views_button = Q.QToolButton()
        self.views_button.setText('Views (3)')
        self.views_button.setPopupMode(Q.QToolButton.InstantPopup)
        self.views_menu = Q.QMenu(self.views_button)
        self.views_button.setMenu(self.views_menu)
        self.view_actions = {}
        for key, title in VIEW_SOURCES:
            action = self.views_menu.addAction(title)
            action.setCheckable(True)
            action.setChecked(True)
            self.view_actions[key] = action
        bar.addWidget(self.views_button)
        self.settings_button = Q.QPushButton('Settings')
        self.settings_button.setCheckable(True)
        self.settings_button.setToolTip('Show or hide camera parameters')
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
        panel.addWidget(hardware)
        processing = Q.QGroupBox('E2FAI processing')
        processing_form = Q.QFormLayout(processing)
        self.window_ms = Q.QComboBox()
        for value in WINDOW_OPTIONS_MS:
            self.window_ms.addItem('{} ms'.format(value), value)
        default_window = int(rospy.get_param('~window_ms', 200))
        window_index = self.window_ms.findData(default_window)
        self.window_ms.setCurrentIndex(window_index if window_index >= 0 else self.window_ms.findData(200))
        self.resolution = Q.QComboBox()
        for width, height in RESOLUTION_OPTIONS:
            self.resolution.addItem('{} × {}'.format(width, height), '{}x{}'.format(width, height))
        default_resolution = (int(rospy.get_param('~processing_width', 960)),
                              int(rospy.get_param('~processing_height', 720)))
        resolution_index = self.resolution.findData('{}x{}'.format(*default_resolution))
        self.resolution.setCurrentIndex(
            resolution_index if resolution_index >= 0 else self.resolution.findData('960x720'))
        self.window_ms.setToolTip('Shorter windows reduce collection delay but increase inference frequency.')
        self.resolution.setToolTip('Changes decoded events, voxel tensors, model outputs and the raw preview.')
        self.voxel_mode = Q.QComboBox()
        self.voxel_mode.addItem('Compatibility (event span)', 'event_span')
        self.voxel_mode.addItem('Incremental (fixed window)', 'fixed_window')
        default_voxel_mode = rospy.get_param('~voxel_mode', 'fixed_window')
        voxel_mode_index = self.voxel_mode.findData(default_voxel_mode)
        self.voxel_mode.setCurrentIndex(
            voxel_mode_index if voxel_mode_index >= 0 else self.voxel_mode.findData('fixed_window'))
        self.voxel_mode.setToolTip(
            'Compatibility keeps the original event-span normalization. Incremental overlaps voxel filling with inference.')
        self.catchup_enabled = Q.QCheckBox('Automatic catch-up')
        self.catchup_enabled.setChecked(bool(rospy.get_param('~catchup_enabled', True)))
        self.catchup_enabled.setToolTip(
            'Use the 800 ms and 1.5 GiB proactive triggers. Hard limits always discard oldest data and keep running.')
        processing_form.addRow('Time window', self.window_ms)
        processing_form.addRow('Processing resolution', self.resolution)
        processing_form.addRow('Voxel mode', self.voxel_mode)
        processing_form.addRow('Host queue', self.catchup_enabled)
        panel.addWidget(processing)
        self.apply_button = Q.QPushButton('Apply parameters')
        self.apply_button.setToolTip('Apply all edits. Processing or camera changes restart acquisition.')
        self.apply_button.clicked.connect(self.apply_parameters)
        self.parameter_status = Q.QLabel('Parameters applied')
        panel.addWidget(self.apply_button)
        panel.addWidget(self.parameter_status)
        save = Q.QPushButton('Save profile')
        load = Q.QPushButton('Load profile')
        save.clicked.connect(self.save_profile)
        load.clicked.connect(self.load_profile)
        panel.addWidget(save)
        panel.addWidget(load)
        panel.addStretch()
        body.addWidget(controls)
        self.views = {}
        self.view_panels = {}
        self.view_grid = Q.QGridLayout()
        body.addLayout(self.view_grid, 1)
        for key, title in VIEW_SOURCES:
            container = Q.QWidget()
            column = Q.QVBoxLayout(container)
            column.addWidget(Q.QLabel(title))
            view = Q.QLabel('Waiting for camera data')
            view.setAlignment(QtCore.Qt.AlignCenter)
            view.setMinimumSize(320, 240)
            view.setStyleSheet('background:#111827;color:#cbd5e1;border:1px solid #334155;')
            column.addWidget(view, 1)
            self.views[key] = view
            self.view_panels[key] = container
        self.relayout_views()
        # Set after parenting/styling: pixmaps must not become layout size hints.
        for view in self.views.values():
            view.setSizePolicy(Q.QSizePolicy.Ignored, Q.QSizePolicy.Ignored)
        for action in self.view_actions.values():
            action.toggled.connect(self.change_views)
        self.metrics = Q.QLabel('RAW: — MB/s   Packets: —   RAW gaps: —')
        layout.addWidget(self.metrics)
        self.model_status = Q.QLabel('E2FAI: waiting for connection' if self.e2fai_enabled else 'E2FAI: disabled')
        layout.addWidget(self.model_status)
        self.log = Q.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        self.log.document().setMaximumBlockCount(150)
        layout.addWidget(self.log)
        self.subscribers = [
            rospy.Subscriber('/delta_renderer/image', Image, self.on_image, callback_args='raw', queue_size=1),
            rospy.Subscriber('/nrv_capture/status', String, self.on_status, queue_size=1),
            rospy.Subscriber('/nrv_e2fai/result', E2faiResult, self.on_model_result,
                             queue_size=1, buff_size=64 * 1024 * 1024),
            rospy.Subscriber('/nrv_e2fai/status', String, self.on_model_status, queue_size=1),
        ]
        if self.profile_path.exists():
            try:
                self.set_profile(json.loads(self.profile_path.read_text()))
                self.log.appendPlainText('Restored last applied parameters.')
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.log.appendPlainText('Could not restore saved parameters: ' + str(error))
        requested_views = rospy.get_param('~visible_views', [])
        if requested_views:
            self.set_profile(dict(bias=self.bias_values(), views=requested_views))
        self.active_bias = self.bias_values()
        self.active_processing = self.processing_values()
        for control in self.bias.values():
            control.valueChanged.connect(self.update_pending)
        self.window_ms.currentIndexChanged.connect(self.update_pending)
        self.resolution.currentIndexChanged.connect(self.update_pending)
        self.voxel_mode.currentIndexChanged.connect(self.update_pending)
        self.catchup_enabled.toggled.connect(self.update_pending)
        self.update_pending()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)
        if rospy.get_param('~auto_start', True):
            QtCore.QTimer.singleShot(0, self.start)

    def bias_values(self):
        return {address: widget.value() for address, widget in self.bias.items()}

    def visible_views(self):
        return [key for key, _ in VIEW_SOURCES if self.view_actions[key].isChecked()]

    def processing_values(self):
        width, height = self.resolution.currentData().split('x')
        return dict(window_ms=int(self.window_ms.currentData()), width=int(width), height=int(height),
                    voxel_mode=str(self.voxel_mode.currentData()),
                    catchup_enabled=self.catchup_enabled.isChecked())

    def relayout_views(self):
        while self.view_grid.count():
            self.view_grid.takeAt(0)
        visible = self.visible_views()
        for key, panel in self.view_panels.items():
            panel.setVisible(key in visible)
        for index, key in enumerate(visible):
            self.view_grid.addWidget(self.view_panels[key], 0, index)
        for key, action in self.view_actions.items():
            action.setEnabled(len(visible) > 1 or key not in visible)
        self.views_button.setText('Views ({})'.format(len(visible)))

    def change_views(self):
        if not self.visible_views():
            action = self.view_actions['raw']
            action.blockSignals(True)
            action.setChecked(True)
            action.blockSignals(False)
        self.relayout_views()
        if hasattr(self, 'active_bias'):
            self.persist_applied_profile()

    def persist_applied_profile(self):
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(json.dumps(dict(bias=self.active_bias,
                                                     processing=self.active_processing,
                                                     views=self.visible_views()), indent=2) + '\n')

    def update_pending(self):
        pending = (self.bias_values() != self.active_bias
                   or self.processing_values() != self.active_processing)
        self.apply_button.setEnabled(pending)
        self.parameter_status.setText('Pending changes — click Apply parameters' if pending else 'Parameters applied')
        self.parameter_status.setWordWrap(True)

    def write_settings(self):
        text = self.sensor_text
        for address, value in self.active_bias.items():
            # Change only the threshold's low six bits; keep the source setting's other bits.
            pattern = r'(20:' + address + r'=)([0-9a-fA-F]+)'
            text = re.sub(pattern, lambda m: m[1] + format((int(m[2], 16) & ~63) | value, '02X'), text, flags=re.I)
        path = self.output / 'sensor_settings.txt'
        path.write_text(text)
        return path

    def start(self):
        if self.process.state() != QtCore.QProcess.NotRunning:
            return
        self.stopping = False
        settings = self.write_settings()
        self.session_id = uuid.uuid4().hex
        args = ['nrv_demo', 'demo.launch', 'show_gui:=false', 'sensor_setting_path:=' + str(settings),
                'session_id:=' + self.session_id,
                'window_ms:=' + str(self.active_processing['window_ms']),
                'processing_width:=' + str(self.active_processing['width']),
                'processing_height:=' + str(self.active_processing['height']),
                'voxel_mode:=' + self.active_processing['voxel_mode'],
                'catchup_enabled:=' + str(self.active_processing['catchup_enabled']).lower()]
        for name in ('serial_number', 'device_index', 'duration', 'output_dir', 'message_threshold_time_ms', 'image_fps'):
            args.append(name + ':=' + str(rospy.get_param('~' + name)))
        for name, default in (('e2fai_enabled', True), ('e2fai_host', '127.0.0.1'),
                              ('e2fai_port', 8765), ('perf_enabled', False), ('start_driver', True),
                              ('perf_interval_s', 5.0), ('perf_capacity', 4096)):
            value = rospy.get_param('~' + name, default)
            args.append(name + ':=' + (str(value).lower() if isinstance(value, bool) else str(value)))
        with self.lock:
            self.frames.clear()
            self.latest_status = None
            self.latest_model_status = None
            self.last_frame = 0
            self.model_generation = 0
        for view in self.views.values():
            view.clear()
            view.setText('Waiting for camera data')
        self.metrics.setText('RAW: — MB/s   Packets: —   RAW gaps: —')
        self.model_status.setText('E2FAI: connecting' if self.e2fai_enabled else 'E2FAI: disabled')
        self.process.start('roslaunch', args)
        self.state.setText('Starting')
        self.start_button.setEnabled(False)

    def stop(self):
        self.restart_pending = False
        self.interrupt()

    def interrupt(self):
        # Stop accepting in-flight results before acquisition termination completes.
        with self.lock:
            self.session_id = None
            self.latest_model_status = None
            for key in ('e2fai_gray', 'e2fai_flow'):
                self.frames.pop(key, None)
        self.model_status.setText('E2FAI: acquisition stopped' if self.e2fai_enabled else 'E2FAI: disabled')
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.stopping = True
            self.state.setText('Stopping')
            if self.process.processId():
                try:
                    os.kill(int(self.process.processId()), signal.SIGINT)
                except ProcessLookupError:
                    pass
            else:
                QtCore.QTimer.singleShot(50, self.interrupt)

    def apply_parameters(self):
        bias_changed = self.bias_values() != self.active_bias
        processing_changed = self.processing_values() != self.active_processing
        self.active_bias = self.bias_values()
        self.active_processing = self.processing_values()
        self.write_settings()
        self.persist_applied_profile()
        self.update_pending()
        if (bias_changed or processing_changed) and self.process.state() != QtCore.QProcess.NotRunning:
            self.log.appendPlainText('Applying parameters: restarting acquisition and model state.')
            self.restart_pending = True
            self.interrupt()
        else:
            self.log.appendPlainText('Parameters applied.')

    def finished(self, *_):
        with self.lock:
            self.session_id = None
            self.latest_model_status = None
            for key in ('e2fai_gray', 'e2fai_flow'):
                self.frames.pop(key, None)
        self.model_status.setText('E2FAI: acquisition stopped' if self.e2fai_enabled else 'E2FAI: disabled')
        self.state.setText('Stopped')
        self.start_button.setEnabled(True)
        if self.closing:
            QtCore.QTimer.singleShot(0, self.close)
        elif self.restart_pending:
            self.restart_pending = False
            QtCore.QTimer.singleShot(0, self.start)

    def read_log(self):
        text = bytes(self.process.readAllStandardOutput()).decode(errors='replace')
        text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
        text = re.sub(r'\x1b\][^\x07]*\x07', '', text)
        self.log.appendPlainText(text.rstrip())

    def on_image(self, message, key):
        with self.lock:
            if key in self.frames:
                self.perf.increment('latest_frame_overwrites_' + key)
            self.frames[key] = (message, self.perf.tick(), {})
            self.last_frame = time.monotonic()

    def on_status(self, message):
        try:
            status = json.loads(message.data)
        except (ValueError, TypeError):
            return
        with self.lock:
            self.latest_status = status

    def on_model_result(self, message):
        received = self.perf.tick()
        with self.lock:
            if self.session_id is None or message.session_id != self.session_id:
                self.perf.increment('stale_session_results')
                return
            generation = getattr(message, 'processing_generation', 0)
            if generation < getattr(self, 'model_generation', 0):
                self.perf.increment('catchup_stale_results')
                return
            self.model_generation = generation
            if 'e2fai_gray' in self.frames or 'e2fai_flow' in self.frames:
                self.perf.increment('latest_result_overwrites')
            metadata = dict(window_id=message.window_id, source_callback_ns=message.source_callback_ns,
                            window_end_ns=message.window_end_ns, processing_generation=generation)
            self.frames['e2fai_gray'] = (message.gray, received, metadata)
            self.frames['e2fai_flow'] = (message.flow_preview, received, metadata)

    def on_model_status(self, message):
        try:
            status = json.loads(message.data)
        except (ValueError, TypeError):
            return
        with self.lock:
            if self.session_id is not None and status.get('session_id') == self.session_id:
                generation = status.get('processing_generation', 0)
                if generation < getattr(self, 'model_generation', 0):
                    return
                if generation > getattr(self, 'model_generation', 0):
                    for key in ('e2fai_gray', 'e2fai_flow'):
                        self.frames.pop(key, None)
                self.model_generation = generation
                self.latest_model_status = status

    def _image_from_message(self, message, key):
        if key == 'raw' and message.encoding == 'mono8':
            image = QtGui.QImage(bytes(message.data), message.width, message.height,
                                 message.step, QtGui.QImage.Format_Indexed8).copy()
            image.setColorTable(self.raw_color_table)
            return image
        formats = {'bgr8': QtGui.QImage.Format_RGB888,
                   'rgb8': QtGui.QImage.Format_RGB888,
                   'mono8': QtGui.QImage.Format_Grayscale8}
        if message.encoding not in formats:
            return None
        image = QtGui.QImage(bytes(message.data), message.width, message.height,
                             message.step, formats[message.encoding]).copy()
        return image.rgbSwapped() if message.encoding == 'bgr8' else image

    def refresh(self):
        if self.perf.enabled:
            refresh_ns = self.perf.tick()
            previous_refresh = getattr(self, '_previous_refresh_ns', None)
            if previous_refresh is not None:
                interval_ms = (refresh_ns - previous_refresh) / 1e6
                self.perf.observe('refresh_interval', interval_ms)
                self.perf.observe('refresh_timer_lateness', max(0, interval_ms - 100))
            self._previous_refresh_ns = refresh_ns
        if rospy.is_shutdown():
            self.close()
            return
        with self.lock:
            frames, self.frames = self.frames, {}
            status = self.latest_status
            model_status = self.latest_model_status
            last = self.last_frame
        for key, (message, received, metadata) in frames.items():
            if metadata.get('processing_generation', getattr(self, 'model_generation', 0)) < getattr(self, 'model_generation', 0):
                continue
            if not self.view_actions[key].isChecked():
                continue
            self.perf.elapsed('receive_to_refresh', received, source=key, **metadata)
            tick = self.perf.tick()
            image = self._image_from_message(message, key)
            if image is None:
                self.log.appendPlainText('Unsupported image encoding: ' + message.encoding)
                continue
            view = self.views[key]
            pixmap = QtGui.QPixmap.fromImage(image).scaled(view.contentsRect().size(), QtCore.Qt.KeepAspectRatio,
                                                         QtCore.Qt.SmoothTransformation)
            # Status can arrive on a ROS thread while image conversion runs.
            # Recheck atomically with presentation; keep scaling outside the lock.
            with self.lock:
                if metadata.get('processing_generation', getattr(self, 'model_generation', 0)) < getattr(self, 'model_generation', 0):
                    continue
                view.setPixmap(pixmap)
            self.perf.elapsed('image_convert_scale_setpixmap', tick, source=key, **metadata)
            self.perf.increment('presentations_' + key)
            if self.perf.enabled and metadata.get('source_callback_ns'):
                self.perf.observe('callback_to_setpixmap',
                                  (time.monotonic_ns() - metadata['source_callback_ns']) / 1e6,
                                  source=key, window_id=metadata['window_id'])
                # A mapped-time proxy can be negative when the decoder clock
                # jumps into the future; it is never physical sensor latency.
                self.perf.observe('mapped_window_age_at_setpixmap',
                                  (rospy.Time.now().to_nsec() - metadata['window_end_ns']) / 1e6,
                                  source=key, window_id=metadata['window_id'])
        if model_status:
            text = 'E2FAI: {}   Results: {}   Queue: {}'.format(
                '正在追赶' if model_status['state'] == 'catching_up' else model_status['state'],
                model_status['results'], model_status['queue_batches'])
            if model_status.get('detail'):
                text += '   ' + model_status['detail']
            self.model_status.setText(text)
        if self.process.state() == QtCore.QProcess.Running and not self.stopping:
            active = '{} ms, {} × {}'.format(
                self.active_processing['window_ms'], self.active_processing['width'],
                self.active_processing['height'])
            active += ', {}'.format('incremental voxel' if self.active_processing['voxel_mode'] == 'fixed_window'
                                    else 'compatibility voxel')
            self.state.setText(('Streaming — ' + active) if last and time.monotonic() - last < 2 else
                               'Waiting for data (check the camera and log below)')
        if status:
            self.metrics.setText('RAW: {:.2f} MB/s   Packets: {:,}   RAW gaps: {}'.format(
                status['bytes_per_second'] / 1e6, status['raw_packets'], status['raw_sequence_gaps']))

    def save_profile(self):
        path, _ = Q.QFileDialog.getSaveFileName(self, 'Save profile', str(self.output / 'camera_profile.json'), 'JSON (*.json)')
        if path:
            Path(path).write_text(json.dumps(dict(bias={a: w.value() for a, w in self.bias.items()},
                                               processing=self.processing_values(),
                                               views=self.visible_views()), indent=2) + '\n')

    def set_profile(self, config):
        if not isinstance(config, dict):
            raise ValueError('Profile must be a JSON object')
        processing = config.get('processing')
        if processing is not None:
            if not isinstance(processing, dict):
                raise ValueError('processing must be an object')
            try:
                window_ms = int(processing['window_ms'])
                resolution = (int(processing['width']), int(processing['height']))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError('processing requires integer window_ms, width and height') from error
            if window_ms not in WINDOW_OPTIONS_MS or resolution not in RESOLUTION_OPTIONS:
                raise ValueError('processing contains an unsupported window or resolution')
            catchup_enabled = processing.get('catchup_enabled', self.catchup_enabled.isChecked())
            if type(catchup_enabled) is not bool:
                raise ValueError('processing catchup_enabled must be a boolean')
            voxel_mode = processing.get('voxel_mode', self.voxel_mode.currentData())
            if voxel_mode not in ('event_span', 'fixed_window'):
                raise ValueError('processing voxel_mode must be event_span or fixed_window')
        visible = config.get('views', [key for key, _ in VIEW_SOURCES])
        if not isinstance(visible, list) or any(not isinstance(key, str) for key in visible):
            raise ValueError('views must be a list of image sources')
        # Profiles from the centroid demo still load; its removed view/filter
        # settings cannot affect E2FAI input or the current layout.
        visible = [key for key in visible if key != 'algorithm']
        if not visible:
            visible = [key for key, _ in VIEW_SOURCES]
        if (any(key not in self.view_actions for key in visible) or len(set(visible)) != len(visible)):
            raise ValueError('views must contain one to three distinct known image sources')
        for address, widget in self.bias.items():
            widget.setValue(config['bias'][address])
        if processing is not None:
            self.window_ms.setCurrentIndex(self.window_ms.findData(window_ms))
            self.resolution.setCurrentIndex(
                self.resolution.findData('{}x{}'.format(*resolution)))
            self.voxel_mode.setCurrentIndex(self.voxel_mode.findData(voxel_mode))
            self.catchup_enabled.setChecked(catchup_enabled)
        for key, action in self.view_actions.items():
            action.blockSignals(True)
            action.setChecked(key in visible)
            action.blockSignals(False)
        self.relayout_views()
        if hasattr(self, 'active_bias'):
            self.persist_applied_profile()

    def load_profile(self):
        path, _ = Q.QFileDialog.getOpenFileName(self, 'Load profile', '/output', 'JSON (*.json)')
        if path:
            try:
                config = json.loads(Path(path).read_text())
                self.set_profile(config)
                self.update_pending()
                self.log.appendPlainText('Profile loaded. Click Apply parameters to apply the changes.')
            except (OSError, ValueError, KeyError, TypeError) as error:
                Q.QMessageBox.warning(self, 'Cannot load profile', str(error))

    def closeEvent(self, event):
        self.closing = True
        self.stop()
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.state.setText('Closing — waiting for acquisition to stop')
            self.setEnabled(False)
            event.ignore()
            return
        self.timer.stop()
        for subscriber in self.subscribers:
            subscriber.unregister()
        self.perf.close()
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
