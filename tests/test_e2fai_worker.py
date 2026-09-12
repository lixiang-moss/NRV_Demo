from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ros_ws/src/nrv_demo/scripts"))
import numpy as np
import torch
import examples.e2fai_worker as worker
from e2fai_protocol import EVENT_DTYPE, ProtocolError, recv_packet, send_packet
from examples.e2fai_worker import BoundedEventQueue, SessionReceiver, run_session, validate_events, parse_args
from performance import PerfRecorder


def input_packet(session="s", batch=0):
    array = np.array([(0, 0, 0, 1), (1, 1, 50_000_000, 0),
                      (2, 2, 100_000_000, 1)], dtype=EVENT_DTYPE)
    meta = dict(session_id=session, batch_seq=batch, ros_header_seq=batch,
                header_stamp_ns=100_000_000, width=48, height=32, count=3,
                callback_ns=time.monotonic_ns())
    return meta, array


class FakeModel:
    def __init__(self):
        self.reset_inputs = []

    def forward_step(self, voxel, state):
        self.reset_inputs.append(state is None)
        return {"log_image": torch.zeros(1, 1, 32, 48),
                "flow": torch.ones(1, 2, 32, 48)}, torch.ones(1)


class QueueTests(unittest.TestCase):
    def batch(self, sequence, callback_ns):
        meta, array = input_packet(batch=sequence)
        meta['callback_ns'] = callback_ns
        return array, meta, callback_ns

    def test_capacity_catchup_trims_whole_prefix_before_insertion(self):
        mib = 1024 * 1024
        fifo = BoundedEventQueue()
        for seq in range(3):
            fifo.trim(1, (self.batch(seq, 1), 128 * mib))
        reason, dropped = fifo.trim(1, (self.batch(3, 1), 128 * mib))
        self.assertEqual(reason, 'queue_capacity')
        self.assertEqual(dropped, [2, 6, 256 * mib])
        self.assertEqual(fifo.bytes, 256 * mib)
        self.assertEqual([fifo.get()[1]['batch_seq'] for _ in range(2)], [2, 3])
        reason, dropped = fifo.trim(1, (self.batch(4, 1), 1024 * mib + 1))
        self.assertEqual(dropped[0], 1)
        self.assertEqual(fifo.bytes, 0)
        self.assertLess(fifo.peak_bytes, 1024 * mib)

    def test_freshness_uses_callback_age_and_retains_recent_suffix(self):
        fifo = BoundedEventQueue()
        for seq, callback in enumerate((0, 150_000_000, 200_000_000)):
            fifo.put(self.batch(seq, callback), 39)
        reason, dropped = fifo.trim(300_000_000)
        self.assertEqual(reason, 'queue_freshness')
        self.assertEqual(dropped, [2, 6, 78])
        self.assertEqual(fifo.get()[1]['batch_seq'], 2)
        # A just-received TCP packet can still be old at the source callback.
        _, dropped = fifo.trim(400_000_001, (self.batch(3, 0), 39))
        self.assertEqual(dropped[0], 1)
        self.assertIsNone(fifo.get())

    def test_batch_guard_and_disabled_freshness(self):
        fifo = BoundedEventQueue()
        for seq in range(255):
            fifo.put(self.batch(seq, 0), 39)
        _, dropped = fifo.trim(1_000_000_000, (self.batch(255, 0), 39), freshness=False)
        self.assertEqual(dropped[0], 128)
        self.assertEqual(fifo.get()[1]['batch_seq'], 128)
        self.assertEqual(fifo.peak_batches, 255)

    def test_window_and_preview_scale_defaults_and_override(self):
        required = ["--backbone", "backbone.ckpt", "--image-checkpoint", "image.pt", "--output-dir", "/tmp/unused"]
        args = parse_args(required)
        self.assertEqual((args.window_ms, args.flow_max_px, args.result_mode), (250.0, 50.0, "thread"))
        self.assertEqual(parse_args(required + ["--result-mode", "inline"]).result_mode, "inline")
        self.assertEqual(args.queue_batches, 256)
        self.assertTrue(args.freshness)
        self.assertFalse(parse_args(required + ["--no-freshness"]).freshness)
        self.assertEqual(parse_args(required + ["--queue-batches", "128"]).queue_batches, 128)
        self.assertEqual(parse_args(required + ["--window-ms", "100"]).flow_max_px, 20.0)
        self.assertEqual(parse_args(required + ["--flow-max-px", "7"]).flow_max_px, 7.0)

    def test_capacity_failure_does_not_replace_older_item(self):
        fifo = BoundedEventQueue(max_batches=2, max_bytes=20)
        fifo.put("first", 10)
        fifo.put("second", 10)
        with self.assertRaises(OverflowError):
            fifo.put("third", 1)
        self.assertEqual(fifo.get(), "first")
        self.assertEqual(fifo.get(), "second")
        self.assertIsNone(fifo.get())

    def test_default_queue_accepts_256_in_order_and_reports_actual_limit(self):
        fifo = BoundedEventQueue()
        for index in range(256):
            fifo.put(index, 1)
        with self.assertRaisesRegex(OverflowError, "256 batches / 1024 MiB"):
            fifo.put(256, 1)
        self.assertEqual([fifo.get() for _ in range(256)], list(range(256)))

    def test_byte_limit_is_independent_of_batch_limit(self):
        fifo = BoundedEventQueue(max_batches=100, max_bytes=10)
        with self.assertRaises(OverflowError):
            fifo.put("large", 11)
        self.assertIsNone(fifo.get())

    def test_default_byte_capacity_accepts_one_gib_and_rejects_next_byte(self):
        fifo = BoundedEventQueue()
        self.assertEqual(fifo.max_bytes, 1024 * 1024 * 1024)
        # Exercise size accounting without allocating a gigabyte of test data.
        fifo.put("first", 512 * 1024 * 1024)
        fifo.put("second", 512 * 1024 * 1024)
        with self.assertRaisesRegex(OverflowError, "1024 MiB"):
            fifo.put("overflow", 1)
        self.assertEqual([fifo.get(), fifo.get()], ["first", "second"])
        self.assertEqual(fifo.bytes, 0)

    def test_invalid_event_size_and_polarity_are_explicit(self):
        meta, array = input_packet()
        with self.assertRaises(ProtocolError):
            validate_events(meta, array.tobytes()[:-1])
        array["polarity"][0] = 2
        with self.assertRaises(ProtocolError):
            validate_events(meta, array.tobytes())


class SessionTests(unittest.TestCase):
    def test_normal_window_collection_is_not_counted_as_backlog(self):
        model = FakeModel()
        args = SimpleNamespace(window_ms=250, queue_batches=256, sensor_width=48,
                               sensor_height=32, flow_max_px=50, result_mode='thread')
        host, client = socket.socketpair()
        summaries = []
        thread = threading.Thread(target=lambda: summaries.append(run_session(
            host, model, torch.device('cpu'), args, self.perf, lambda: False)))
        thread.start()
        try:
            for batch, times in enumerate(([0, 100], [200, 250])):
                meta, array = input_packet(batch=batch)
                array = array[:2].copy()
                array['timestamp_ns'] = np.array(times, dtype=np.uint64) * 1_000_000
                meta['count'] = 2
                send_packet(client, 'events', meta, array.tobytes())
                if batch == 0:
                    time.sleep(.28)
            kind, meta, _ = recv_packet(client)
            self.assertEqual(kind, 'result')
            self.assertEqual(meta['event_count'], 3)
            self.assertEqual(meta['reset_count'], 0)
        finally:
            client.shutdown(socket.SHUT_RDWR)
            client.close()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(summaries[0]['catchup_count'], 0)

    def test_complete_window_wait_and_buffer_overflow_both_resume(self):
        for cause in ('window_freshness', 'window_buffer_overflow'):
            with self.subTest(cause=cause):
                model = FakeModel()
                forward = model.forward_step

                def slow_first(voxel, state):
                    output = forward(voxel, state)
                    if len(model.reset_inputs) == 1 and cause == 'window_freshness':
                        time.sleep(.27)
                    return output

                model.forward_step = slow_first
                buffer = worker.EventWindowBuffer(100_000_000)
                if cause == 'window_buffer_overflow':
                    buffer.max_buffer_bytes = 20
                args = SimpleNamespace(window_ms=100, queue_batches=256, sensor_width=48,
                                       sensor_height=32, flow_max_px=20, result_mode='thread')
                host, client = socket.socketpair()
                summaries = []
                with patch.object(worker, 'EventWindowBuffer', return_value=buffer):
                    thread = threading.Thread(target=lambda: summaries.append(run_session(
                        host, model, torch.device('cpu'), args, self.perf, lambda: False)))
                    thread.start()
                    try:
                        meta, _ = input_packet()
                        array = np.zeros(5, dtype=EVENT_DTYPE)
                        array['timestamp_ns'] = np.arange(5, dtype=np.uint64) * 50_000_000
                        meta['count'] = len(array)
                        send_packet(client, 'events', meta, array.tobytes())
                        while True:
                            kind, status, _ = recv_packet(client)
                            if kind == 'status' and status['state'] == 'catching_up':
                                self.assertEqual(status['reason'], cause)
                                break
                        buffer.max_buffer_bytes = 256 * 1024 * 1024
                        meta, array = input_packet(batch=1)
                        array['timestamp_ns'] += 300_000_000
                        send_packet(client, 'events', meta, array.tobytes())
                        while True:
                            kind, meta, _ = recv_packet(client)
                            if kind == 'result':
                                break
                        self.assertEqual(meta['source_batch_first'], 1)
                        self.assertEqual(meta['reset_reason'], cause)
                        self.assertTrue(all(model.reset_inputs))
                        while True:
                            kind, status, _ = recv_packet(client)
                            if kind == 'status' and status['state'] == 'connected':
                                break
                    finally:
                        client.shutdown(socket.SHUT_RDWR)
                        client.close()
                        thread.join(3)
                self.assertFalse(thread.is_alive())
                self.assertIsNone(summaries[0]['error'])
                self.assertEqual(summaries[0]['catchup_count'], 1)

    def test_overload_during_forward_discards_old_output_and_resets_gru(self):
        entered, release = threading.Event(), threading.Event()
        model = FakeModel()
        forward = model.forward_step

        def blocked(voxel, state):
            output = forward(voxel, state)
            if len(model.reset_inputs) == 1:
                entered.set()
                release.wait(2)
            return output

        model.forward_step = blocked
        args = SimpleNamespace(window_ms=100, queue_batches=2, sensor_width=48,
                               sensor_height=32, flow_max_px=20, result_mode='thread', freshness=False)
        host, client = socket.socketpair()
        summaries = []
        thread = threading.Thread(target=lambda: summaries.append(run_session(
            host, model, torch.device('cpu'), args, self.perf, lambda: False)))
        thread.start()
        try:
            for batch in range(3):
                meta, array = input_packet(batch=batch)
                meta.update(ros_header_seq=batch + 50, ros_sequence_valid=True)
                array['timestamp_ns'] += batch * 100_000_000
                send_packet(client, 'events', meta, array.tobytes())
                if batch == 0:
                    self.assertTrue(entered.wait(2))
            # Result thread reports catch-up even while forward is blocked.
            kind, status, _ = recv_packet(client)
            self.assertEqual((kind, status['state']), ('status', 'catching_up'))
            release.set()
            while True:
                kind, metadata, _ = recv_packet(client)
                if kind == 'result':
                    break
            self.assertEqual(metadata['source_batch_first'], 2)
            self.assertEqual(metadata['reset_reason'], 'queue_capacity')
            self.assertEqual(metadata['processing_generation'], 1)
            self.assertEqual(model.reset_inputs, [True, True])
            # Stop without draining status bytes: a peer TCP reset is still a
            # disconnect, not a model/protocol failure.
        finally:
            release.set()
            client.shutdown(socket.SHUT_RDWR)
            client.close()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(summaries[0]['error'])
        self.assertEqual(summaries[0]['catchup_stale_inferences'], 1)
        self.assertEqual(summaries[0]['ros_sequence_gap_incidents'], 0)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.perf = PerfRecorder(self.temp.name, "worker_test", enabled=False)

    def tearDown(self):
        self.perf.close()
        self.temp.cleanup()

    def test_receiver_overflow_keeps_latest_and_connection_alive(self):
        host, client = socket.socketpair()
        receiver = SessionReceiver(host, self.perf, max_batches=2, freshness=False)
        try:
            for batch in range(3):
                meta, array = input_packet(batch=batch)
                send_packet(client, "events", meta, array.tobytes())
            deadline = time.monotonic() + 2
            while receiver.received_batches < 3 and time.monotonic() < deadline:
                time.sleep(.001)
            self.assertEqual(receiver.received_batches, 3)
            self.assertFalse(receiver.failed.is_set())
            self.assertFalse(receiver.done.is_set())
            self.assertEqual(receiver.take()[1]["batch_seq"], 2)
            self.assertEqual(receiver.rejected_batches, 2)
            self.assertEqual(receiver.processing_generation, 1, "Consecutive drops merge before work resumes")
        finally:
            receiver.close()
            client.close()

    def test_reconnect_never_reuses_recurrent_state_or_session_id(self):
        for mode in ("inline", "thread"):
            with self.subTest(mode=mode):
                model = FakeModel()
                args = SimpleNamespace(window_ms=100, queue_batches=64, sensor_width=48,
                                       sensor_height=32, flow_max_px=20, result_mode=mode)
                for session in ("first", "second"):
                    host, client = socket.socketpair()
                    summaries = []
                    thread = threading.Thread(target=lambda: summaries.append(
                        run_session(host, model, torch.device("cpu"), args, self.perf, lambda: False)))
                    thread.start()
                    try:
                        meta, array = input_packet(session)
                        send_packet(client, "events", meta, array.tobytes())
                        kind, result_meta, payload = recv_packet(client)
                        self.assertEqual(kind, "result")
                        self.assertEqual(result_meta["session_id"], session)
                        self.assertEqual(result_meta["event_count"], 2)
                        self.assertEqual(result_meta["source_callback_ns"], meta["callback_ns"])
                        self.assertEqual(len(payload), 48 * 32 * 12)
                        flow = np.frombuffer(payload, dtype="<f4", offset=48 * 32 * 4).reshape(32, 48, 2)
                        np.testing.assert_array_equal(flow, np.ones_like(flow))
                    finally:
                        client.shutdown(socket.SHUT_RDWR)
                        client.close()
                        thread.join(3)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(summaries[0]["session_id"], session)
                    self.assertEqual(summaries[0]["partial_window_events_at_disconnect"], 1)
                self.assertEqual(model.reset_inputs, [True, True])

    def test_session_id_change_is_rejected(self):
        host, client = socket.socketpair()
        receiver = SessionReceiver(host, self.perf)
        try:
            for session in ("a", "b"):
                meta, array = input_packet(session)
                send_packet(client, "events", meta, array.tobytes())
            kind, meta, _ = recv_packet(client)
            self.assertEqual(kind, "error")
            self.assertEqual(meta["code"], "protocol_or_connection_error")
        finally:
            receiver.close()
            client.close()

    def test_time_discontinuity_resets_gru_before_next_segment(self):
        model = FakeModel()
        args = SimpleNamespace(window_ms=100, queue_batches=64, sensor_width=48, sensor_height=32, flow_max_px=20, result_mode="thread")
        host, client = socket.socketpair()
        thread = threading.Thread(target=lambda: run_session(
            host, model, torch.device("cpu"), args, self.perf, lambda: False))
        thread.start()
        try:
            meta, _ = input_packet()
            array = np.zeros(8, dtype=EVENT_DTYPE)
            array["timestamp_ns"] = np.array([0, 50, 100, 150, 200, 10, 60, 110], dtype=np.uint64) * 1_000_000
            meta["count"] = len(array)
            send_packet(client, "events", meta, array.tobytes())
            results = [recv_packet(client) for _ in range(3)]
            self.assertEqual([result[1]["reset_count"] for result in results], [0, 0, 1])
            self.assertEqual(model.reset_inputs, [True, False, True])
        finally:
            client.shutdown(socket.SHUT_RDWR)
            client.close()
            thread.join(3)
        self.assertFalse(thread.is_alive())

    def test_disabled_measurement_adds_no_cuda_sync_or_events(self):
        model = FakeModel()
        args = SimpleNamespace(window_ms=100, queue_batches=64, sensor_width=48, sensor_height=32, flow_max_px=20, result_mode="thread")
        host, client = socket.socketpair()
        with patch("examples.e2fai_worker.voxelize", return_value=torch.zeros(1, 15, 32, 48)), \
                patch("torch.cuda.synchronize") as synchronize, \
                patch("torch.cuda.Event") as event:
            thread = threading.Thread(target=lambda: run_session(
                host, model, torch.device("cuda:0"), args, self.perf, lambda: False))
            thread.start()
            try:
                meta, array = input_packet()
                send_packet(client, "events", meta, array.tobytes())
                self.assertEqual(recv_packet(client)[0], "result")
            finally:
                client.shutdown(socket.SHUT_RDWR)
                client.close()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            synchronize.assert_not_called()
            event.assert_not_called()

    def test_ros_gap_resets_gru_without_bridge_packet_gap(self):
        model = FakeModel()
        args = SimpleNamespace(window_ms=100, queue_batches=64, sensor_width=48, sensor_height=32, flow_max_px=20, result_mode="thread")
        host, client = socket.socketpair()
        summaries = []
        thread = threading.Thread(target=lambda: summaries.append(run_session(
            host, model, torch.device("cpu"), args, self.perf, lambda: False)))
        thread.start()
        try:
            for batch, sequence in enumerate((50, 51, 53)):
                meta, array = input_packet(batch=batch)
                meta.update(ros_header_seq=sequence, ros_sequence_valid=True)
                array["timestamp_ns"] += batch * 100_000_000
                send_packet(client, "events", meta, array.tobytes())
                result = recv_packet(client)
                self.assertEqual(result[0], "result")
                if batch == 2:
                    self.assertEqual(result[1]["reset_reason"], "ros_sequence_gap")
                    self.assertEqual(result[1]["reset_count"], 1)
            self.assertEqual(model.reset_inputs, [True, False, True])
        finally:
            client.shutdown(socket.SHUT_RDWR)
            client.close()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(summaries[0]["ros_sequence_gap_incidents"], 1)


class CapturingReceiver:
    def __init__(self):
        self.done = threading.Event()
        self.failed = threading.Event()
        self.packets = []
        self.failures = []
        self.condition = threading.Condition()

    def send(self, kind, metadata, payload=b""):
        if self.done.is_set():
            return False
        with self.condition:
            self.packets.append((kind, dict(metadata), bytes(payload)))
            self.condition.notify_all()
        return True

    def fail(self, code, message):
        self.failures.append((code, str(message)))
        self.failed.set()
        self.done.set()

    def wait_for_packets(self, count):
        with self.condition:
            return self.condition.wait_for(lambda: len(self.packets) >= count, timeout=2)


def result_input(index, offset=0, generation=0, size=4, session="result-test"):
    log_image = torch.arange(size * size, dtype=torch.float32).reshape(1, 1, size, size) + offset
    flow = torch.arange(2 * size * size, dtype=torch.float32).reshape(1, 2, size, size) / 10
    snapshot = worker.snapshot_output(dict(log_image=log_image, flow=flow))
    metadata = dict(session_id=session, window_id=index, source_batch_first=index,
                    source_batch_last=index, window_start_ns=index * 250_000_000,
                    window_end_ns=(index + 1) * 250_000_000, event_first_ns=index * 250_000_000,
                    event_last_ns=index * 250_000_000 + 100, event_count=2, width=size, height=size,
                    reset_reason="session_start" if not generation else "event_time_gap",
                    reset_count=generation, source_callback_ns=time.monotonic_ns())
    return snapshot, metadata


class ResultProcessorTests(unittest.TestCase):
    def test_catchup_invalidates_active_render_and_pending_result_without_stopping(self):
        receiver = CapturingReceiver()
        receiver.processing_generation = 0
        entered, release = threading.Event(), threading.Event()
        original = worker.render_result

        def render(snapshot, image_range, scale):
            if not entered.is_set():
                entered.set()
                release.wait(2)
            return original(snapshot, image_range, scale)

        with patch.object(worker, 'render_result', side_effect=render):
            processor = worker.ResultProcessor(receiver, self.perf, 20, 'thread', lambda: False)
            try:
                processor.submit(*result_input(1), events=2)
                self.assertTrue(entered.wait(2))
                processor.submit(*result_input(2), events=2)
                receiver.processing_generation = 1
                release.set()
                snapshot, metadata = result_input(3, offset=100, generation=1)
                metadata['processing_generation'] = 1
                processor.submit(snapshot, metadata, events=2)
                self.assertTrue(receiver.wait_for_packets(1))
                self.assertEqual([p[1]['window_id'] for p in receiver.packets], [3])
                self.assertEqual(receiver.packets[0][2], original(snapshot, worker.RunningRange(), 20))
                self.assertEqual(processor.stale_results, 2)
            finally:
                release.set()
                processor.close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.perf = PerfRecorder(self.temp.name, "result_test", enabled=False)

    def tearDown(self):
        self.perf.close()
        self.temp.cleanup()

    def test_inline_and_thread_keep_bytes_order_and_reset_display_range(self):
        inputs = [result_input(1), result_input(2, offset=100), result_input(3, offset=100, generation=1)]
        np.testing.assert_array_equal(inputs[0][0][0], np.arange(16, dtype=np.float32).reshape(4, 4))
        self.assertEqual(inputs[0][0][1].shape, (2, 4, 4))
        outputs = {}
        for mode in ("inline", "thread"):
            receiver = CapturingReceiver()
            processor = worker.ResultProcessor(receiver, self.perf, 20, mode, lambda: False)
            try:
                for snapshot, metadata in inputs:
                    self.assertTrue(processor.submit(snapshot, metadata, events=2))
                self.assertTrue(receiver.wait_for_packets(3))
            finally:
                processor.close()
            self.assertEqual(receiver.failures, [])
            self.assertEqual(processor.result_count, 3)
            self.assertEqual([packet[1]["window_id"] for packet in receiver.packets], [1, 2, 3])
            outputs[mode] = [packet[2] for packet in receiver.packets]
        self.assertEqual(outputs["inline"], outputs["thread"])
        self.assertEqual(outputs["thread"][1][:16], bytes([255]) * 16)
        fresh_range = worker.RunningRange()
        self.assertEqual(outputs["thread"][2], worker.render_result(inputs[2][0], fresh_range, 20))
        self.assertGreater(len(set(outputs["thread"][2][:16])), 4)

    def test_full_result_queue_wait_is_released_by_stop(self):
        receiver = CapturingReceiver()
        sending = threading.Event()
        stop = threading.Event()

        def blocked_send(_kind, _metadata, _payload=b""):
            sending.set()
            receiver.done.wait(2)
            return False

        receiver.send = blocked_send
        processor = worker.ResultProcessor(receiver, self.perf, 20, "thread", stop.is_set)
        submission = []
        submitter = None
        try:
            self.assertTrue(processor.submit(*result_input(1), events=2))
            self.assertTrue(sending.wait(2))
            self.assertTrue(processor.submit(*result_input(2), events=2))
            submitter = threading.Thread(target=lambda: submission.append(
                processor.submit(*result_input(3), events=2)))
            submitter.start()
            submitter.join(0.05)
            self.assertTrue(submitter.is_alive(), "A full one-item result FIFO must wait instead of dropping or growing")
            stop.set()
            receiver.done.set()
            submitter.join(2)
            self.assertFalse(submitter.is_alive())
            self.assertEqual(submission, [False])
        finally:
            stop.set()
            receiver.done.set()
            if submitter is not None:
                submitter.join(2)
            cleanup = processor.close()
        self.assertEqual(processor.result_count, 0)
        self.assertEqual(cleanup["pending_results_cleared_on_disconnect"], 2)
        self.assertEqual(cleanup["pending_result_events_cleared_on_disconnect"], 4)

    def test_disconnect_releases_blocked_socket_send_and_cleans_result_thread(self):
        host, client = socket.socketpair()
        host.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        receiver = SessionReceiver(host, self.perf)
        sending, send_finished = threading.Event(), threading.Event()
        original_send = receiver.send

        def tracked_send(kind, metadata, payload=b""):
            if kind == "result":
                sending.set()
            try:
                return original_send(kind, metadata, payload)
            finally:
                if kind == "result":
                    send_finished.set()

        receiver.send = tracked_send
        processor = worker.ResultProcessor(receiver, self.perf, 20, "thread", lambda: False)
        try:
            self.assertTrue(processor.submit(*result_input(1, size=64), events=2))
            self.assertTrue(sending.wait(2))
            self.assertFalse(send_finished.wait(0.05), "The deliberately small socket must block the result payload")
            client.shutdown(socket.SHUT_RDWR)
            client.close()
            self.assertTrue(send_finished.wait(2))
        finally:
            client.close()
            receiver.close()
            cleanup = processor.close()
        self.assertEqual(processor.result_count, 0)
        self.assertEqual(cleanup["pending_results_cleared_on_disconnect"], 1)
        self.assertEqual(cleanup["pending_result_events_cleared_on_disconnect"], 2)


if __name__ == "__main__":
    unittest.main()
