import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from examples.e2fai_realtime import (
    BRIDGE_HEADER, BRIDGE_V2_METADATA, COMPACT_EVENT_DTYPE, EVENT_DTYPE,
    EventTcpReceiver, EventWindowBuffer, join_events, pipeline_diagnostics, unpack_compact_events,
)


class CompactEventsTests(unittest.TestCase):
    def test_matches_legacy_interpolation_exactly(self):
        start = 1_789_149_000_000_000_019
        for count in (0, 1, 2, 7, 10003):
            with self.subTest(count=count):
                packed = np.zeros(count, dtype=COMPACT_EVENT_DTYPE)
                packed["x"] = np.arange(count) % 960
                packed["y"] = np.arange(count) % 720
                packed["polarity"] = np.arange(count) % 2
                span = 10_000_003
                actual = unpack_compact_events(packed.tobytes(), count, start, start + span)
                for field in ("x", "y", "polarity"):
                    np.testing.assert_array_equal(actual[field], packed[field])
                expected = [start + (span * i // (count - 1) if count > 1 else 0) for i in range(count)]
                np.testing.assert_array_equal(actual["timestamp_ns"], np.asarray(expected, dtype=np.uint64))

    def test_invalid_metadata(self):
        for count, start, end, payload in [
            (-1, 0, 1, b""), (1, 10, 9, bytes(5)),
            (1, 0, 100_000_001, bytes(5)), (1, 0, 1, bytes(4)),
            (1, 0, 1 << 64, bytes(5)),
        ]:
            with self.subTest(count=count, start=start, end=end):
                with self.assertRaises(ValueError):
                    unpack_compact_events(payload, count, start, end)


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.stopped = threading.Event()
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(0.05)
        self.receiver = EventTcpReceiver(server, self.stopped.is_set)
        self.client = socket.create_connection(server.getsockname(), timeout=1)

    def tearDown(self):
        self.stopped.set()
        self.client.close()
        self.receiver.close()

    def packet(self, sequence=0, missing=0):
        packed = np.array([(959, 719, 1), (10, 20, 0)], dtype=COMPACT_EVENT_DTYPE)
        return (BRIDGE_HEADER.pack(b"NRV2", 960, 720, 2, sequence)
                + BRIDGE_V2_METADATA.pack(100, 200, missing, 0) + packed.tobytes())

    def test_fragmented_v2_and_raw_gap_notification(self):
        payload = self.packet()
        for index in range(0, len(payload), 3):
            self.client.sendall(payload[index:index + 3])
        _, width, height, events, gap = self.receiver.get(timeout=2)
        self.assertEqual((width, height), (960, 720))
        self.assertFalse(gap)
        np.testing.assert_array_equal(events["timestamp_ns"], [100, 200])
        self.client.sendall(self.packet(sequence=1, missing=3))
        self.assertTrue(self.receiver.get(timeout=2)[-1])
        self.assertEqual(self.receiver.dropped_batches, 0)
        self.assertEqual(self.receiver.adapter_raw_missing_reported, 3)
        self.assertEqual(self.receiver.received_bytes, len(payload) * 2)

    def test_legacy_and_transport_gap(self):
        expected = np.array([(3, 4, 100, 1), (5, 6, 200, 0)], dtype=EVENT_DTYPE)
        for sequence in (10, 13):
            self.client.sendall(BRIDGE_HEADER.pack(b"NRV1", 960, 720, 2, sequence) + expected.tobytes())
            events = self.receiver.get(timeout=2)[3]
            np.testing.assert_array_equal(events, expected)
        self.assertEqual(self.receiver.dropped_batches, 2)
        self.assertEqual(self.receiver.protocols, {"NRV1"})

    def test_full_queue_can_stop(self):
        self.client.sendall(b"".join(self.packet(sequence=n) for n in range(40)))
        deadline = time.monotonic() + 2
        while self.receiver.received_batches < 33 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.receiver.batches.qsize(), 32)
        self.stopped.set()
        self.receiver.close()
        self.assertFalse(self.receiver.thread.is_alive())
        self.assertEqual(self.receiver.dropped_batches, 0)


class DiagnosticsTests(unittest.TestCase):
    def test_missing_summary_is_not_zero_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(pipeline_diagnostics(Path(directory))["upstream_loss_detected"])

    def test_adapter_loss_is_visible_even_if_raw_monitor_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(json.dumps({"raw_sequence_gaps": 0, "status": "PASS"}))
            (root / "adapter_summary.json").write_text(json.dumps({"raw_missing_packets": 17}))
            self.assertTrue(pipeline_diagnostics(root)["upstream_loss_detected"])


class WindowTests(unittest.TestCase):
    def test_fragmented_windows_preserve_every_event_and_boundary(self):
        rng = np.random.default_rng(42)
        events = np.zeros(10003, dtype=EVENT_DTYPE)
        events["timestamp_ns"] = np.arange(len(events), dtype=np.uint64) * 10
        events["x"] = np.arange(len(events)) % 960
        events["polarity"] = np.arange(len(events)) % 2
        buffer = EventWindowBuffer(1000)
        windows = []
        offset = 0
        while offset < len(events):
            cut = offset + int(rng.integers(1, 703))
            complete, reset = buffer.push(events[offset:cut], 960, 720)
            self.assertFalse(reset)
            windows.extend(complete)
            offset = cut
        self.assertEqual(len(windows), 100)
        for index, window in enumerate(windows):
            np.testing.assert_array_equal(window, events[index * 100:(index + 1) * 100])
        self.assertEqual(join_events(windows + [buffer.events]).tobytes(), events.tobytes())

    def test_time_gap_resets_cached_chunks(self):
        buffer = EventWindowBuffer(100)
        first = np.array([(1, 2, 0, 1), (2, 3, 50, 0)], dtype=EVENT_DTYPE)
        second = first.copy()
        second["timestamp_ns"] += 1000
        buffer.push(first, 960, 720)
        windows, reset = buffer.push(second, 960, 720)
        self.assertTrue(reset)
        self.assertEqual(windows, [])
        np.testing.assert_array_equal(buffer.events, second)

    def test_byte_join_accepts_strided_records(self):
        data = np.array([(i, i, i, i % 2) for i in range(10)], dtype=EVENT_DTYPE)
        chunks = [data[::2], data[1::2]]
        self.assertEqual(join_events(chunks).tobytes(), np.concatenate(chunks).tobytes())


if __name__ == "__main__":
    unittest.main()
