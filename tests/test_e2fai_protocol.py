import json
from pathlib import Path
import socket
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros_ws/src/nrv_demo/scripts"))

import numpy as np
from e2fai_protocol import (EVENT_DTYPE, HEADER, MAGIC, MAX_PAYLOAD_BYTES,
                            ProtocolError, recv_packet, send_packet)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.left, self.right = socket.socketpair()

    def tearDown(self):
        self.left.close()
        self.right.close()

    def test_original_nanoseconds_survive_lossless_round_trip(self):
        events = np.array([(959, 719, 1_800_000_000_000_000_019, 1),
                           (0, 0, 1_800_000_000_000_000_020, 0)], dtype=EVENT_DTYPE)
        meta = {"session_id": "sample", "batch_seq": 42, "count": 2}
        send_packet(self.left, "events", meta, events.tobytes())
        kind, actual_meta, payload = recv_packet(self.right)
        self.assertEqual(kind, "events")
        self.assertEqual(actual_meta, meta)
        self.assertEqual(EVENT_DTYPE.itemsize, 13)
        np.testing.assert_array_equal(np.frombuffer(payload, dtype=EVENT_DTYPE), events)

    def test_fragmented_frame_keeps_partial_bytes(self):
        encoded = json.dumps({"kind": "error", "meta": {"message": "test"}}).encode()
        frame = HEADER.pack(MAGIC, len(encoded), 0) + encoded
        thread = threading.Thread(target=lambda: [self.left.sendall(frame[i:i+3])
                                                   for i in range(0, len(frame), 3)])
        thread.start()
        self.assertEqual(recv_packet(self.right), ("error", {"message": "test"}, b""))
        thread.join()

    def test_clean_eof_and_partial_eof_are_distinct(self):
        self.left.shutdown(socket.SHUT_WR)
        self.assertIsNone(recv_packet(self.right))
        self.left.close()
        self.right.close()
        self.left, self.right = socket.socketpair()
        self.left.sendall(b"NEF")
        self.left.shutdown(socket.SHUT_WR)
        with self.assertRaises(EOFError):
            recv_packet(self.right)

    def test_oversized_or_unknown_header_is_rejected_before_payload(self):
        for header in (HEADER.pack(b"NRV2", 1, 0), HEADER.pack(MAGIC, 1, MAX_PAYLOAD_BYTES + 1)):
            self.left.sendall(header)
            with self.assertRaises(ProtocolError):
                recv_packet(self.right)


if __name__ == "__main__":
    unittest.main()
