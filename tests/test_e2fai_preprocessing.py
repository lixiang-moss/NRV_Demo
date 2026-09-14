from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT / "ros_ws/src/nrv_demo/scripts"))
import numpy as np
import torch
from e2fai_protocol import EVENT_DTYPE
from nrv_e2fai.preprocessing import EventWindowBuffer, voxelize


def metadata(batch=0, session="s"):
    return dict(session_id=session, batch_seq=batch, ros_header_seq=batch,
                header_stamp_ns=0, width=48, height=32, count=0, callback_ns=1000 + batch)


def events(times):
    array = np.zeros(len(times), dtype=EVENT_DTYPE)
    array["timestamp_ns"] = times
    array["x"] = np.arange(len(times)) % 48
    array["y"] = np.arange(len(times)) % 32
    array["polarity"] = np.arange(len(times)) % 2
    return array


class WindowTests(unittest.TestCase):
    def test_defaults_use_200ms_windows_but_300ms_gap(self):
        buffer = EventWindowBuffer()
        first = events([0, 199_999_999, 200_000_000])
        complete = buffer.push(first, metadata())
        self.assertEqual(complete[0].events.tobytes(), first[:2].tobytes())
        self.assertEqual(complete[0].metadata["window_end_ns"], 200_000_000)
        buffer.push(events([500_000_000]), metadata(1))
        self.assertEqual(buffer.reset_count, 0)  # Exactly 300 ms is not a gap.
        buffer.push(events([800_000_001]), metadata(2))
        self.assertEqual(buffer.reset_reason, "event_time_gap")
        self.assertEqual(buffer.reset_count, 1)

    def test_half_open_boundaries_and_source_batch_provenance(self):
        buffer = EventWindowBuffer(100)
        first, second = events([0, 50, 99]), events([100, 150, 199, 200])
        self.assertEqual(buffer.push(first, metadata()), [])
        complete = buffer.push(second, metadata(1))
        self.assertEqual(len(complete), 2)
        np.testing.assert_array_equal(complete[0].events, first)
        np.testing.assert_array_equal(complete[1].events, second[:3])
        self.assertEqual(complete[0].metadata["window_end_ns"], 100)
        self.assertEqual(complete[1].metadata["window_start_ns"], 100)
        self.assertEqual(complete[1].metadata["source_callback_ns"], 1001)
        self.assertEqual(buffer.buffered_events, 1)

    def test_fragmented_input_preserves_every_event(self):
        buffer = EventWindowBuffer(100)
        original = events(np.arange(1003, dtype=np.uint64))
        output = []
        for batch, start in enumerate(range(0, len(original), 37)):
            output.extend(buffer.push(original[start:start+37], metadata(batch)))
        restored = np.concatenate([window.events for window in output] + [p[0] for p in buffer.parts])
        self.assertEqual(restored.tobytes(), original.tobytes())

    def test_intra_batch_reversal_preserves_order_and_resets(self):
        buffer = EventWindowBuffer(100)
        original = events([0, 50, 100, 10, 60, 110])
        complete = buffer.push(original, metadata())
        self.assertEqual(len(complete), 2)
        np.testing.assert_array_equal(complete[0].events, original[:2])
        np.testing.assert_array_equal(complete[1].events, original[3:5])
        self.assertEqual(complete[1].metadata["reset_reason"], "event_time_reversed")
        self.assertEqual(complete[1].metadata["reset_count"], 1)
        self.assertEqual(buffer.discarded_partial_events, 1)

    def test_intra_batch_large_forward_jump_has_no_empty_window_loop(self):
        buffer = EventWindowBuffer(100)
        complete = buffer.push(events([0, 50, 100, 10**18, 10**18 + 50, 10**18 + 100]), metadata())
        self.assertEqual(len(complete), 2)
        self.assertEqual(complete[-1].metadata["reset_reason"], "event_time_gap")
        self.assertEqual(buffer.empty_windows, 0)

    def test_cross_batch_gap_sequence_gap_and_empty_batch(self):
        buffer = EventWindowBuffer(100, max_gap_ns=300)
        buffer.push(events([10, 50]), metadata())
        self.assertEqual(buffer.push(events([]), metadata(1)), [])
        complete = buffer.push(events([1000, 1050, 1100]), metadata(2))
        self.assertEqual(complete[0].metadata["reset_reason"], "event_time_gap")
        complete = buffer.push(events([1200, 1250, 1300]), metadata(4))
        self.assertEqual(complete[0].metadata["reset_reason"], "batch_sequence_gap")
        self.assertEqual(buffer.reset_count, 2)

    def test_equal_timestamp_order_is_unchanged(self):
        buffer = EventWindowBuffer(100)
        original = events([100, 100, 100, 200])
        complete = buffer.push(original, metadata())
        self.assertEqual(complete[0].events.tobytes(), original[:3].tobytes())

    def test_stalled_timestamp_buffer_fails_at_capacity_without_sampling(self):
        buffer = EventWindowBuffer(100, max_buffer_bytes=3 * 13)
        original = events([0, 0, 0])
        buffer.push(original, metadata())
        with self.assertRaises(OverflowError):
            buffer.push(events([0]), metadata(1))
        self.assertEqual(buffer.buffered_events, 3)
        self.assertEqual(buffer.parts[0][0].tobytes(), original.tobytes())

    def test_ros_gap_resets_even_when_bridge_sequence_is_contiguous(self):
        buffer = EventWindowBuffer(100)
        first = dict(metadata(0), ros_header_seq=50, ros_sequence_valid=True)
        second = dict(metadata(1), ros_header_seq=52, ros_sequence_valid=True)
        buffer.push(events([0, 50]), first)
        complete = buffer.push(events([100, 150, 200]), second)
        self.assertEqual(len(complete), 1)
        self.assertEqual(complete[0].metadata["reset_reason"], "ros_sequence_gap")
        self.assertEqual(complete[0].metadata["window_start_ns"], 100)
        self.assertEqual(buffer.discarded_partial_events, 2)
        self.assertEqual(buffer.ros_sequence_gap_incidents, 1)
        self.assertEqual(buffer.ros_sequence_forward_missing, 1)

    def test_ros_uint32_wrap_is_contiguous(self):
        buffer = EventWindowBuffer(100)
        for batch, sequence in enumerate(((1 << 32) - 2, (1 << 32) - 1, 0, 1)):
            buffer.push(events([batch * 50]), dict(metadata(batch),
                        ros_header_seq=sequence, ros_sequence_valid=True))
        self.assertEqual(buffer.reset_count, 0)
        self.assertEqual(buffer.ros_sequence_gap_incidents, 0)
        self.assertEqual(buffer.ros_sequence_tracking, "active")

    def test_constant_zero_or_unconfirmed_ros_sequences_are_not_false_gaps(self):
        for mark_valid in (False, True):
            buffer = EventWindowBuffer(100)
            for batch in range(5):
                buffer.push(events([batch * 50]), dict(metadata(batch),
                            ros_header_seq=0, ros_sequence_valid=mark_valid))
            self.assertEqual(buffer.ros_sequence_gap_incidents, 0)
            self.assertEqual(buffer.reset_count, 0)
            self.assertEqual(buffer.ros_sequence_unknown_batches, 5)

    def test_duplicate_and_backward_ros_sequences_reset_without_huge_missing_count(self):
        buffer = EventWindowBuffer(100)
        for batch, sequence in enumerate((50, 51, 51, 2)):
            buffer.push(events([batch * 50]), dict(metadata(batch),
                        ros_header_seq=sequence, ros_sequence_valid=True))
        self.assertEqual(buffer.ros_sequence_gap_incidents, 2)
        self.assertEqual(buffer.ros_sequence_forward_missing, 0)
        self.assertEqual(buffer.reset_count, 2)


class VoxelTests(unittest.TestCase):
    def test_native_full_event_algorithm_matches_source(self):
        # Execute only the source's pure voxel function; no research transport.
        source = subprocess.check_output(["git", "show", "7d2fb86:examples/e2fai_realtime.py"], cwd=ROOT, text=True)
        code = source[source.index("def voxelize("):source.index("\ndef label(")]
        namespace = {"np": np, "torch": torch, "NUM_BINS": 15}
        exec(compile(code, "7d2fb86_voxelize", "exec"), namespace)
        original = events(np.arange(2000, dtype=np.uint64) + 1_800_000_000_000_000_000)
        actual = voxelize(original, 48, 32, torch.device("cpu"), 48, 32)
        expected = namespace["voxelize"](original, 48, 32, torch.device("cpu"), len(original), 48, 32)
        self.assertTrue(torch.equal(actual, expected))
        # More than the research branch's 1M limit must remain unsampled.
        many = events(np.zeros(1_000_003, dtype=np.uint64))
        many["polarity"] = 1
        voxel = voxelize(many, 48, 32, torch.device("cpu"), 48, 32)
        self.assertAlmostEqual(voxel.sum().item(), len(many), delta=1.0)

    def test_reversed_timestamps_are_rejected_instead_of_sorted(self):
        with self.assertRaises(ValueError):
            voxelize(events([3, 2, 1]), 48, 32, torch.device("cpu"), 48, 32)


if __name__ == "__main__":
    unittest.main()
