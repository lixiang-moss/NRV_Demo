import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros_ws/src/nrv_demo/scripts"))
from performance import PerfRecorder


class PerformanceTests(unittest.TestCase):
    def test_disabled_has_no_clock_thread_file_or_samples(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("performance.time.perf_counter_ns", side_effect=AssertionError("clock")), \
                mock.patch("performance.threading.Thread", side_effect=AssertionError("thread")):
            recorder = PerfRecorder(Path(directory) / "unused", "disabled")
            self.assertEqual(recorder.tick(), 0)
            recorder.elapsed("stage", 0, events=100)
            recorder.observe("stage", 10)
            recorder.increment("batches")
            recorder.close()
            self.assertFalse(hasattr(recorder, "_metrics"))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_bounded_quantiles_keep_full_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PerfRecorder(directory, "example", True, interval_s=60, capacity=3)
            for value in [1, 2, 3, 4, 5]:
                recorder.observe("work", value, events=value, batch=value)
            recorder.close()
            recorder.close()
            records = [json.loads(line) for line in recorder.path.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            metric = records[0]["metrics"]["work"]
            self.assertEqual((metric["count"], metric["sample_count"], metric["overflow"]), (5, 3, 2))
            self.assertEqual((metric["events"], metric["sample_events"], metric["sum"]), (15, 12, 15))
            self.assertEqual((metric["min"], metric["max"], metric["mean"], metric["p50"]), (1, 5, 3, 4))
            self.assertAlmostEqual(metric["p95"], 4.9)
            self.assertAlmostEqual(metric["p99"], 4.98)
            self.assertEqual(metric["metadata_last"], {"batch": 5})
            self.assertEqual([row[:2] for row in metric["samples"]], [[3, 3], [4, 4], [5, 5]])
            self.assertTrue(all(records[0]["interval_start_ns"] <= row[2] <= records[0]["interval_end_ns"]
                                for row in metric["samples"]))
            self.assertEqual(metric["sample_metadata"], [{"batch": 3}, {"batch": 4}, {"batch": 5}])

    def test_periodic_output_exists_before_close_and_idle_periods(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PerfRecorder(directory, "periodic", True, interval_s=.02)
            try:
                recorder.observe("work", 2)
                deadline = time.monotonic() + 2
                while (not recorder.path.exists() or not recorder.path.read_text().endswith("\n")) \
                        and time.monotonic() < deadline:
                    time.sleep(.005)
                records = [json.loads(line) for line in recorder.path.read_text().splitlines()]
                self.assertFalse(records[0]["final"])
                self.assertEqual(records[0]["metrics"]["work"]["count"], 1)
            finally:
                recorder.close()

    def test_concurrent_observers_and_counter_are_not_lost(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PerfRecorder(directory, "threads", True, interval_s=60, capacity=64)
            def observe():
                for _ in range(200):
                    recorder.observe("work", 1, events=2)
                    recorder.increment("batches")
            threads = [threading.Thread(target=observe) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            recorder.close()
            record = json.loads(recorder.path.read_text())
            self.assertEqual(record["metrics"]["work"]["count"], 800)
            self.assertEqual(record["metrics"]["work"]["events"], 1600)
            self.assertEqual(record["counters"]["batches"], 800)

    def test_elapsed_clock_units_and_invalid_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PerfRecorder(directory, "elapsed", True, interval_s=60)
            with mock.patch("performance.time.perf_counter_ns", return_value=2_500_000):
                recorder.elapsed("work", 1_000_000)
            recorder.observe("work", float("nan"))
            with self.assertRaises(ValueError):
                recorder.observe("work", 1, unit="bytes")
            recorder.close()
            record = json.loads(recorder.path.read_text())
            self.assertEqual(record["metrics"]["work"]["mean"], 1.5)
            self.assertEqual(record["metrics"]["work"]["samples"][0][2], 2_500_000)
            self.assertEqual(record["counters"]["invalid_measurements"], 1)

    def test_resource_stat_parser_handles_spaces_and_parentheses(self):
        spec = importlib.util.spec_from_file_location("resources", ROOT / "scripts/monitor_resources.py")
        resources = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resources)
        fields = ["S"] + ["0"] * 30
        fields[1], fields[11], fields[12], fields[19], fields[21] = "7", "12", "5", "900", "4"
        path = mock.Mock()
        path.read_text.return_value = "42 (a (worker) name) " + " ".join(fields)
        result = resources.read_stat(path)
        self.assertEqual(result["name"], "a (worker) name")
        self.assertEqual((result["ppid"], result["ticks"], result["start_ticks"]), (7, 17, 900))
        self.assertEqual(result["rss_bytes"], 4 * resources.PAGE_BYTES)

    def test_event_time_audit_preserves_integer_nanoseconds_and_boundaries(self):
        spec = importlib.util.spec_from_file_location("event_timing", ROOT / "tests/inspect_event_timing.py")
        timing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(timing)
        epoch = 1_789_219_083_107_543_144
        row = timing.summarize_timestamps([epoch, epoch, epoch - 1, epoch + 300_000_000],
                                          previous=epoch + 1)
        self.assertEqual(row["first_event_ns"], epoch)
        self.assertEqual(row["within_batch_negative_deltas"], 1)
        self.assertEqual(row["within_batch_duplicate_deltas"], 1)
        self.assertEqual(row["within_batch_positive_jumps"], 1)
        self.assertEqual(row["previous_batch_last_to_first_ns"], -1)
        self.assertEqual(row["boundary_negative_delta"], 1)
        self.assertEqual(timing.summarize_timestamps([])["events"], 0)


if __name__ == "__main__":
    unittest.main()
