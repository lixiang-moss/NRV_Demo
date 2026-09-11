"""Run with PYTHONPATH=runtime python -m unittest discover -s tests -v."""

import contextlib
import io
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from examples.e2fai_realtime import (
    EVENT_DTYPE, NUM_BINS, RunningRange, montage, parse_args, voxelize,
)


class ResolutionTests(unittest.TestCase):
    def events(self):
        # Edges, colliding pixels, and invalid sensor coordinates.
        return np.array([
            (0, 0, 0, 1), (1, 1, 25, 1), (2, 2, 50, 0),
            (959, 719, 75, 1), (959, 719, 100, 1),
            (960, 0, 100, 1), (0, 720, 100, 1),
        ], dtype=EVENT_DTYPE)

    def voxel(self, events, width=640, height=480):
        return voxelize(
            events, 960, 720, torch.device("cpu"), 1000, 960, 720,
            input_width=width, input_height=height,
        )

    def test_downsample_edges_collisions_and_time_weights(self):
        actual = self.voxel(self.events()).numpy()
        expected = np.zeros((1, NUM_BINS, 480, 640), dtype=np.float32)
        expected[0, 0, 0, 0] = 1
        expected[0, 3:5, 0, 0] = 0.5
        expected[0, 7, 1, 1] = -1
        expected[0, 10:12, 479, 639] = 0.5
        expected[0, 14, 479, 639] = 1
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(float(actual.sum()), 3.0)

    def test_native_default_unchanged(self):
        events = self.events()
        legacy = voxelize(events, 960, 720, torch.device("cpu"), 1000, 960, 720)
        explicit = self.voxel(events, 960, 720)
        self.assertEqual(tuple(legacy.shape), (1, NUM_BINS, 720, 960))
        self.assertTrue(torch.equal(legacy, explicit))
        self.assertEqual(float(legacy.sum()), 3.0)

    def test_camera_mismatch_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "configured sensor"):
            voxelize(self.events(), 640, 480, torch.device("cpu"), 1000, 960, 720,
                     input_width=640, input_height=480)

    def test_two_panels_use_model_resolution(self):
        output = {"log_image": torch.zeros(1, 1, 480, 640),
                  "flow": torch.zeros(1, 2, 480, 640)}
        frame = montage(output, RunningRange(), 20, 10, 30)
        self.assertEqual(frame.shape, (508, 1280, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def parse(self, *args):
        with patch.object(sys, "argv", ["e2fai_realtime.py", "--self-check", *args]):
            return parse_args()

    def test_cli_defaults_and_640(self):
        default = self.parse()
        self.assertEqual((default.input_width, default.input_height), (960, 720))
        low = self.parse("--input-width", "640", "--input-height", "480")
        self.assertEqual((low.sensor_width, low.sensor_height), (960, 720))
        self.assertEqual((low.input_width, low.input_height), (640, 480))

    def test_cli_rejects_invalid_sizes(self):
        for args in [
            ["--input-width", "640"],
            ["--input-width", "0", "--input-height", "480"],
            ["--input-width", "1280", "--input-height", "960"],
            ["--input-width", "640", "--input-height", "400"],
            ["--input-width", "600", "--input-height", "450"],
        ]:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    self.parse(*args)
                self.assertEqual(failure.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
