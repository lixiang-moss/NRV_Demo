import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ros_ws/src/nrv_demo/scripts'))
from noise_filter import NoiseFilter


class FilterTests(unittest.TestCase):
    def apply(self, f, points, **changes):
        config = dict(background=True, window_ms=5., refractory=False, interval_ms=1.)
        config.update(changes)
        x, y, ts = map(np.array, zip(*points))
        return f.apply(x.astype(int), y.astype(int), ts, (8, 8), config).tolist()

    def test_isolation_and_cross_batch_neighbours(self):
        f = NoiseFilter()
        self.assertEqual(self.apply(f, [(2, 2, 1.), (2, 2, 1.001)]), [False, False])
        self.assertEqual(self.apply(f, [(3, 2, 1.002), (7, 7, 1.003), (3, 3, 1.020)]), [True, False, False])

    def test_refractory_uses_last_retained_event(self):
        self.assertEqual(self.apply(NoiseFilter(), [(1, 1, 1.), (1, 1, 1.0005), (1, 1, 1.0015)],
                                    background=False, refractory=True), [True, False, True])

    def test_reset_after_timestamp_restart(self):
        f = NoiseFilter()
        self.apply(f, [(2, 2, 10.)])
        self.assertEqual(self.apply(f, [(3, 2, 1.)]), [False])

    def test_disabled_preserves_all_events_and_input(self):
        self.assertEqual(self.apply(NoiseFilter(), [(1, 1, 1.), (1, 1, 1.)], background=False), [True, True])

    def test_border_has_no_wrapping_neighbours(self):
        self.assertEqual(self.apply(NoiseFilter(), [(7, 7, 1.), (0, 0, 1.001), (1, 0, 1.002)]), [False, False, True])

if __name__ == '__main__':
    unittest.main()
