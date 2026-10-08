"""Unit checks for the external benchmark's information boundary."""
from __future__ import annotations

import unittest

import numpy as np

from jepa_asteroids.arc_agi3_bridge import _eye


class ArcAgi3BridgeTests(unittest.TestCase):
    def test_arc_layers_become_only_pixels_at_wailah_shape(self):
        first = np.zeros((64, 64), dtype=np.uint8); first[8:16, 8:16] = 3
        second = np.zeros((64, 64), dtype=np.uint8); second[12:20, 12:20] = 9
        eye = _eye([first.tolist(), second.tolist()], (1, 32, 32))
        self.assertEqual(eye.shape, (1, 32, 32))
        self.assertEqual(eye.dtype, np.uint8)
        self.assertGreater(int(eye.max()), 0)

    def test_category_boundaries_are_not_smoothed(self):
        frame = np.zeros((64, 64), dtype=np.uint8); frame[:, 32:] = 15
        eye = _eye([frame.tolist()], (1, 32, 32))
        self.assertEqual(set(np.unique(eye).tolist()), {0, 255})


if __name__ == "__main__":
    unittest.main()
