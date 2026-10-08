"""Integrity tests for the held-out schema-free Warehouse Spill transfer."""
from __future__ import annotations

import unittest
import numpy as np

from jepa_asteroids.freeform_lab import run_object_episode
from jepa_asteroids.object_centric_agent import ObjectCentricAgent
from jepa_asteroids.object_transfer_audit import FROZEN_SHA, agent_hash
from jepa_asteroids.warehouse_spill import WarehouseSpill


class WarehouseSpillTests(unittest.TestCase):
    def test_freeform_pixels_and_anonymous_controls(self):
        game = WarehouseSpill(710001); frame = game.reset()
        self.assertEqual(frame.shape, (400, 540, 3)); self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(set(game.control) | {game.interact_action}, set(range(5)))

    def test_object_agent_is_byte_frozen(self):
        self.assertEqual(agent_hash(), FROZEN_SHA)

    def test_frozen_agent_learns_first_held_out_warehouse(self):
        result = run_object_episode(ObjectCentricAgent(), WarehouseSpill(710001), 5000, 1500)
        self.assertTrue(result["success"]); self.assertEqual(result["protected_damaged"], 0)
        self.assertTrue(result["controls_grounded"]); self.assertTrue(result["recipe_learned"])


if __name__ == "__main__": unittest.main()
