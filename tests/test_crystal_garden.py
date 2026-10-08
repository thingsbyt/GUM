"""Integrity contracts for the held-out Crystal Garden transfer world."""
from __future__ import annotations

import unittest
import numpy as np

from jepa_asteroids.crystal_garden import CrystalGardenGame
from jepa_asteroids.frozen_transfer_audit import FROZEN_SHA256, agent_hash, run_frozen_episode
from jepa_asteroids.universal_goal_state import UniversalGoalStateAgent


class CrystalGardenTests(unittest.TestCase):
    def test_public_interface_is_pixels_and_anonymous_controls(self):
        game = CrystalGardenGame(420001); frame = game.reset()
        self.assertEqual(frame.shape, (3, 126, 112)); self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(game.action_dim, 5); self.assertEqual(set(game.control) | {game.interact_action}, set(range(5)))

    def test_v14_is_byte_frozen(self):
        self.assertEqual(agent_hash(), FROZEN_SHA256)

    def test_frozen_agent_can_learn_one_held_out_world(self):
        result = run_frozen_episode(UniversalGoalStateAgent(), CrystalGardenGame(420001), 3000)
        self.assertTrue(result["success"]); self.assertEqual(result["protected_damaged"], 0)
        self.assertTrue(result["recipe_learned"]); self.assertTrue(result["controls_grounded"])


if __name__ == "__main__": unittest.main()
