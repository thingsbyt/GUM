"""Contracts for the held-out adaptive Data Recovery Console evaluation."""
from __future__ import annotations

import unittest
import numpy as np

from jepa_asteroids.adaptive_transfer_audit import ADAPTER_SHA, CORE_SHA, hashes, run_episode
from jepa_asteroids.adaptive_visual_agent import AdaptiveUniversalAgent
from jepa_asteroids.data_recovery_console import DataRecoveryConsole


class DataRecoveryConsoleTests(unittest.TestCase):
    def test_raw_dashboard_interface_hides_viewport_parameters(self):
        game = DataRecoveryConsole(510001); frame = game.reset()
        self.assertEqual(frame.shape, (440, 620, 3)); self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(game.action_dim, 5)

    def test_adaptive_agent_and_core_are_byte_frozen(self):
        self.assertEqual(hashes(), {"adapter": ADAPTER_SHA, "core": CORE_SHA})

    def test_frozen_adaptive_agent_learns_first_held_out_console(self):
        agent = AdaptiveUniversalAgent(); result = run_episode(agent, DataRecoveryConsole(510001), 4000)
        self.assertTrue(result["success"]); self.assertEqual(result["protected_damaged"], 0)
        self.assertTrue(result["recipe_learned"]); self.assertTrue(result["controls_grounded"])
        self.assertEqual(result["adapter_geometry"], [DataRecoveryConsole(510001).scale,
                                                      DataRecoveryConsole(510001).viewport_top,
                                                      DataRecoveryConsole(510001).viewport_left])


if __name__ == "__main__": unittest.main()
