"""Development tests for schema-free object segmentation and agency."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from jepa_asteroids.freeform_lab import FreeformLab, run_object_episode
from jepa_asteroids.object_centric_agent import ObjectCentricAgent, segment_components


class ObjectCentricAgentTests(unittest.TestCase):
    def test_components_are_position_invariant(self):
        game = FreeformLab(610001); first = segment_components(game.reset())
        game.position = (game.position[0] + 9, game.position[1]); second = segment_components(game.render())
        shared = set(row["signature"] for row in first["components"]) & set(row["signature"] for row in second["components"])
        self.assertGreaterEqual(len(shared), 7)

    def test_agent_discovers_effector_recipe_and_safe_cleanup(self):
        agent = ObjectCentricAgent(); result = run_object_episode(agent, FreeformLab(610001), 5000)
        self.assertTrue(result["success"]); self.assertEqual(result["protected_damaged"], 0)
        self.assertTrue(result["controls_grounded"]); self.assertTrue(result["recipe_learned"])

    def test_persistent_object_mind_round_trip(self):
        agent = ObjectCentricAgent(); run_object_episode(agent, FreeformLab(610002), 5000)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mind.json"; agent.save(path); restored = ObjectCentricAgent.load(path)
        result = run_object_episode(restored, FreeformLab(610002), 900)
        self.assertTrue(result["success"]); self.assertEqual(result["protected_damaged"], 0)


if __name__ == "__main__": unittest.main()
