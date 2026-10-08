"""Contracts for the two-guardian cooperative disease game."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from jepa_asteroids.cooperative_immune_savior import (
    CooperativeImmuneSaviorGame, CooperativeImmuneTeam, run_cooperative_audit, run_team_episode,
)


class CooperativeImmuneSaviorTests(unittest.TestCase):
    def test_public_interface_has_two_private_pixel_views(self):
        game = CooperativeImmuneSaviorGame(1001); frames = game.reset()
        self.assertEqual(len(frames), 2); self.assertEqual(frames[0].shape, (3, 126, 112))
        self.assertEqual(frames[1].shape, (3, 126, 112))

    def test_recipe_requires_complementary_agents_and_joint_interaction(self):
        game = CooperativeImmuneSaviorGame(1002, spread_interval=10_000); game.reset()
        pair = sorted(game.correct_pair)
        owners = [next(i for i in range(2) if item in game.accessible[i]) for item in pair]
        self.assertEqual(set(owners), {0, 1})
        for item, owner in zip(pair, owners):
            game.positions[owner] = game.fixed[item]
            actions = [game.interact_actions[i] if i == owner else next(iter(game.control_maps[i])) for i in range(2)]
            game.step(actions)
        game.positions = [game.fixed["synthesizer"]] * 2
        game.step(game.interact_actions)
        self.assertTrue(game.treatment)

    def test_communicating_pair_learns_and_cures_safely(self):
        result = run_team_episode(CooperativeImmuneTeam(), CooperativeImmuneSaviorGame(1003), 1500)
        self.assertTrue(result["success"]); self.assertEqual(result["healthy_damaged"], 0)
        self.assertTrue(result["team"]["joint_recipe_learned"])
        self.assertTrue(result["team"]["growing_concept_learned"])
        self.assertEqual(result["team"]["cooperation_mode"], "coordinated")
        self.assertTrue(result["team"]["cooperation_decision"]["necessary"])
        self.assertTrue(all(value > 0 for value in result["team"]["independent_probe_counts"]))

    def test_small_audit_beats_random(self):
        with tempfile.TemporaryDirectory() as folder:
            report = run_cooperative_audit(Path(folder) / "audit.json", worlds=2, seed=1101, budget=1500)
        self.assertEqual(report["aggregate"]["communicating_pair"]["successes"], 2)
        self.assertEqual(report["aggregate"]["communicating_pair"]["zero_damage_worlds"], 2)
        self.assertEqual(report["aggregate"]["same_agents_cooperation_disabled"]["successes"], 0)
        self.assertEqual(report["aggregate"]["random_pair"]["successes"], 0)


if __name__ == "__main__": unittest.main()
