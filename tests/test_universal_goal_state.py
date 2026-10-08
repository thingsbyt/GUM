"""Contracts for general set goals, temporal concepts and safe composition."""
from __future__ import annotations
import tempfile
from pathlib import Path
import unittest

from jepa_asteroids.immune_savior import ImmuneSaviorGame
from jepa_asteroids.universal_goal_state import (
    GoalProgram, UniversalGoalStateAgent, run_episode_series, run_immune_development_audit,
)


class UniversalGoalStateTests(unittest.TestCase):
    def test_goal_program_contains_eliminate_preserve_avoid_and_budget(self):
        operators = [row["operator"] for row in GoalProgram.safe_dynamic_cleanup().to_json()]
        self.assertEqual(operators, ["eliminate", "preserve", "avoid", "budget"])

    def test_agent_learns_combination_and_cures_without_damage(self):
        agent = UniversalGoalStateAgent(); result = run_episode_series(agent, ImmuneSaviorGame(501), 2400)
        self.assertTrue(result["success"]); self.assertEqual(result["healthy_damaged"], 0)
        self.assertTrue(result["recipe_learned"]); self.assertTrue(result["controls_grounded"])

    def test_memory_round_trip_reuses_safe_recipe(self):
        agent = UniversalGoalStateAgent(); run_episode_series(agent, ImmuneSaviorGame(602), 2400)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "brain.json"; agent.save(path); restored = UniversalGoalStateAgent.load(path)
        result = run_episode_series(restored, ImmuneSaviorGame(602), 700)
        self.assertTrue(result["success"]); self.assertEqual(result["healthy_damaged"], 0)

    def test_small_development_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            report = run_immune_development_audit(Path(folder) / "audit.json", worlds=2, seed=703, budget=2400)
        self.assertEqual(report["aggregate"]["universal_goal_state"]["success_rate"], 1.0)
        self.assertEqual(report["aggregate"]["revisit_after_all_worlds"]["success_rate"], 1.0)
        self.assertEqual(report["aggregate"]["random_controls"]["success_rate"], 0.0)
        self.assertEqual(report["aggregate"]["unsafe_no_preservation_gate"]["zero_damage_worlds"], 0)


if __name__ == "__main__": unittest.main()
