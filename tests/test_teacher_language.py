"""Contracts for the visual teacher-language bridge."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jepa_asteroids.teacher_language import (
    TaughtConcepts,
    TaughtInstructionWorld,
    TaughtLanguageAgent,
    _demonstrate,
    _evaluate,
    _teach_extension,
)


class TeacherLanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.concepts = TaughtConcepts.discover(
            lambda: TaughtInstructionWorld(), observations=32, seed_base=139_000_000)

    def make_agent(self): return TaughtLanguageAgent(self.concepts)

    def test_written_commands_are_visual_and_reward_is_terminal(self):
        env = TaughtInstructionWorld(); frame = env.reset(140_000_000)
        self.assertEqual(frame.shape, (3, 150, 360)); self.assertEqual(len(env.clauses), 2)
        _, reward, done, _ = env.step(env.target_actions[0])
        self.assertEqual(reward, 0.0); self.assertFalse(done)
        _, reward, done, info = env.step(env.target_actions[1])
        self.assertEqual(reward, 1.0); self.assertTrue(done); self.assertTrue(info['success'])

    def test_demonstrations_ground_words_and_pixels_execute_withheld_commands(self):
        agent = self.make_agent()
        result = _demonstrate(agent, lambda: TaughtInstructionWorld(), 12, 140_100_000)
        self.assertEqual(result['initial_hypotheses'], 1728)
        self.assertEqual(result['remaining_hypotheses'], 1)
        held = _evaluate(agent, lambda: TaughtInstructionWorld(held_out=True), 32, 140_200_000)
        self.assertEqual(held['success_rate'], 1.0)

    def test_new_relation_and_alias_compose_without_forgetting(self):
        agent = self.make_agent(); _demonstrate(agent, lambda: TaughtInstructionWorld(), 12, 140_300_000)
        _teach_extension(agent, lambda: TaughtInstructionWorld(relation=4), 'near', 'relation', 4, 140_400_000)
        _teach_extension(agent, lambda: TaughtInstructionWorld(color_alias='crimson'), 'crimson', 'color', 4, 140_500_000)
        combined = _evaluate(agent, lambda: TaughtInstructionWorld(relation=4, color_alias='crimson'), 32, 140_600_000)
        retained = _evaluate(agent, lambda: TaughtInstructionWorld(held_out=True), 32, 140_700_000)
        self.assertEqual(combined['success_rate'], 1.0); self.assertEqual(retained['success_rate'], 1.0)

    def test_teacher_memory_survives_reload(self):
        agent = self.make_agent(); _demonstrate(agent, lambda: TaughtInstructionWorld(), 12, 140_800_000)
        _teach_extension(agent, lambda: TaughtInstructionWorld(relation=4), 'near', 'relation', 4, 140_900_000)
        _teach_extension(agent, lambda: TaughtInstructionWorld(color_alias='crimson'), 'crimson', 'color', 4, 141_000_000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.json'; agent.save(path); loaded = TaughtLanguageAgent.load(path)
            result = _evaluate(loaded, lambda: TaughtInstructionWorld(relation=4, color_alias='crimson'), 16, 141_100_000)
            self.assertEqual(result['success_rate'], 1.0)


if __name__ == '__main__': unittest.main()
