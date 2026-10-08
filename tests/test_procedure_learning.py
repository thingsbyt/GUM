"""Contracts for named visual procedure learning."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jepa_asteroids.procedure_learning import (
    ProcedureMemory,
    ProcedureStrategy,
    RescueWorld,
    _evaluate,
    demonstration,
)


class ProcedureLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demos = [demonstration(160_000_000 + i) for i in range(3)]
        cls.memory = ProcedureMemory(); cls.explanation = cls.memory.learn('rescue', cls.demos)

    def test_demonstrations_are_successful_and_reward_is_terminal(self):
        for demo in self.demos:
            self.assertEqual(demo['rewards'][-1], 1.0)
            self.assertTrue(all(value == 0.0 for value in demo['rewards'][:-1]))

    def test_rescue_event_order_is_inferred_from_visual_changes(self):
        steps = self.explanation['learned_steps']
        self.assertEqual(len(steps), 4)
        self.assertEqual([row['interaction'] for row in steps], ['enter', 'use', 'use', 'enter'])
        self.assertEqual(self.explanation['source'], 'ordered visual changes in successful demonstrations')

    def test_named_procedure_plans_across_unseen_layouts(self):
        result = _evaluate(lambda: ProcedureStrategy(self.memory), 64, 160_100_000)
        self.assertEqual(result['success_rate'], 1.0)
        self.assertLessEqual(result['max_steps'], RescueWorld.horizon)

    def test_procedure_memory_survives_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rescue.json'; self.memory.save(path)
            loaded = ProcedureMemory.load(path)
            result = _evaluate(lambda: ProcedureStrategy(loaded), 32, 160_200_000)
            self.assertEqual(result['success_rate'], 1.0)
            self.assertEqual(loaded.name, 'rescue')


if __name__ == '__main__': unittest.main()
