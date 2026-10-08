"""Contracts for autonomous hypothesis switching and procedure growth."""
from __future__ import annotations

from collections import Counter
import unittest

from jepa_asteroids.hypothesis_engine import CausalHypothesisEngine


class CausalHypothesisEngineTests(unittest.TestCase):
    def test_stagnation_abandons_spatial_mode_and_tests_alternatives(self):
        engine = CausalHypothesisEngine(); actions = [1, 2, 3]; seen = []
        for step in range(24):
            mode, proposed = engine.recommend("same", actions, step, True, {1, 2}, Counter())
            action = proposed if proposed is not None else 1
            seen.append(mode)
            engine.observe("same", action, "same", 0.0, False, False)
        self.assertTrue(set(seen) & {"contextual-interaction", "temporal-persistence", "graph-frontier"})
        self.assertTrue(any(mode != "spatial-contact" for mode in seen[9:]))
        self.assertGreater(engine.status()["experiments"], 0)
        self.assertGreater(engine.status()["abandoned_hypotheses"], 0)

    def test_progress_becomes_guarded_replayable_procedure(self):
        engine = CausalHypothesisEngine()
        engine.observe("s0", 1, "s1", .2, False, False)
        engine.observe("s1", 2, "s2", .2, True, False)
        self.assertEqual(engine.status()["learned_procedures"], 1)
        engine.reset_episode()
        mode, action = engine.recommend("s0", [1, 2], 0, False, set(), Counter())
        self.assertEqual((mode, action), ("procedure-replay", 1))
        engine.observe("s0", 1, "s1", .2, False, False)
        mode, action = engine.recommend("s1", [1, 2], 1, False, set(), Counter())
        self.assertEqual((mode, action), ("procedure-replay", 2))

    def test_procedure_aborts_on_changed_world_and_memory_round_trips(self):
        engine = CausalHypothesisEngine()
        engine.observe("s0", 1, "s1", .2, True, False)
        saved = engine.export(); restored = CausalHypothesisEngine(); restored.restore(saved)
        restored.reset_episode()
        mode, action = restored.recommend("s0", [1], 0, False, set(), Counter())
        self.assertEqual((mode, action), ("procedure-replay", 1))
        restored.observe("s0", 1, "changed", .1, False, False)
        self.assertEqual(restored.status()["procedure_aborts"], 1)


if __name__ == "__main__": unittest.main()
