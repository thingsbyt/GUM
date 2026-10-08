from __future__ import annotations

import unittest
import numpy as np

from jepa_asteroids.causal_discovery import CausalDiscoveryEngine


class CausalDiscoveryTests(unittest.TestCase):
    def test_common_visual_change_is_recognized_as_autonomous(self):
        engine = CausalDiscoveryEngine(); before = np.zeros((8, 8), dtype=np.uint8)
        after = before.copy(); after[3, 3] = 2
        for action in (1, 2):
            engine.note_action("s", action); engine.observe("s", action, "n", before, after, False, False)
        self.assertGreater(engine.status()["autonomous_effects"], 0)

    def test_hazard_is_avoided_during_experiment_selection(self):
        engine = CausalDiscoveryEngine(); frame = np.zeros((4, 4), dtype=np.uint8)
        for _ in range(3):
            engine.note_action("s", 1); engine.observe("s", 1, "dead", frame, frame, False, True)
        self.assertEqual(engine.experiment_action("s", [1, 2]), 2)

    def test_novel_frontier_is_replayed_and_round_trips(self):
        engine = CausalDiscoveryEngine(); frame = np.zeros((4, 4), dtype=np.uint8)
        engine.note_action("s0", 3); engine.observe("s0", 3, "s1", frame, frame, False, False)
        engine.reset_episode(); self.assertEqual(engine.replay("s0", [3])[0], 3)
        restored = CausalDiscoveryEngine(); restored.restore(engine.export()); restored.reset_episode()
        self.assertEqual(restored.replay("s0", [3])[0], 3)

    def test_successful_coordinate_trajectory_is_replayed_with_data(self):
        engine = CausalDiscoveryEngine(); frame = np.zeros((4, 4), dtype=np.uint8)
        engine.note_action("s0", 6, {"x": 17, "y": 23})
        engine.observe("s0", 6, "s1", frame, frame, True, False)
        engine.reset_episode(); action, data, reason = engine.replay("s0", [6])
        self.assertEqual((action, data, reason),
                         (6, {"x": 17, "y": 23}, "replay-successful-coordinate-procedure"))
        restored = CausalDiscoveryEngine(); restored.restore(engine.export()); restored.reset_episode()
        self.assertEqual(restored.replay("s0", [6])[1], {"x": 17, "y": 23})

    def test_frontier_that_leads_to_failure_is_abandoned(self):
        engine = CausalDiscoveryEngine(); frame = np.zeros((4, 4), dtype=np.uint8)
        engine.note_action("s0", 2); engine.observe("s0", 2, "s1", frame, frame, False, False)
        engine.reset_episode(); self.assertEqual(engine.replay("s0", [2])[0], 2)
        engine.note_action("s0", 2); engine.observe("s0", 2, "dead", frame, frame, False, True)
        self.assertGreaterEqual(engine.status()["hazardous_frontiers"], 1)


if __name__ == "__main__": unittest.main()
