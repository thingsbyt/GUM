import unittest

import numpy as np

from jepa_asteroids.predictive_search import (
    PredictiveSearchPlanner, feature_key, predictive_features)


def scene(x: int, y: int = 12) -> np.ndarray:
    grid = np.zeros((32, 32), dtype=np.uint8)
    grid[y:y + 4, x:x + 4] = 3
    return grid


class PredictiveSearchTests(unittest.TestCase):
    def test_features_represent_spatial_state(self):
        left = predictive_features(scene(4))
        right = predictive_features(scene(20))
        self.assertEqual(left.shape, right.shape)
        self.assertNotEqual(feature_key(left), feature_key(right))

    def test_learns_action_effect_and_searches_toward_progress(self):
        planner = PredictiveSearchPlanner()
        left, middle, right = scene(4), scene(12), scene(20)
        for _ in range(6):
            planner.observe(left, 1, middle, False, False)
            planner.observe(middle, 2, left, False, False)
        planner.observe(middle, 1, right, True, False)
        planner.observe(middle, 1, right, True, False)
        # Exercise the learned world model rather than the one-state reward cache.
        planner.q.clear()
        action, reason = planner.recommend(left, [1, 2])
        self.assertEqual(action, 1)
        self.assertEqual(reason, "predictive-search-progress")
        self.assertGreater(len(planner.last_plan["sequence"]), 1)

    def test_memory_round_trip_preserves_predictions(self):
        planner = PredictiveSearchPlanner()
        for _ in range(8):
            planner.observe(scene(4), 3, scene(12), False, False)
            planner.observe(scene(12), 3, scene(20), True, False)
        memory = planner.export()
        restored = PredictiveSearchPlanner(); restored.restore(memory)
        self.assertEqual(restored.observations, planner.observations)
        self.assertEqual(restored.progress_examples, planner.progress_examples)
        self.assertEqual(restored.recommend(scene(4), [2, 3])[0], 3)


if __name__ == "__main__":
    unittest.main()
