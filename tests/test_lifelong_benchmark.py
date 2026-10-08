"""Environment-contract tests for the real continual pixel-control suite."""
from __future__ import annotations

import unittest
import numpy as np

from jepa_asteroids.lifelong_benchmark import (AvoidGame, CatchGame, NavigateGame, SignalGame, MazeGame,
                                               evaluate_pixel_maze_memory, evaluate_random, tasks)


class LifelongBenchmarkTests(unittest.TestCase):
    def test_games_expose_pixels_actions_rewards_and_endings(self):
        for game_type in (CatchGame, AvoidGame, NavigateGame, SignalGame, MazeGame):
            game = game_type(horizon=3); observation = game.reset(17)
            self.assertEqual(observation.shape, (1, 32, 32))
            self.assertEqual(observation.dtype, np.uint8)
            for _ in range(3):
                observation, reward, done, info = game.step(0)
            self.assertTrue(done); self.assertIsInstance(reward, float)
            self.assertIsInstance(info, dict)

    def test_random_baselines_are_deterministic(self):
        for task in tasks().values():
            first = evaluate_random(task, episodes=3, seed_base=90)
            second = evaluate_random(task, episodes=3, seed_base=90)
            self.assertEqual(first, second)

    def test_pixel_memory_escapes_unseen_mazes(self):
        result = evaluate_pixel_maze_memory(episodes=8, seed_base=7_123_000)
        self.assertEqual(result['maze_success_rate'], 1.0)
        self.assertLessEqual(result['worst_success_steps'], 256)


if __name__ == '__main__': unittest.main()
