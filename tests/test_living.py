"""Contract tests for the task-free living-system layer."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from jepa_asteroids.living import OptionLibrary, RehearsalScheduler, RouterNet, SharedWorldModel
from jepa_asteroids.living_benchmark import HybridCatchAvoid


class LivingSystemTests(unittest.TestCase):
    def test_hybrid_game_hides_mode_but_encodes_it_in_pixels(self):
        env = HybridCatchAvoid(horizon=2)
        catch = avoid = None
        for seed in range(100):
            observation = env.reset(seed)
            if env.mode == 'catch' and catch is None: catch = observation.copy()
            if env.mode == 'avoid' and avoid is None: avoid = observation.copy()
            if catch is not None and avoid is not None: break
        self.assertEqual(catch.shape, (1, 32, 32))
        self.assertFalse(np.array_equal(catch, avoid))
        result = env.step(0)
        self.assertEqual(len(result), 4)

    def test_router_and_world_model_accept_pixels(self):
        pixels = torch.randint(0, 256, (5, 1, 32, 32), dtype=torch.uint8)
        router = RouterNet(1, 3)
        self.assertEqual(tuple(router(pixels).shape), (5, 3))
        world = SharedWorldModel(1, 4, latent=16)
        actions = torch.tensor([0, 1, 2, 3, 0])
        rewards = torch.zeros(5); dones = torch.zeros(5)
        loss, *_ = world.loss(pixels, actions, rewards, pixels, dones)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(world.transition_error(pixels, actions, rewards, pixels, dones).shape), (5,))

    def test_option_mining_finds_repeated_rewarded_sequence(self):
        replay = {'actions': np.tile(np.array([1, 2, 1, 2]), 8),
                  'rewards': np.tile(np.array([0., 0., 0., 1.]), 8)}
        expert = SimpleNamespace(task_id='skill', replay=replay)
        found = OptionLibrary().discover(expert, length=4)
        self.assertTrue(found)
        self.assertGreaterEqual(found[0]['count'], 3)

    def test_rehearsal_prioritizes_forgotten_skill(self):
        scheduler = RehearsalScheduler()
        scheduler.update('stable', best=1.0, current=1.0, uncertainty=.01, step=10)
        scheduler.update('forgotten', best=1.0, current=.2, uncertainty=.01, step=10)
        self.assertEqual(scheduler.choose(11), ['forgotten'])


if __name__ == '__main__': unittest.main()
