"""Tests for the stable return-grounded pixel controller."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from test_method import tiny_config
from jepa_asteroids.stable_learning import (DuelingQNetwork, StableLearner,
    StableReplay, evaluate, random_shift, run_session)
from jepa_asteroids.mastery import (calibrate_visual_policy, evaluate_visual_policy,
    train_vision_curriculum)


class NoHardwareGuard:
    def check(self, force=False):
        pass


def stable_cfg(**changes):
    base = dict(stable_latent_dim=32, stable_batch_size=4,
        stable_updates_per_episode=1, stable_warmup_transitions=8,
        stable_replay_episodes=4, stable_n_step=2, stable_target_interval=2,
        stable_epsilon_decay_steps=8, stable_random_shift=1,
        stable_milestone_updates=1)
    base.update(changes)
    return tiny_config(**base)


class StableNetworkTests(unittest.TestCase):
    def test_trainable_eyes_and_dueling_outputs(self):
        cfg = stable_cfg()
        model = DuelingQNetwork(cfg)
        frames = torch.randint(0, 256, (4, cfg.self_frame_stack,
            cfg.self_observation_height, cfg.self_observation_width), dtype=torch.uint8)
        shifted = random_shift(frames, 1)
        self.assertEqual(shifted.shape, frames.shape)
        before = model.eyes[0].weight.detach().clone()
        loss = model(shifted).square().mean()
        loss.backward()
        torch.optim.Adam(model.parameters(), 1e-3).step()
        self.assertFalse(torch.equal(before, model.eyes[0].weight))
        self.assertEqual(model(frames).shape, (4, cfg.action_dim))

    def test_vectorized_random_shift_preserves_constant_images(self):
        frames = torch.stack([
            torch.full((4, 64, 128), value, dtype=torch.uint8)
            for value in range(6)
        ])
        shifted = random_shift(frames, 4)
        torch.testing.assert_close(shifted, frames)

    def test_n_step_replay_shapes_and_discount(self):
        cfg = stable_cfg()
        with tempfile.TemporaryDirectory() as td:
            learner = StableLearner(cfg, Path(td), torch.device('cpu'), fresh=True)
            replay = StableReplay(cfg, Path(td), learner.rng)
            count = 12
            frames = np.zeros((count + 1, cfg.self_observation_height,
                               cfg.self_observation_width), dtype=np.uint8)
            replay.add(frames, np.arange(count) % cfg.action_dim,
                       np.ones(count, dtype=np.float32), np.zeros(count, dtype=np.bool_),
                       seed=1, hits=1, updates=0)
            obs, actions, returns, nxt, one_nxt, dones, discounts = replay.sample(4)
            self.assertEqual(obs.shape, nxt.shape)
            self.assertEqual(obs.shape, one_nxt.shape)
            self.assertEqual(obs.shape[1], cfg.self_frame_stack)
            self.assertEqual(actions.shape, returns.shape)
            self.assertTrue(torch.all(discounts <= cfg.stable_discount))
            self.assertTrue(replay._episode_cache)


class StableSessionTests(unittest.TestCase):
    def test_resume_frozen_watch_and_evaluation(self):
        cfg = stable_cfg()
        guard = NoHardwareGuard()
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            first = run_session(cfg, work, torch.device('cpu'), guard, episodes=1, fresh=True)
            self.assertEqual(first['training_updates'], 1)
            second = run_session(cfg, work, torch.device('cpu'), guard, episodes=1)
            self.assertEqual(second['training_updates'], 2)
            self.assertGreater(second['results'][0]['greedy_decisions'], 0)
            brain = (work / 'stable_brain.pt').read_bytes()
            replay = (work / 'stable_replay' / 'manifest.json').read_bytes()
            watched = run_session(cfg, work, torch.device('cpu'), guard,
                                  episodes=1, learning=False)
            self.assertFalse(watched['learning'])
            self.assertEqual((work / 'stable_brain.pt').read_bytes(), brain)
            self.assertEqual((work / 'stable_replay' / 'manifest.json').read_bytes(), replay)
            report = evaluate(cfg, work, torch.device('cpu'), guard, episodes=2)
            self.assertEqual(set(report['summaries']), {'random', 'constant_fire',
                'untrained_model', 'trained_model', 'corrupted_action_mapping'})
            self.assertEqual((work / 'stable_brain.pt').read_bytes(), brain)
            with self.assertRaises(FileExistsError):
                run_session(cfg, work, torch.device('cpu'), guard, episodes=1, fresh=True)

    def test_prior_to_visual_curriculum_and_frozen_ablation_protocol(self):
        cfg = stable_cfg(episode_steps=16)
        guard = NoHardwareGuard()
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            run_session(cfg, work, torch.device('cpu'), guard, episodes=1, fresh=True)
            (work / 'mastery_policy.json').write_text(
                '{"probabilities":[0.1,0.1,0.2,0.1,0.5]}', encoding='utf-8')
            trained = train_vision_curriculum(cfg, work, torch.device('cpu'), guard,
                                              episodes=2, start_prior_weight=.8,
                                              end_prior_weight=0.0)
            self.assertEqual(trained['episodes_completed'], 2)
            self.assertEqual(trained['results'][-1]['prior_weight'], 0.0)
            calibrated = calibrate_visual_policy(cfg, work, torch.device('cpu'), guard,
                                                 episodes=2)
            self.assertEqual(calibrated['selected']['prior_weight'], 0.0)
            brain = (work / 'stable_brain.pt').read_bytes()
            replay = (work / 'stable_replay' / 'manifest.json').read_bytes()
            report = evaluate_visual_policy(cfg, work, torch.device('cpu'), guard,
                                            episodes=2, seed_offset=9)
            self.assertEqual(set(report['summaries']), {'random', 'constant_fire',
                'learned_prior', 'visual_selected', 'visual_greedy', 'occluded_visual',
                'shuffled_frame_order', 'corrupted_visual_mapping'})
            self.assertEqual(set(report['gates']), {'beats_learned_prior', 'uses_pixels',
                'control_mapping_matters', 'promoted'})
            self.assertEqual((work / 'stable_brain.pt').read_bytes(), brain)
            self.assertEqual((work / 'stable_replay' / 'manifest.json').read_bytes(), replay)


if __name__ == '__main__':
    unittest.main()
