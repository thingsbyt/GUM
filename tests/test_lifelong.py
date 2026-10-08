"""Lifelong skill-bank tests use synthetic pixels and make no game-mastery claim."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from jepa_asteroids.lifelong import LifelongSkillBank, TaskSpec, run_toy_continual_proof


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observation(value: int) -> np.ndarray:
    result = np.zeros((1, 16, 16), dtype=np.uint8)
    result[:, value:value + 3, value:value + 3] = 255
    return result


class LifelongBankTests(unittest.TestCase):
    def spec(self, task_id: str, seed: int = 3) -> TaskSpec:
        return TaskSpec(task_id, (1, 16, 16), 3, replay_capacity=32,
                        batch_size=4, latent_dim=32, target_interval=2, seed=seed)

    def test_new_task_cannot_modify_committed_old_skill(self):
        with tempfile.TemporaryDirectory() as td:
            bank = LifelongSkillBank(Path(td))
            old = bank.open(self.spec('old'))
            for i in range(6): old.observe(observation(i), i % 3, 1.0, observation(i), True)
            old.learn(2); bank.consolidate(old, .5)
            before = digest(old.path)
            new = bank.open(self.spec('new', 4))
            for i in range(6): new.observe(observation(i), (i + 1) % 3, 1.0, observation(i), True)
            new.learn(3); event = bank.consolidate(new, .6)
            self.assertTrue(event['other_skills_unchanged'])
            self.assertEqual(digest(old.path), before)
            self.assertEqual(new.state['parent_task'], 'old')

    def test_bad_revisit_is_rolled_back(self):
        with tempfile.TemporaryDirectory() as td:
            bank = LifelongSkillBank(Path(td)); skill = bank.open(self.spec('remember'))
            for i in range(6): skill.observe(observation(i), i % 3, 1.0, observation(i), True)
            skill.learn(2); bank.consolidate(skill, .8)
            before = digest(skill.path)
            with torch.no_grad():
                for parameter in skill.online.parameters(): parameter.add_(10)
            event = bank.consolidate(skill, .2)
            self.assertFalse(event['accepted'])
            self.assertEqual(digest(skill.path), before)
            saved = bank.open(self.spec('remember'))
            self.assertLess(max(float(p.abs().max()) for p in saved.online.parameters()), 10)

    def test_context_router_handles_known_and_novel_scenes(self):
        with tempfile.TemporaryDirectory() as td:
            bank = LifelongSkillBank(Path(td)); bank.open(self.spec('scene'))
            known = observation(2); bank.record_context('scene', known)
            self.assertEqual(bank.route(known, 3), 'scene')
            novel = np.full((1, 16, 16), 255, dtype=np.uint8)
            self.assertIsNone(bank.route(novel, 3, max_distance=.05))

    def test_a_b_a_proof_retains_and_revisits(self):
        with tempfile.TemporaryDirectory() as td:
            report = run_toy_continual_proof(Path(td))
            self.assertTrue(report['a_checkpoint_unchanged_while_b_learned'])
            self.assertEqual(report['a_after_learning_b'], report['a_after_first_encounter'])
            self.assertGreater(report['a_after_revisit'], report['a_after_first_encounter'])
            self.assertTrue(report['events'][-1]['accepted'])

    def test_failed_skill_can_be_recoverably_quarantined(self):
        with tempfile.TemporaryDirectory() as td:
            bank = LifelongSkillBank(Path(td)); skill = bank.open(self.spec('weak'))
            skill.save(); event = bank.quarantine('weak', 'below baseline')
            self.assertNotIn('weak', bank.registry['tasks'])
            self.assertTrue(Path(event['recoverable_path']).joinpath('skill.pt').exists())
            self.assertEqual(bank.registry['events'][-1]['event'], 'quarantine')


if __name__ == '__main__': unittest.main()
