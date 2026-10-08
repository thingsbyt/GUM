"""Contracts for autonomous competence selection and growth tasks."""
from __future__ import annotations
import unittest
import numpy as np
from jepa_asteroids.autonomy import AnonymousSixGame,AutonomousCompetenceLoop


class AutonomyTests(unittest.TestCase):
    def test_anonymous_world_exposes_only_pixels_actions_and_reward(self):
        env=AnonymousSixGame(horizon=2);obs=env.reset(4)
        self.assertEqual(obs.shape,(1,32,32));self.assertEqual(obs.dtype,np.uint8)
        _,reward,done,info=env.step(0);self.assertIn(reward,(1.0,-.25))
        self.assertIn('correct',info);self.assertFalse(done)

    def test_empirical_competence_requires_real_gain(self):
        random={'mean_return':1.0}
        self.assertFalse(AutonomousCompetenceLoop._gain_is_real({'mean_return':1.05},random))
        self.assertTrue(AutonomousCompetenceLoop._gain_is_real({'mean_return':1.5},random))


if __name__=='__main__':unittest.main()
