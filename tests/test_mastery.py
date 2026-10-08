"""Tests for outcome-only prior learning and the mastered visual controller."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_method import tiny_config
from jepa_asteroids.mastery import _softmax, learn_prior


class NoHardwareGuard:
    def check(self, force=False):
        pass


class MasteryTests(unittest.TestCase):
    def test_softmax_and_prior_search_are_valid_and_persistent(self):
        probabilities = _softmax(np.asarray([-2.0, 0.0, 1.0, 3.0, .5]))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue(np.all(probabilities > 0))
        cfg = tiny_config(episode_steps=16)
        with tempfile.TemporaryDirectory() as td:
            result = learn_prior(cfg, Path(td), NoHardwareGuard(), generations=1,
                                 population=4, episodes=2, checkpoint='fixture.pt')
            self.assertAlmostEqual(sum(result['probabilities']), 1.0)
            self.assertEqual(result['training_signal'],
                             'episode returns only; no engine state or action labels')
            self.assertTrue((Path(td) / 'mastery_policy.json').exists())


if __name__ == '__main__':
    unittest.main()
