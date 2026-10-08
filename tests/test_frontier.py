"""Contracts for the frontier autonomy audit."""
from __future__ import annotations

import unittest
import numpy as np

from jepa_asteroids.frontier import (DeceptiveLogicGame,PartialCueGame,SeededSevenGame,
                                     SwitchingCueGame,_deceptive_audit,_partial_audit,_switching_audit)
from jepa_asteroids.strategy_selector import LearnedStrategySelector


class FrontierTests(unittest.TestCase):
    def test_seeded_growth_world_varies_mapping_not_interface(self):
        a=SeededSevenGame(1);b=SeededSevenGame(2);obs=a.reset(3)
        self.assertEqual(obs.shape,(1,32,32));self.assertEqual(a.action_dim,7)
        self.assertNotEqual(a.mapping,b.mapping)

    def test_changing_world_requires_abandon_and_reacquire(self):
        env=SwitchingCueGame();env.reset(5)
        self.assertEqual(env.mapping_a,(0,1,2,3));self.assertNotEqual(env.mapping_a,env.mapping_b)
        result=_switching_audit(episodes=4,seed_base=700)
        self.assertGreater(result['phase_accuracy_b_after_change'],.50)
        self.assertGreater(result['phase_accuracy_a2_after_return'],.50)

    def test_strategy_selector_learns_centroids(self):
        rows=[(np.asarray([0.,0.]),'left'),(np.asarray([.1,0.]),'left'),
              (np.asarray([1.,1.]),'right'),(np.asarray([.9,1.]),'right')]
        model=LearnedStrategySelector.fit(rows)
        self.assertEqual(model.predict_features(np.asarray([.05,.02]))[0][0],'left')
        self.assertEqual(model.predict_features(np.asarray([.95,.98]))[0][0],'right')

    def test_partial_memory_and_deceptive_reward_adversaries(self):
        partial=PartialCueGame(rounds=2);self.assertEqual(partial.reset(4).shape,(1,32,32))
        self.assertGreater(_partial_audit(episodes=4,seed_base=800)['accuracy'],.50)
        deceptive=DeceptiveLogicGame();deceptive.reset(7)
        self.assertGreaterEqual(deceptive.step(0)[1],0.0)
        self.assertEqual(_deceptive_audit(episodes=8,seed_base=900)['causal_success_rate'],1.0)


if __name__=='__main__':unittest.main()
