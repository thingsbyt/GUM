"""Tests for sparse-reward causal-chain reasoning."""
from __future__ import annotations
import unittest
import numpy as np
from jepa_asteroids.logic_benchmark import CausalChainMemory,LogicChainGame,_evaluate


class LogicBenchmarkTests(unittest.TestCase):
    def test_intermediate_events_have_no_reward(self):
        env=LogicChainGame(); env.reset(4)
        for action in env.chain[:-1]:
            _,reward,done,_=env.step(action)
            self.assertEqual(reward,0.0); self.assertFalse(done)
        _,reward,done,info=env.step(env.chain[-1])
        self.assertEqual(reward,1.0); self.assertTrue(done); self.assertTrue(info['success'])

    def test_wrong_event_resets_visual_progress(self):
        env=LogicChainGame(); root=env.reset(9); env.step(env.chain[0])
        wrong=next(x for x in range(5) if x!=env.chain[1]); obs,_,_,info=env.step(wrong)
        self.assertEqual(info['progress'],0); self.assertTrue(np.array_equal(obs,root))

    def test_causal_memory_solves_unseen_chains(self):
        result=_evaluate('causal_memory',episodes=32,seed_base=12_000_000)
        self.assertEqual(result['success_rate'],1.0)
        self.assertLessEqual(result['worst_steps_on_success'],64)


if __name__=='__main__': unittest.main()
