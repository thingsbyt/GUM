"""Stress-wrapper contracts."""
from __future__ import annotations
import unittest
import numpy as np
from jepa_asteroids.lifelong_benchmark import MazeGame,NavigateGame,PixelMazeMemory
from jepa_asteroids.logic_benchmark import CausalChainMemory,LogicChainGame
from jepa_asteroids.state_memory import perceptual_state_key
from jepa_asteroids.stress import FlickerNoise,ObservationShift,RemappedActions,RepeatedLogicGame


class StressTests(unittest.TestCase):
    def test_visual_shift_preserves_contract_but_changes_pixels(self):
        base=NavigateGame();plain=base.reset(7)
        shifted=ObservationShift(NavigateGame()).reset(7)
        self.assertEqual(shifted.shape,plain.shape);self.assertEqual(shifted.dtype,np.uint8)
        self.assertFalse(np.array_equal(shifted,plain))

    def test_control_remap_and_repeated_chain_change_problem(self):
        wrapped=RemappedActions(NavigateGame(horizon=1),(2,3,1,0));wrapped.reset(8)
        self.assertTrue(wrapped.step(0)[2])
        normal=LogicChainGame();normal.reset(9);repeated=RepeatedLogicGame();repeated.reset(9)
        self.assertEqual(len(normal.chain),5);self.assertEqual(len(repeated.chain),6)

    def test_perceptual_identity_ignores_bounded_sensor_noise(self):
        frame=MazeGame().reset(17);rng=np.random.default_rng(991)
        noisy=[]
        for _ in range(2):
            noise=rng.integers(-20,21,frame.shape,dtype=np.int16)
            noisy.append(np.clip(frame.astype(np.int16)+noise,0,255).astype(np.uint8))
        self.assertEqual(perceptual_state_key(noisy[0]),perceptual_state_key(noisy[1]))

    def test_reasoning_memories_survive_flicker_and_repetition(self):
        for seed in range(8):
            maze=FlickerNoise(MazeGame());obs=maze.reset(800+seed);memory=PixelMazeMemory()
            while True:
                action=memory.act(obs);obs,_,done,info=maze.step(action);memory.observe(obs)
                if done:break
            self.assertTrue(info['success'])
        for seed in range(16):
            logic=RepeatedLogicGame();obs=logic.reset(900+seed);memory=CausalChainMemory();memory.reset(obs)
            while True:
                action=memory.act(obs);obs,_,done,info=logic.step(action);memory.observe(obs,done)
                if done:break
            self.assertTrue(info['success'])


if __name__=='__main__':unittest.main()
