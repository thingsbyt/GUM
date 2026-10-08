"""Continual-loop tests use synthetic features and make no DINO/gameplay claim."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch import nn

from test_method import tiny_config
from jepa_asteroids.autonomous import encoding_identity, run_session
from jepa_asteroids.continual import ReplayBuffer


class NoHardwareGuard:
    def check(self, force=False): pass


class SyntheticEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg=cfg
        self.identity={'model_id':'TEST_ONLY_SYNTHETIC_FEATURES_NOT_DINOV3',
                       'height':cfg.image_height,'width':cfg.image_width}
        self.matrix=torch.linspace(-1,1,3*cfg.encoder_dim).reshape(3,cfg.encoder_dim)
    def forward(self, frames):
        x=frames.permute(0,3,1,2).float()/255
        x=torch.nn.functional.adaptive_avg_pool2d(x,self.cfg.grid).permute(0,2,3,1)
        return (x@self.matrix).permute(0,3,1,2).contiguous()


class ReplayTests(unittest.TestCase):
    def test_bounded_recent_plus_reservoir_and_goal_progress(self):
        cfg=tiny_config(online_replay_episodes=4,online_recent_fraction=.5)
        encoder=SyntheticEncoder(cfg);encoding=encoding_identity(cfg,encoder)
        with tempfile.TemporaryDirectory() as td:
            replay=ReplayBuffer(cfg,Path(td),encoding,np.random.default_rng(9))
            for seed in range(9):
                frames=np.zeros((13,cfg.image_height,cfg.image_width,3),dtype=np.uint8)
                actions=np.arange(12,dtype=np.int64)%cfg.action_dim
                feedback=np.zeros((12,4),dtype=np.float32);feedback[-1,1]=seed%2
                features=np.full((13,cfg.encoder_dim,*cfg.grid),seed,dtype=cfg.feature_dtype)
                replay.add_episode(frames,actions,feedback,features,seed=seed,
                    policy='test',model_updates=seed)
            self.assertEqual(len(replay.entries),4)
            self.assertEqual(len(replay.manifest['recent']),2)
            self.assertEqual(len(replay.manifest['archive']),2)
            self.assertLess(min(int(e['id'].split('_')[1]) for e in replay.entries),7)
            batch,actions,ids=replay.sample_batch(cfg.batch_size)
            self.assertEqual(batch.shape,(cfg.batch_size,cfg.sequence_frames,cfg.encoder_dim,*cfg.grid))
            self.assertEqual(actions.shape,(cfg.batch_size,cfg.sequence_frames-1,cfg.action_dim))
            replay.record_learning_progress(ids,.25)
            self.assertTrue(any(g.learning_progress>0 for g in replay.goals()))


class SessionTests(unittest.TestCase):
    def test_fresh_resume_and_frozen_watch(self):
        cfg=tiny_config(online_min_windows=1,online_updates_per_episode=1,
                        online_milestone_updates=1,online_replay_episodes=4,
                        online_epsilon_decay_steps=8)
        encoder=SyntheticEncoder(cfg);guard=NoHardwareGuard()
        with tempfile.TemporaryDirectory() as td:
            work=Path(td)
            first=run_session(cfg,work,torch.device('cpu'),guard,encoder,
                              episodes=1,fresh=True,learning=True)
            self.assertEqual(first['training_updates'],1)
            second=run_session(cfg,work,torch.device('cpu'),guard,encoder,
                               episodes=1,learning=True)
            self.assertEqual(second['brain_episodes'],2)
            self.assertEqual(second['training_updates'],2)
            self.assertGreater(second['results'][0]['planned_decisions'],0)
            brain=(work/'brain.pt').read_bytes();manifest=(work/'continual'/'manifest.json').read_bytes()
            watched=run_session(cfg,work,torch.device('cpu'),guard,encoder,
                                episodes=1,learning=False)
            self.assertFalse(watched['learning'])
            self.assertEqual((work/'brain.pt').read_bytes(),brain)
            self.assertEqual((work/'continual'/'manifest.json').read_bytes(),manifest)
            with self.assertRaises(FileExistsError):
                run_session(cfg,work,torch.device('cpu'),guard,encoder,
                            episodes=1,fresh=True,learning=True)
            state=json.loads((work/'brain.json').read_text(encoding='utf-8'))
            self.assertEqual(state['updates'],2)


if __name__=='__main__':unittest.main()
