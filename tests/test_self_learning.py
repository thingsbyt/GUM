"""End-to-end tests for the no-pretraining pixel learner."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from test_method import tiny_config
from jepa_asteroids.self_learning import (PixelReplay,SelfLearner,VisionWorldModel,
                                           evaluate,frame_to_eye,run_session)


class NoHardwareGuard:
    def check(self,force=False):pass


def self_cfg(**changes):
    base=dict(self_latent_dim=32,self_ensemble=2,self_batch_size=4,
        self_updates_per_episode=1,self_warmup_transitions=8,self_replay_episodes=4,
        self_planner_candidates=8,self_planner_horizon=3,self_epsilon_decay_steps=8,
        self_milestone_updates=1)
    base.update(changes);return tiny_config(**base)


class VisionTests(unittest.TestCase):
    def test_eyes_and_world_model_train_from_pixels(self):
        cfg=self_cfg();model=VisionWorldModel(cfg)
        obs=torch.randint(0,256,(4,cfg.self_frame_stack,cfg.self_observation_height,
                                 cfg.self_observation_width),dtype=torch.uint8)
        nxt=torch.randint_like(obs,0,256);actions=torch.arange(4)%cfg.action_dim
        before=model.eyes[0].weight.detach().clone();opt=torch.optim.Adam(model.parameters(),1e-3)
        total,parts,errors=model.losses(obs,actions,torch.zeros(4),nxt,torch.zeros(4))
        total.backward();opt.step()
        self.assertFalse(torch.equal(before,model.eyes[0].weight))
        self.assertEqual(errors.shape,(4,));self.assertTrue(all(torch.isfinite(v) for v in parts.values()))
    def test_eye_input_is_pixels_not_telemetry(self):
        cfg=self_cfg();frame=np.zeros((cfg.image_height,cfg.image_width,3),dtype=np.uint8)
        eye=frame_to_eye(frame,cfg)
        self.assertEqual(eye.shape,(cfg.self_observation_height,cfg.self_observation_width))
        self.assertEqual(eye.dtype,np.uint8)


class SelfSessionTests(unittest.TestCase):
    def test_fresh_learning_resume_planning_and_frozen_watch(self):
        cfg=self_cfg();guard=NoHardwareGuard()
        with tempfile.TemporaryDirectory() as td:
            work=Path(td)
            first=run_session(cfg,work,torch.device('cpu'),guard,episodes=1,fresh=True)
            self.assertEqual(first['training_updates'],1)
            second=run_session(cfg,work,torch.device('cpu'),guard,episodes=1)
            self.assertEqual(second['training_updates'],2)
            self.assertGreater(second['results'][0]['planned_decisions'],0)
            brain=(work/'self_brain.pt').read_bytes();replay=(work/'self_replay'/'manifest.json').read_bytes()
            watched=run_session(cfg,work,torch.device('cpu'),guard,episodes=1,learning=False)
            self.assertFalse(watched['learning'])
            self.assertEqual((work/'self_brain.pt').read_bytes(),brain)
            self.assertEqual((work/'self_replay'/'manifest.json').read_bytes(),replay)
            report=evaluate(cfg,work,torch.device('cpu'),guard,episodes=2)
            self.assertEqual(set(report['summaries']),{'random','constant_fire','untrained_model',
                'trained_model','corrupted_action_model'})
            self.assertEqual((work/'self_brain.pt').read_bytes(),brain)
            with self.assertRaises(FileExistsError):run_session(cfg,work,torch.device('cpu'),guard,episodes=1,fresh=True)


if __name__=='__main__':unittest.main()
