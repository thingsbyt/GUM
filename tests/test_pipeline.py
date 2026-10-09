"""Pipeline tests use explicitly synthetic features, never advertised as DINO runs."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
import numpy as np
import torch
from torch import nn
from test_method import tiny_config
from jepa_asteroids.data import collect,encode_dataset,EpisodeWindows,load_manifest,game_config,validate_raw
from jepa_asteroids.engine import Asteroids
from jepa_asteroids.training import train,load_model
from jepa_asteroids.evaluation import evaluate
from jepa_asteroids.runtime import Stopped,Guard
from jepa_asteroids.dashboard import Controller,make_handler

class NoHardwareGuard:
    """Only test processes use this explicit fixture; production always uses Guard."""
    def check(self,force=False):pass

class SyntheticEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.identity={'model_id':'TEST_ONLY_SYNTHETIC_FEATURES_NOT_DINOV3','height':cfg.image_height,'width':cfg.image_width}
        self.matrix=torch.linspace(-1,1,3*cfg.encoder_dim).reshape(3,cfg.encoder_dim)
    def forward(self,frames):
        x=frames.permute(0,3,1,2).float()/255
        x=torch.nn.functional.adaptive_avg_pool2d(x,self.cfg.grid).permute(0,2,3,1)
        return (x@self.matrix).permute(0,3,1,2).contiguous()

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.cfg=tiny_config()
        cls.guard=NoHardwareGuard()
        collect(cls.cfg,cls.root,cls.guard)
        encode_dataset(cls.cfg,cls.root,SyntheticEncoder(cls.cfg),cls.guard)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_raw_episode_contains_only_images_and_actions(self):
        manifest=load_manifest(self.root,self.cfg)
        for e in manifest['episodes']:
            with np.load(self.root/'raw'/e['raw'],allow_pickle=False) as raw:
                self.assertEqual(set(raw.files),{'frames','actions'});validate_raw(raw,e,self.cfg)
    def test_splits_are_seed_disjoint(self):
        m=load_manifest(self.root,self.cfg);groups={s:{e['seed'] for e in m['episodes'] if e['split']==s} for s in ['train','validation','test']}
        self.assertFalse(groups['train']&groups['test']);self.assertFalse(groups['validation']&groups['test'])
    def test_windows_never_cross_episode_boundary(self):
        d=EpisodeWindows(self.cfg,self.root,'train')
        for name,start in d.windows:self.assertLessEqual(start+self.cfg.sequence_frames,d.entries[name]['frames'])
        f,a=d[0];self.assertEqual(f.shape,(12,24,4,8));self.assertEqual(a.shape,(11,5))
    def test_cache_records_synthetic_provenance(self):
        m=load_manifest(self.root,self.cfg)
        self.assertIn('TEST_ONLY_SYNTHETIC',m['encoding']['encoder']['model_id'])
    def test_recollect_is_idempotent(self):
        before=load_manifest(self.root,self.cfg)
        after=collect(self.cfg,self.root,self.guard)
        self.assertEqual(len(before['episodes']),len(after['episodes']))
    def test_engine_is_deterministic_rgb(self):
        a=Asteroids(game_config(self.cfg),99);b=Asteroids(game_config(self.cfg),99)
        for action in [0,1,3,4,2,4]:
            x,*_=a.step(action);y,*_=b.step(action)
            self.assertTrue(np.array_equal(x,y));self.assertEqual(x.dtype,np.uint8)
    def test_hud_reward_does_not_enter_observation(self):
        env=Asteroids(game_config(self.cfg),99);before=env.observe()
        env.total_reward+=1000;env.hits+=999;env.frames+=5000
        self.assertTrue(np.array_equal(before,env.observe()))
    def test_train_resume_and_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as td:
            work=Path(td)/'run';shutil.copytree(self.root,work)
            first=train(self.cfg,work,torch.device('cpu'),self.guard,max_steps=2)
            self.assertEqual(first['steps'],2)
            second=train(self.cfg,work,torch.device('cpu'),self.guard,max_steps=1)
            self.assertEqual(second['steps'],3)
            before=(work/'model.pt').read_bytes()
            result=evaluate(self.cfg,work,torch.device('cpu'),self.guard,trials=2,replay_delay=0)
            self.assertEqual(result['summary']['trials'],2)
            self.assertEqual((work/'model.pt').read_bytes(),before)
            self.assertTrue((work/'current.png').exists());self.assertTrue((work/'goal.png').exists())
            with tempfile.TemporaryDirectory() as td2:
                uninterrupted=Path(td2)/'run';shutil.copytree(self.root,uninterrupted)
                train(self.cfg,uninterrupted,torch.device('cpu'),self.guard,max_steps=3)
                a,_=load_model(self.cfg,work,torch.device('cpu'));b,_=load_model(self.cfg,uninterrupted,torch.device('cpu'))
                for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)
    def test_stop_file_is_honored(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);(path/'STOP').touch()
            with self.assertRaises(Stopped):Guard(self.cfg,path,torch.device('cpu')).check()
    def test_cache_rejects_changed_encoder_identity(self):
        e=SyntheticEncoder(self.cfg);e.identity={'different':True}
        with self.assertRaises(RuntimeError):encode_dataset(self.cfg,self.root,e,self.guard)

class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name)
        cls.controller=Controller(tiny_config(),cls.path/'config.json',cls.path)
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(cls.controller))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.server.server_address[1]}'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join();cls.temp.cleanup()
    def request(self,path,*,method='GET',token=True,origin=None,payload=None):
        headers={}
        if token:headers['X-Lab-Token']=self.controller.token
        if origin:headers['Origin']=origin
        data=json.dumps(payload).encode() if payload is not None else None
        req=urllib.request.Request(self.base+path,headers=headers,data=data,method=method)
        return urllib.request.urlopen(req,timeout=3)
    def test_missing_token_forbidden(self):
        with self.assertRaises(urllib.error.HTTPError) as result:self.request('/api/state',token=False)
        self.assertEqual(result.exception.code,403)
    def test_foreign_origin_forbidden(self):
        with self.assertRaises(urllib.error.HTTPError) as result:self.request('/api/state',origin='https://foreign.example')
        self.assertEqual(result.exception.code,403)
    def test_state_json_and_ui(self):
        with self.request('/api/state') as r:self.assertIn('progress',json.load(r))
        with self.request('/') as r:
            html=r.read().decode();self.assertIn(self.controller.token,html);self.assertNotIn('__LAB_TOKEN__',html)
            self.assertIn("Agent's visual input",html);self.assertNotIn('Predicted outcome',html)
            self.assertIn('Watch learned policy live',html)
    def test_state_exposes_compact_mastery_evaluation(self):
        report={'summaries':{'visual_mastery':{'episodes':64,'mean_hits':7.3,
            'mean_return':7.1,'termination_rate':.09,'mean_steps':123}}}
        (self.path/'mastery_evaluation.json').write_text(json.dumps(report),encoding='utf-8')
        with self.request('/api/state') as r:state=json.load(r)
        self.assertEqual(state['evaluation']['visual_mastery'],{
            'mean_hits':7.3,'mean_return':7.1,'termination_rate':.09})
    def test_stop_post_creates_file(self):
        with self.request('/api/action',method='POST',payload={'action':'stop'}) as r:self.assertTrue(json.load(r)['ok'])
        self.assertTrue((self.path/'STOP').exists())
    def test_unknown_action_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as result:self.request('/api/action',method='POST',payload={'action':'rm -rf'})
        self.assertEqual(result.exception.code,400)
    def test_worker_preserves_immediate_stop_request(self):
        with tempfile.TemporaryDirectory() as td:
            work=Path(td);controller=Controller(tiny_config(),work/'config.json',work)
            with mock.patch('jepa_asteroids.dashboard.subprocess.Popen') as start:
                start.return_value.poll.return_value=None
                controller.start('collect');controller.stop()
                command=start.call_args.args[0]
                self.assertIn('--keep-stop',command)
                self.assertTrue((work/'STOP').exists())
            if controller.log:controller.log.close()
    def test_path_traversal_not_exposed(self):
        with self.assertRaises(urllib.error.HTTPError) as result:self.request('/../config.json')
        self.assertEqual(result.exception.code,404)

if __name__=='__main__':unittest.main()
