"""Algorithm/software tests. Synthetic tensors here are NOT DINOv3 checkpoints."""
from __future__ import annotations
from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import torch
from torch import nn
from jepa_asteroids.config import Config
from jepa_asteroids.encoder import FrozenDINOv3,patch_tokens
from jepa_asteroids.model import WorldModel,SpatialProjector,ActionConditionedPredictor,block_causal_mask,axial_rope
from jepa_asteroids.objective import loss_from_embeddings,target_mse
from jepa_asteroids.planning import build_vocabulary,plan_to_goal,rollout
from jepa_asteroids.sigreg import PatchSIGReg,direct_statistic
from jepa_asteroids.training import method_signature,verify_checkpoint,checkpoint_header,lr_at

torch.set_num_threads(2)


def tiny_config(**changes):
    base=dict(name='TEST_ONLY_SYNTHETIC',encoder_dim=24,projector_hidden=16,projected_dim=12,
        predictor_dim=48,predictor_depth=2,predictor_heads=4,image_height=64,image_width=128,
        sigreg_directions=16,sigreg_site_chunk=3,sigreg_direction_chunk=7,batch_size=4,
        activation_checkpointing=False,projection_frame_chunk=8,candidates=10,candidate_chunk=4,
        train_episodes=2,validation_episodes=1,test_episodes=2,episode_steps=16,epochs=2,
        step_pause_seconds=0,device='cpu')
    base.update(changes);return Config(**base)


class SigregTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)
    def sample(self):
        z=torch.randn(4,3,2,2,12,dtype=torch.float64,requires_grad=True)
        a=torch.randn(12,11,dtype=torch.float64);a/=a.norm(dim=0)
        return z,a
    def test_default_directions_and_knots(self):
        r=PatchSIGReg();self.assertEqual((r.directions,r.knots),(1024,17))
    def test_value_matches_direct(self):
        z,a=self.sample();r=PatchSIGReg(11,17,3,4)
        reference=direct_statistic(z.permute(1,2,3,0,4).reshape(-1,4,12),a)
        self.assertAlmostEqual(float(r(z,a)),float(reference),places=12)
    def test_gradient_matches_direct(self):
        z,a=self.sample();v=PatchSIGReg(11,17,3,4)(z,a);g,=torch.autograd.grad(v,z)
        reference=direct_statistic(z.permute(1,2,3,0,4).reshape(-1,4,12),a)
        expected,=torch.autograd.grad(reference,z)
        self.assertLess(float((g-expected).abs().max()),1e-12)
    def test_chunk_sizes_do_not_change_value_or_gradient(self):
        z,a=self.sample();one=PatchSIGReg(11,17,1,1)(z,a);many=PatchSIGReg(11,17,500,500)(z,a)
        g,=torch.autograd.grad(one,z);h,=torch.autograd.grad(many,z)
        torch.testing.assert_close(one,many,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(g,h,atol=1e-12,rtol=1e-12)
    def test_finite_difference_gradcheck(self):
        z=torch.randn(3,1,1,1,4,dtype=torch.float64,requires_grad=True)
        a=torch.randn(4,7,dtype=torch.float64);a/=a.norm(dim=0)
        self.assertTrue(torch.autograd.gradcheck(lambda q:PatchSIGReg(7,17,2,3)(q,a),(z,),eps=1e-6,atol=1e-5))
    def test_locations_not_pooled(self):
        z=torch.randn(1,3,2,2,12,dtype=torch.float64).expand(4,-1,-1,-1,-1).clone()
        a=torch.randn(12,11,dtype=torch.float64);a/=a.norm(dim=0)
        patch=PatchSIGReg(11)(z,a)
        pooled=direct_statistic(z.reshape(1,-1,12),a)
        self.assertGreater(abs(float(patch-pooled)),.1)
    def test_batch_microaveraging_is_not_equivalent(self):
        z,a=self.sample();r=PatchSIGReg(11)
        whole=r(z,a);micro=(r(z[:2],a)+r(z[2:],a))/2
        self.assertGreater(abs(float(whole-micro)),1e-5)
    def test_collapsed_representation_has_larger_statistic(self):
        r=PatchSIGReg(128,17,2,32)
        z=torch.randn(128,1,1,1,16)
        a=torch.randn(16,128);a/=a.norm(dim=0)
        self.assertGreater(float(r(torch.zeros_like(z),a)),float(r(z,a))*10)
    def test_reject_batch_one(self):
        with self.assertRaises(ValueError):PatchSIGReg()(torch.randn(1,2,2,2,8))
    def test_reject_non_unit_directions(self):
        with self.assertRaises(ValueError):PatchSIGReg(3)(torch.randn(2,1,1,1,4),torch.ones(4,3))


class ZeroPredictor(nn.Module):
    def __init__(self):super().__init__();self.calls=[]
    def forward(self,z,a):self.calls.append((z.requires_grad,z.detach().clone(),a.detach().clone()));return z*0


class ObjectiveTests(unittest.TestCase):
    def test_multistep_equation_normalization_and_call_count(self):
        cfg=tiny_config();z=torch.arange(12.).view(1,12,1,1,1).expand(2,-1,-1,-1,-1).clone().requires_grad_()
        actions=torch.zeros(2,11,5);p=ZeroPredictor()
        loss=loss_from_embeddings(p,z,actions,cfg,lambda z,d=None:z.sum()*0)
        self.assertAlmostEqual(float(loss.teacher_forcing),46.,places=5)
        self.assertAlmostEqual(float(loss.rollout),476.,places=5)
        self.assertAlmostEqual(float(loss.prediction),65.25,places=5)
        self.assertEqual(len(p.calls),15)  # first TF reused, no duplicate first AR
    def test_rollout_contexts_detach_but_teacher_contexts_do_not(self):
        z=torch.randn(2,12,1,1,12,requires_grad=True);p=ZeroPredictor()
        loss_from_embeddings(p,z,torch.zeros(2,11,5),tiny_config(),lambda z,d=None:z.sum()*0)
        self.assertTrue(all(c[0] for c in p.calls[:8]));self.assertFalse(any(c[0] for c in p.calls[8:]))
    def test_target_mse_stop_gradient(self):
        prediction=torch.ones(2,requires_grad=True);target=torch.zeros(2,requires_grad=True)
        target_mse(prediction,target).backward()
        self.assertIsNotNone(prediction.grad);self.assertIsNone(target.grad)
    def test_sigreg_receives_all_observed_frames_and_target_gradient(self):
        z=torch.ones(2,12,1,1,12,requires_grad=True);seen=[]
        def reg(q,d=None):seen.append(q.shape);return q.square().mean()
        terms=loss_from_embeddings(ZeroPredictor(),z,torch.zeros(2,11,5),tiny_config(),reg)
        terms.total.backward();self.assertEqual(seen,[z.shape]);self.assertGreater(float(z.grad[:,-1].abs().sum()),0)
    def test_no_target_gradient_without_regularizer(self):
        z=torch.ones(2,12,1,1,12,requires_grad=True)
        terms=loss_from_embeddings(ZeroPredictor(),z,torch.zeros(2,11,5),tiny_config(),lambda q,d=None:q.sum()*0)
        terms.total.backward();self.assertEqual(float(z.grad[:,-1].abs().sum()),0)
    def test_single_step_is_equation_11(self):
        cfg=tiny_config(rollout_training=False,sigreg_weight=.09)
        z=torch.arange(5.).view(1,5,1,1,1).expand(2,-1,-1,-1,-1).clone();p=ZeroPredictor()
        terms=loss_from_embeddings(p,z,torch.zeros(2,4,5),cfg,lambda q,d=None:q.new_tensor(2.))
        self.assertAlmostEqual(float(terms.total),7.5+.18,places=5);self.assertEqual(len(p.calls),1)
    def test_bad_temporal_alignment_rejected(self):
        with self.assertRaises(ValueError):loss_from_embeddings(ZeroPredictor(),torch.zeros(2,12,1,1,12),torch.zeros(2,10,5),tiny_config(),PatchSIGReg(16))


class ArchitectureTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(7)
    def test_paper_projector_geometry(self):
        cfg=Config(activation_checkpointing=False,projection_frame_chunk=1)
        layer=SpatialProjector(cfg)
        output=layer(torch.randn(1,2,1024,16,32))
        self.assertEqual(output.shape,(1,2,4,8,256))
        convs=[x for x in layer.modules() if isinstance(x,nn.Conv2d)]
        self.assertEqual(len(convs),2);self.assertTrue(all(x.stride==(2,2) for x in convs))
    def test_projector_is_shared(self):
        model=WorldModel(tiny_config());self.assertEqual(sum(isinstance(m,SpatialProjector) for m in model.modules()),1)
        self.assertFalse(any('target' in n or 'ema' in n for n,_ in model.named_parameters()))
    def test_future_cannot_leak_into_past(self):
        cfg=tiny_config();p=ActionConditionedPredictor(cfg).eval()
        z=torch.randn(2,4,1,2,12);a=torch.randn(2,4,5)
        initial=p(z,a);z[:,3]+=100;a[:,3]+=100
        changed=p(z,a)
        torch.testing.assert_close(initial[:,:3],changed[:,:3],rtol=1e-5,atol=1e-6)
        self.assertGreater(float((initial[:,3]-changed[:,3]).abs().max()),1e-4)
    def test_spatial_block_mask(self):
        m=block_causal_mask(3,2,3,torch.device('cpu'))
        self.assertTrue(m[:6,:6].all());self.assertFalse(m[:6,6:].any());self.assertTrue(m[12:,:].all())
    def test_rope_preserves_norm(self):
        x=torch.randn(2,4,24,12);rotated=axial_rope(x,4,2,3)
        torch.testing.assert_close(x.square().sum(-1),rotated.square().sum(-1),atol=2e-5,rtol=2e-5)
    def test_activation_checkpointing_matches(self):
        off=tiny_config();on=replace(off,activation_checkpointing=True)
        a=WorldModel(off);b=WorldModel(on);b.load_state_dict(a.state_dict())
        f=torch.randn(2,4,24,4,8);actions=torch.randn(2,4,5)
        x=a(f,actions);y=b(f,actions)
        x.square().mean().backward();y.square().mean().backward()
        torch.testing.assert_close(x,y)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)
    def test_no_control_value_or_reward_heads(self):
        names=' '.join(n for n,_ in WorldModel(tiny_config()).named_parameters())
        for word in ('reward','value_head','q_head','policy','death','inverse'):self.assertNotIn(word,names)
    def test_strict_production_config_rejects_tiny_fixture(self):
        with self.assertRaises(ValueError):tiny_config().validate()
    def test_real_default_core_dimensions(self):
        c=Config().validate();self.assertEqual((c.encoder_dim,c.projected_dim,c.history,c.horizon),(1024,256,4,8))
        self.assertEqual((c.predictor_dim,c.predictor_depth,c.predictor_heads),(1024,12,16))


class EncoderContractTests(unittest.TestCase):
    def test_cls_and_registers_are_removed(self):
        x=torch.arange(1*11*8).reshape(1,11,8).float()
        y=patch_tokens(x,4,2,3,8)
        torch.testing.assert_close(y[:,:,0,0],x[:,5]);torch.testing.assert_close(y[:,:,-1,-1],x[:,-1])
    def test_bad_token_count_rejected(self):
        with self.assertRaises(ValueError):patch_tokens(torch.zeros(1,11,8),4,3,3,8)
    def test_loader_frozen_local_only_no_remote_code(self):
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__();self.weight=nn.Parameter(torch.ones(1));self.last_pixels=None
                self.config=SimpleNamespace(hidden_size=1024,patch_size=16,num_hidden_layers=24,model_type='dinov3_vit',num_register_tokens=4)
            def forward(self,pixel_values):
                self.last_pixels=pixel_values
                return SimpleNamespace(last_hidden_state=torch.zeros(len(pixel_values),5+32,1024))
        net=Backbone();factory=mock.Mock(return_value=(net,{}))
        with tempfile.TemporaryDirectory() as td:
            Path(td,'config.json').write_text('{}');Path(td,'model.safetensors').write_bytes(b'test fixture only')
            cfg=replace(Config(),encoder_path=td,image_height=64,image_width=128)
            with mock.patch.dict('sys.modules',{'transformers':SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=factory))}):
                enc=FrozenDINOv3(cfg,torch.device('cpu'));enc.train();out=enc(torch.zeros(2,64,128,3,dtype=torch.uint8))
            self.assertFalse(net.training);self.assertFalse(net.weight.requires_grad)
            self.assertFalse(out.requires_grad);self.assertEqual(out.shape,(2,1024,4,8))
            call=factory.call_args.kwargs;self.assertTrue(call['local_files_only']);self.assertFalse(call['trust_remote_code']);self.assertTrue(call['use_safetensors'])
            self.assertAlmostEqual(float(net.last_pixels[0,0,0,0]),-.485/.229,places=5)
    def test_incomplete_load_refuses_random_weights(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td,'config.json').write_text('{}');Path(td,'model.safetensors').write_bytes(b'fixture')
            cfg=replace(Config(),encoder_path=td)
            factory=mock.Mock(return_value=(nn.Identity(),{'missing_keys':['layer.weight']}))
            with mock.patch.dict('sys.modules',{'transformers':SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=factory))}):
                with self.assertRaisesRegex(RuntimeError,'refusing random'):FrozenDINOv3(cfg,torch.device('cpu'))


class AdditivePredictor(nn.Module):
    def __init__(self):super().__init__();self.calls=[]
    def forward(self,z,a):
        self.calls.append(a.detach().clone())
        delta=(a*torch.arange(a.shape[-1],device=a.device)).sum(-1)
        return z+delta[:,:,None,None,None]


class PlanningTests(unittest.TestCase):
    def setup_scene(self):
        return torch.zeros(1,4,1,1,1),torch.nn.functional.one_hot(torch.tensor([[0,1,2]]),5).float(),torch.full((1,1,1,1,1),32.)
    def test_vocabulary_fixed_unique_and_contains_constant_controls(self):
        v=build_vocabulary();torch.testing.assert_close(v,build_vocabulary())
        self.assertEqual(len(set(map(tuple,v.tolist()))),256)
        for i in range(5):self.assertTrue((v[i]==i).all())
    def test_goal_distance_selects_correct_candidate(self):
        z,a,g=self.setup_scene();p=AdditivePredictor();v=build_vocabulary(10)
        result=plan_to_goal(p,z,a,g,v,chunk=3)
        self.assertEqual(result.index,4);self.assertEqual(float(result.costs[4]),0.)
    def test_chunking_does_not_change_scores(self):
        z,a,g=self.setup_scene();v=build_vocabulary(10)
        x=plan_to_goal(AdditivePredictor(),z,a,g,v,chunk=1);y=plan_to_goal(AdditivePredictor(),z,a,g,v,chunk=10)
        torch.testing.assert_close(x.costs,y.costs)
    def test_past_and_candidate_actions_align(self):
        z,a,g=self.setup_scene();p=AdditivePredictor();v=torch.tensor([[4,3,2,1,0,4,3,2]])
        rollout(p,z,a,v,5)
        self.assertEqual(p.calls[0].argmax(-1).tolist(),[[0,1,2,4]])
        self.assertEqual(p.calls[1].argmax(-1).tolist(),[[1,2,4,3]])
        self.assertEqual(p.calls[-1].argmax(-1).tolist(),[[0,4,3,2]])
    def test_physics_is_not_consulted(self):
        z,a,g=self.setup_scene()
        with mock.patch('jepa_asteroids.engine.Asteroids.step',side_effect=AssertionError('oracle physics called')):
            plan_to_goal(AdditivePredictor(),z,a,g,build_vocabulary(10))
        from jepa_asteroids import planning
        source=inspect.getsource(planning)
        self.assertNotIn('from .engine',source);self.assertNotIn('import engine',source)
    def test_nonfinite_costs_refused(self):
        z,a,g=self.setup_scene();g.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):plan_to_goal(AdditivePredictor(),z,a,g,build_vocabulary(10))
    def test_cancellation_between_batches(self):
        z,a,g=self.setup_scene();check=mock.Mock(side_effect=RuntimeError('stop'))
        with self.assertRaisesRegex(RuntimeError,'stop'):plan_to_goal(AdditivePredictor(),z,a,g,build_vocabulary(10),check=check)
    def test_training_mode_restored(self):
        z,a,g=self.setup_scene();p=AdditivePredictor().train()
        plan_to_goal(p,z,a,g,build_vocabulary(10));self.assertTrue(p.training)


class CheckpointTests(unittest.TestCase):
    def test_old_prototype_format_rejected(self):
        with self.assertRaises(ValueError):verify_checkpoint({'model':{}},Config())
    def test_signature_checks_method_not_memory_settings(self):
        c=Config();self.assertEqual(method_signature(c),method_signature(replace(c,max_gpu_gib=9)))
        self.assertNotEqual(method_signature(c),method_signature(replace(c,predictor_dim=256,predictor_heads=4)))
    def test_encoder_identity_mismatch_rejected(self):
        c=Config();v=checkpoint_header(c,{'encoder':'one'},1)
        with self.assertRaises(ValueError):verify_checkpoint(v,c,{'encoder':'another'})
    def test_warmup_and_cosine_endpoints(self):
        c=Config();self.assertAlmostEqual(lr_at(9,10,c),c.learning_rate)
        self.assertAlmostEqual(lr_at(300,10,c),0)

if __name__=='__main__':unittest.main()
