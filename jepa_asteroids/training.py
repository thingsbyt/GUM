"""Self-supervised projector/predictor training. No policy or reward objective."""
from __future__ import annotations
import json
import math
import os
import time
from pathlib import Path
import torch
from .config import Config
from .data import EpisodeWindows, fingerprint, progress
from .model import WorldModel
from .objective import loss_from_embeddings
from .runtime import Guard, Stopped, replace_file
from .sigreg import PatchSIGReg

CHECKPOINT_FORMAT='ad-e2e-jepa-asteroids-method-v1'
METHOD_KEYS=('encoder_id','image_height','image_width','encoder_dim','patch_size',
 'projector_hidden','projected_dim','projector_kernel','predictor_dim','predictor_depth',
 'predictor_heads','mlp_ratio','adaln_init_scale','history','horizon','action_dim',
 'sigreg_directions','sigreg_knots','sigreg_weight','rollout_training')
TRAIN_KEYS=('batch_size','epochs','learning_rate','warmup_epochs','weight_decay','grad_clip','seed')


def method_signature(cfg:Config)->str:
    return fingerprint({k:getattr(cfg,k) for k in METHOD_KEYS})


def regularizer(cfg:Config)->PatchSIGReg:
    return PatchSIGReg(cfg.sigreg_directions,cfg.sigreg_knots,
                       cfg.sigreg_site_chunk,cfg.sigreg_direction_chunk)


def lr_at(step:int,steps_per_epoch:int,cfg:Config)->float:
    warm=cfg.warmup_epochs*steps_per_epoch;total=cfg.epochs*steps_per_epoch
    if warm and step<warm:return cfg.learning_rate*(step+1)/warm
    ratio=min(1,max(0,(step-warm)/max(1,total-warm)))
    return cfg.learning_rate*0.5*(1+math.cos(math.pi*ratio))


def atomic_torch(path:Path,value:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);replace_file(temp,path)


def checkpoint_header(cfg:Config,encoding:dict,steps:int)->dict:
    return {'format':CHECKPOINT_FORMAT,'config':cfg.to_dict(),'method_signature':method_signature(cfg),
            'encoding':encoding,'steps':int(steps),
            'objective':'projected prediction + patch-wise SIGReg; optional detached-context rollouts'}


def verify_checkpoint(value:dict,cfg:Config,encoding:dict|None=None)->None:
    if value.get('format')!=CHECKPOINT_FORMAT:
        raise ValueError('Not a checkpoint from this implementation. Old DQN/demo weights are incompatible.')
    if value.get('method_signature')!=method_signature(cfg):
        raise ValueError('Architecture/objective settings differ from the checkpoint.')
    if encoding is not None and value.get('encoding')!=encoding:
        raise ValueError('Checkpoint and feature-cache encoder identities or precisions differ.')


def load_model(cfg:Config,workspace:Path,device:torch.device):
    path=workspace/'model.pt'
    if not path.exists():raise RuntimeError('No trained model.pt. Train first; random weights are not a playable checkpoint.')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    verify_checkpoint(payload,cfg)
    if payload['steps']<1:raise RuntimeError('Checkpoint has no completed training updates.')
    model=WorldModel(cfg);model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval(),payload


def train(cfg:Config,workspace:Path,device:torch.device,guard:Guard,*,max_steps:int|None=None)->dict:
    if max_steps is not None and max_steps<1:raise ValueError('max_steps must be positive.')
    dataset=EpisodeWindows(cfg,workspace,'train')
    if len(dataset)<cfg.batch_size:
        raise RuntimeError(f'Need at least {cfg.batch_size} training windows; found {len(dataset)}.')
    batches=len(dataset)//cfg.batch_size
    encoding=dataset.manifest.get('encoding')
    if not encoding:raise RuntimeError('Feature encoder provenance is missing.')
    torch.manual_seed(cfg.seed)
    model=WorldModel(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    reg=regularizer(cfg)
    state={'epoch':0,'batch':0,'steps':0}
    training_settings={k:getattr(cfg,k) for k in TRAIN_KEYS}
    checkpoint=workspace/'training.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        verify_checkpoint(saved,cfg,encoding)
        if saved.get('training_settings')!=training_settings or saved.get('windows')!=len(dataset):
            raise RuntimeError('Training schedule/data changed. Use a new workspace for a new experiment.')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        state=saved['cursor'];torch.set_rng_state(saved['torch_rng'])
        if device.type=='cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    start_steps=state['steps'];last_metrics={}

    def save():
        base=checkpoint_header(cfg,encoding,state['steps'])
        base['model']=model.state_dict()
        atomic_torch(workspace/'model.pt',base)
        atomic_torch(checkpoint,{**base,'optimizer':optimizer.state_dict(),'cursor':dict(state),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
            'training_settings':training_settings,'windows':len(dataset)})

    try:
        model.train()
        for epoch in range(state['epoch'],cfg.epochs):
            generator=torch.Generator().manual_seed(cfg.seed+epoch)
            indices=torch.randperm(len(dataset),generator=generator)
            begin=state['batch'] if epoch==state['epoch'] else 0
            for batch_index in range(begin,batches):
                guard.check(force=True);tick=time.monotonic()
                chosen=indices[batch_index*cfg.batch_size:(batch_index+1)*cfg.batch_size].tolist()
                samples=[dataset[i] for i in chosen]
                features=torch.stack([s[0] for s in samples]).to(device)
                actions=torch.stack([s[1] for s in samples]).to(device)
                lr=lr_at(epoch*batches+batch_index,batches,cfg)
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                z=model.projector(features)
                terms=loss_from_embeddings(model.predictor,z,actions,cfg,reg,check=guard.check)
                if not torch.isfinite(terms.total):raise FloatingPointError('Non-finite loss; last valid checkpoint retained.')
                terms.total.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
                optimizer.step()
                state={'epoch':epoch,'batch':batch_index+1,'steps':state['steps']+1}
                if state['batch']==batches:state.update(epoch=epoch+1,batch=0)
                last_metrics={**terms.scalars(),'gradient_norm_before_clip':float(norm),
                    'seconds_per_update':time.monotonic()-tick,'learning_rate':lr}
                with (workspace/'training.jsonl').open('a',encoding='utf-8') as handle:
                    handle.write(json.dumps({'step':state['steps'],'epoch':epoch+1,**last_metrics})+'\n')
                progress(workspace,'train',step=state['steps'],epoch=epoch+1,total_epochs=cfg.epochs,
                    batch=batch_index+1,batches=batches,metrics=last_metrics,
                    message='Training the published prediction/SIGReg objective; no reward targets.')
                print(f"step {state['steps']} | epoch {epoch+1}/{cfg.epochs} | loss {last_metrics['total']:.5f} | {last_metrics['seconds_per_update']:.2f}s/update",flush=True)
                del terms,z,features,actions,samples
                if state['steps']%10==0:save()
                if max_steps is not None and state['steps']-start_steps>=max_steps:
                    save();progress(workspace,'training-paused',step=state['steps'],metrics=last_metrics,
                                    message='Requested update limit reached. Resume retains optimizer and data cursor.')
                    return {'steps':state['steps'],'completed_schedule':False,**last_metrics}
                if cfg.step_pause_seconds:time.sleep(cfg.step_pause_seconds)
        save();progress(workspace,'trained',step=state['steps'],metrics=last_metrics,
                        message='Configured training schedule complete. Evaluate held-out goal trials next.')
        return {'steps':state['steps'],'completed_schedule':True,**last_metrics}
    except (Stopped,KeyboardInterrupt):
        save();progress(workspace,'stopped',step=state['steps'],message='Stopped. Model and optimizer saved.')
        raise
    except Exception:
        # Do not overwrite valid checkpoints with possibly non-finite parameters.
        raise
