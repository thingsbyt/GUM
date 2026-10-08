"""Small, fully local pixel-to-action learner with trainable visual "eyes".

No pretrained model or language model is used. The visual encoder, latent
dynamics ensemble, reward/termination predictors and decoder all start from
random weights and learn from the agent's own Asteroids experience.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Protocol

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .config import Config
from .data import domain, game_config, progress
from .engine import Asteroids
from .evaluation import publish
from .runtime import Guard, Stopped, atomic_json, replace_file
from .training import atomic_torch


FORMAT='wailah-self-learning-v1'
REPLAY_FORMAT='wailah-pixel-replay-v1'
SELF_KEYS=('self_observation_height','self_observation_width','self_frame_stack',
    'self_latent_dim','self_ensemble','self_batch_size','self_learning_rate',
    'self_updates_per_episode','self_warmup_transitions','self_replay_episodes',
    'self_recent_fraction','self_planner_candidates','self_planner_horizon',
    'self_discount','self_curiosity_weight','self_epsilon_start','self_epsilon_end',
    'self_epsilon_decay_steps','self_milestone_updates','action_dim')


def settings(cfg:Config)->dict:
    return {key:getattr(cfg,key) for key in SELF_KEYS}


def signature(cfg:Config)->str:
    return hashlib.sha256(json.dumps(settings(cfg),sort_keys=True).encode()).hexdigest()


def frame_to_eye(frame:np.ndarray,cfg:Config)->np.ndarray:
    image=Image.fromarray(frame).convert('L').resize(
        (cfg.self_observation_width,cfg.self_observation_height),Image.Resampling.BILINEAR)
    return np.asarray(image,dtype=np.uint8)


class EnvironmentAdapter(Protocol):
    action_count:int
    def reset(self,seed:int)->np.ndarray:...
    def step(self,action:int)->tuple[np.ndarray,float,bool,dict]:...


class AsteroidsAdapter:
    """Only RGB, legal actions and permitted outcomes cross this boundary."""
    action_count=5
    def __init__(self,cfg:Config):self.cfg=cfg;self.game=None
    def reset(self,seed:int)->np.ndarray:
        self.game=Asteroids(game_config(self.cfg),seed);return self.game.observe()
    def step(self,action:int)->tuple[np.ndarray,float,bool,dict]:
        frame,reward,ended,timeout,info=self.game.step(action)
        return frame,reward,bool(ended or timeout),{
            'hits':int(info['hits']),'total_hits':int(info['total_hits']),
            'terminated':bool(ended),'truncated':bool(timeout),
            'simulated_seconds':float(info['seconds'])}


def _atomic_npz(path:Path,**arrays)->None:
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as handle:np.savez_compressed(handle,**arrays)
    replace_file(temp,path)


class PixelReplay:
    """Bounded episode replay retaining both recent and reservoir-sampled history."""
    def __init__(self,cfg:Config,workspace:Path,rng:np.random.Generator):
        self.cfg=cfg;self.rng=rng;self.root=workspace/'self_replay';self.raw=self.root/'episodes'
        self.path=self.root/'manifest.json'
        expected={'format':REPLAY_FORMAT,'settings':settings(cfg),'domain':domain(cfg)}
        if self.path.exists():
            self.manifest=json.loads(self.path.read_text(encoding='utf-8'))
            for key,value in expected.items():
                if self.manifest.get(key)!=value:
                    raise RuntimeError(f'Self-learning replay {key} differs; use a new brain.')
        else:
            self.manifest={**expected,'seen_episodes':0,'archive_candidates':0,
                           'recent':[],'archive':[]};self._save()
    def _save(self):atomic_json(self.path,self.manifest)
    @property
    def entries(self):return self.manifest['recent']+self.manifest['archive']
    def transition_count(self):return sum(e['transitions'] for e in self.entries)
    def _drop(self,entry):
        path=self.raw/entry['file']
        if path.exists():path.unlink()
    def _rebalance(self):
        total=self.cfg.self_replay_episodes
        recent_cap=max(1,min(total,int(math.ceil(total*self.cfg.self_recent_fraction))))
        archive_cap=total-recent_cap
        while len(self.manifest['recent'])>recent_cap:
            candidate=self.manifest['recent'].pop(0)
            seen=int(self.manifest['archive_candidates'])+1;self.manifest['archive_candidates']=seen
            archive=self.manifest['archive']
            if archive_cap==0:self._drop(candidate)
            elif len(archive)<archive_cap:archive.append(candidate)
            else:
                position=int(self.rng.integers(seen))
                if position<archive_cap:self._drop(archive[position]);archive[position]=candidate
                else:self._drop(candidate)
    def add(self,frames:np.ndarray,actions:np.ndarray,rewards:np.ndarray,dones:np.ndarray,
            *,seed:int,hits:int,policy:str,updates:int)->dict:
        expected=(len(actions)+1,self.cfg.self_observation_height,self.cfg.self_observation_width)
        if frames.shape!=expected or frames.dtype!=np.uint8:
            raise ValueError(f'Eye frames must be uint8 {expected}.')
        if actions.shape!=rewards.shape or actions.shape!=dones.shape:
            raise ValueError('Transition arrays are misaligned.')
        number=int(self.manifest['seen_episodes']);identity=f'ep_{number:08d}'
        name=identity+'.npz';_atomic_npz(self.raw/name,frames=frames,
            actions=actions.astype(np.uint8),rewards=rewards.astype(np.float32),
            dones=dones.astype(np.bool_))
        entry={'id':identity,'file':name,'transitions':len(actions),'seed':int(seed),
               'hits':int(hits),'return':float(rewards.sum()),'policy':policy,
               'model_updates':int(updates)}
        self.manifest['seen_episodes']=number+1;self.manifest['recent'].append(entry)
        self._rebalance();self._save();return entry
    def sample(self,batch:int):
        stack=self.cfg.self_frame_stack
        eligible=[e for e in self.entries if e['transitions']>=stack]
        if not eligible:raise RuntimeError('Replay has no episode long enough for a stacked transition.')
        recent_ids={e['id'] for e in self.manifest['recent']}
        recent=[e for e in eligible if e['id'] in recent_ids]
        archive=[e for e in eligible if e['id'] not in recent_ids]
        obs=[];nxt=[];actions=[];rewards=[];dones=[]
        for _ in range(batch):
            pool=recent if recent and (not archive or self.rng.random()<self.cfg.self_recent_fraction) else archive
            entry=pool[int(self.rng.integers(len(pool)))]
            with np.load(self.raw/entry['file'],allow_pickle=False) as data:
                index=int(self.rng.integers(stack-1,len(data['actions'])))
                obs.append(data['frames'][index-stack+1:index+1].copy())
                nxt.append(data['frames'][index-stack+2:index+2].copy())
                actions.append(int(data['actions'][index]));rewards.append(float(data['rewards'][index]))
                dones.append(bool(data['dones'][index]))
        return (torch.from_numpy(np.stack(obs)),torch.tensor(actions),torch.tensor(rewards),
                torch.from_numpy(np.stack(nxt)),torch.tensor(dones,dtype=torch.float32))


class VisionWorldModel(nn.Module):
    def __init__(self,cfg:Config):
        super().__init__();self.cfg=cfg;channels=cfg.self_frame_stack;latent=cfg.self_latent_dim
        self.eyes=nn.Sequential(
            nn.Conv2d(channels,32,5,2,2),nn.SiLU(),
            nn.Conv2d(32,64,5,2,2),nn.SiLU(),
            nn.Conv2d(64,96,3,2,1),nn.SiLU(),
            nn.Conv2d(96,128,3,2,1),nn.SiLU())
        h,w=cfg.self_observation_height//16,cfg.self_observation_width//16
        self.to_latent=nn.Sequential(nn.Flatten(),nn.Linear(128*h*w,latent),nn.LayerNorm(latent))
        self.from_latent=nn.Linear(latent,128*h*w)
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(128,96,4,2,1),nn.SiLU(),
            nn.ConvTranspose2d(96,64,4,2,1),nn.SiLU(),
            nn.ConvTranspose2d(64,32,4,2,1),nn.SiLU(),
            nn.ConvTranspose2d(32,1,4,2,1))
        joint=latent+cfg.action_dim
        self.dynamics=nn.ModuleList(nn.Sequential(nn.Linear(joint,256),nn.SiLU(),
            nn.Linear(256,256),nn.SiLU(),nn.Linear(256,latent)) for _ in range(cfg.self_ensemble))
        self.reward_heads=nn.ModuleList(nn.Sequential(nn.Linear(joint,128),nn.SiLU(),
            nn.Linear(128,1)) for _ in range(cfg.self_ensemble))
        self.done_heads=nn.ModuleList(nn.Sequential(nn.Linear(joint,128),nn.SiLU(),
            nn.Linear(128,1)) for _ in range(cfg.self_ensemble))
        self.apply(self._init)
    @staticmethod
    def _init(module):
        if isinstance(module,(nn.Conv2d,nn.ConvTranspose2d,nn.Linear)):
            nn.init.orthogonal_(module.weight,gain=math.sqrt(2))
            if module.bias is not None:nn.init.zeros_(module.bias)
    def encode(self,frames):return self.to_latent(self.eyes(frames.float()/255.0))
    def decode(self,z):
        h,w=self.cfg.self_observation_height//16,self.cfg.self_observation_width//16
        return self.decoder(self.from_latent(z).reshape(len(z),128,h,w))
    def predict_member(self,z,actions,member):
        one=F.one_hot(actions.long(),self.cfg.action_dim).float();joint=torch.cat((z,one),-1)
        next_z=z+0.1*self.dynamics[member](joint)
        return next_z,self.reward_heads[member](joint).squeeze(-1),self.done_heads[member](joint).squeeze(-1)
    def losses(self,obs,actions,rewards,nxt,dones):
        z=self.encode(obs);z_next=self.encode(nxt);target=z_next.detach()
        predictions=[];reward_predictions=[];done_logits=[]
        for member in range(self.cfg.self_ensemble):
            p,r,d=self.predict_member(z,actions,member)
            predictions.append(p);reward_predictions.append(r);done_logits.append(d)
        pred=torch.stack(predictions);reward_pred=torch.stack(reward_predictions);done_logit=torch.stack(done_logits)
        per_sample=(pred-target.unsqueeze(0)).square().mean((0,2))
        prediction=per_sample.mean()
        reward_loss=F.smooth_l1_loss(reward_pred,rewards.clamp(-5,5).unsqueeze(0).expand_as(reward_pred))
        done_loss=F.binary_cross_entropy_with_logits(done_logit,dones.unsqueeze(0).expand_as(done_logit))
        newest=obs[:,-1:].float()/255.0;next_newest=nxt[:,-1:].float()/255.0
        reconstruction=(F.mse_loss(torch.sigmoid(self.decode(z)),newest)+
                        F.mse_loss(torch.sigmoid(self.decode(z_next)),next_newest))/2
        variance=F.relu(1-z.std(0,unbiased=False)).mean()+F.relu(1-z_next.std(0,unbiased=False)).mean()
        total=prediction+0.5*reconstruction+0.5*reward_loss+0.1*done_loss+0.05*variance
        return total,{'prediction':prediction,'reconstruction':reconstruction,
            'reward':reward_loss,'termination':done_loss,'variance':variance},per_sample


@dataclass
class Plan:
    action:int
    score:float
    predicted_frame:np.ndarray


class SelfLearner:
    def __init__(self,cfg:Config,workspace:Path,device:torch.device,*,fresh=False):
        self.cfg=cfg;self.workspace=workspace;self.device=device;self.path=workspace/'self_brain.pt'
        if fresh and self.path.exists():raise FileExistsError('Fresh start refuses to overwrite a saved brain.')
        torch.manual_seed(cfg.seed);self.rng=np.random.default_rng(cfg.seed+417_901)
        self.model=VisionWorldModel(cfg).to(device)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=cfg.self_learning_rate,weight_decay=1e-5)
        self.state={'episodes':0,'environment_steps':0,'updates':0,'created_unix':time.time(),
                    'action_learning_progress':[0.0]*cfg.action_dim}
        if self.path.exists() and not fresh:self._load()
    def _load(self):
        value=torch.load(self.path,map_location='cpu',weights_only=True)
        if value.get('format')!=FORMAT or value.get('signature')!=signature(self.cfg):
            raise RuntimeError('Saved self-learning brain uses different architecture/settings.')
        self.model.load_state_dict(value['model']);self.optimizer.load_state_dict(value['optimizer'])
        self.state=value['state'];self.rng.bit_generator.state=value['numpy_rng']
        torch.set_rng_state(value['torch_rng'])
        if self.device.type=='cuda' and value.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(value['cuda_rng'])
    def epsilon(self):
        ratio=min(1.0,self.state['environment_steps']/self.cfg.self_epsilon_decay_steps)
        return self.cfg.self_epsilon_start+ratio*(self.cfg.self_epsilon_end-self.cfg.self_epsilon_start)
    def save(self,*,milestone=False):
        value={'format':FORMAT,'signature':signature(self.cfg),'settings':settings(self.cfg),
               'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
               'state':dict(self.state),'numpy_rng':self.rng.bit_generator.state,
               'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None}
        atomic_torch(self.path,value)
        atomic_json(self.workspace/'self_brain.json',{'format':FORMAT,'signature':signature(self.cfg),
                    **self.state,'epsilon':self.epsilon(),'parameter_count':sum(p.numel() for p in self.model.parameters())})
        if milestone:atomic_torch(self.workspace/'self_milestones'/f"update_{self.state['updates']:08d}.pt",value)
    def _candidates(self):
        n=self.cfg.self_planner_candidates;h=self.cfg.self_planner_horizon;a=self.cfg.action_dim
        rows=[np.full(h,i,dtype=np.int64) for i in range(a)]
        while len(rows)<n:
            row=self.rng.integers(a,size=h)
            for t in range(1,h):
                if self.rng.random()<.6:row[t]=row[t-1]
            rows.append(row)
        return torch.tensor(np.stack(rows),device=self.device)
    @torch.no_grad()
    def plan(self,stack:np.ndarray,guard:Guard,action_permutation:torch.Tensor|None=None)->Plan:
        self.model.eval();obs=torch.from_numpy(stack[None]).to(self.device);z=self.model.encode(obs)
        candidates=self._candidates();n=len(candidates);scores=[];terminals=[]
        for member in range(self.cfg.self_ensemble):
            current=z.expand(n,-1);alive=torch.ones(n,device=self.device);score=torch.zeros(n,device=self.device)
            for t in range(self.cfg.self_planner_horizon):
                guard.check();conditioned=candidates[:,t]
                if action_permutation is not None:conditioned=action_permutation[conditioned]
                current,reward,done=self.model.predict_member(current,conditioned,member)
                score=score+(self.cfg.self_discount**t)*alive*reward
                alive=alive*(1-torch.sigmoid(done))
            scores.append(score);terminals.append(current)
        value=torch.stack(scores).mean(0)
        progress_values=torch.tensor(self.state['action_learning_progress'],device=self.device)
        bonus=progress_values[candidates].sum(1)*self.cfg.self_curiosity_weight
        value=value+bonus;index=int(torch.argmax(value));terminal=torch.stack(terminals).mean(0)[index:index+1]
        predicted=(torch.sigmoid(self.model.decode(terminal))[0,0]*255).byte().cpu().numpy()
        return Plan(int(candidates[index,0]),float(value[index]),predicted)
    def train(self,replay:PixelReplay,guard:Guard,updates:int):
        if replay.transition_count()<self.cfg.self_warmup_transitions:return []
        records=[];self.model.train()
        for _ in range(updates):
            guard.check(force=True);tick=time.monotonic()
            obs,actions,rewards,nxt,dones=replay.sample(self.cfg.self_batch_size)
            obs=obs.to(self.device);actions=actions.to(self.device);rewards=rewards.to(self.device)
            nxt=nxt.to(self.device);dones=dones.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            total,parts,before=self.model.losses(obs,actions,rewards,nxt,dones)
            if not torch.isfinite(total):raise FloatingPointError('Non-finite self-learning loss.')
            total.backward();norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),10,error_if_nonfinite=True)
            self.optimizer.step()
            with torch.no_grad():_,_,after=self.model.losses(obs,actions,rewards,nxt,dones)
            improvement=(before-after).clamp_min(0)
            for action in range(self.cfg.action_dim):
                mask=actions==action
                if mask.any():
                    measured=float(improvement[mask].mean())
                    old=self.state['action_learning_progress'][action]
                    self.state['action_learning_progress'][action]=.9*old+.1*measured
            self.state['updates']+=1
            record={'update':self.state['updates'],'total':float(total.detach()),
                    **{k:float(v.detach()) for k,v in parts.items()},
                    'gradient_norm':float(norm),'seconds':time.monotonic()-tick,
                    'action_learning_progress':list(self.state['action_learning_progress'])}
            with (self.workspace/'self_training.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(record)+'\n')
            records.append(record)
        return records


def run_episode(cfg:Config,workspace:Path,learner:SelfLearner,replay:PixelReplay,
                guard:Guard,*,seed:int,learning:bool,replay_delay:float=0.0)->dict:
    env=AsteroidsAdapter(cfg);frame=env.reset(seed);eye=frame_to_eye(frame,cfg)
    stack=deque([eye.copy() for _ in range(cfg.self_frame_stack)],maxlen=cfg.self_frame_stack)
    eyes=[eye];actions=[];rewards=[];dones=[];planned=0;previous=int(learner.rng.integers(cfg.action_dim))
    hits=0;started=time.monotonic();learner.model.eval()
    for decision in range(cfg.episode_steps):
        can_plan=(learner.state['updates']>0 and replay.transition_count()>=cfg.self_warmup_transitions)
        exploring=not can_plan or learner.rng.random()<learner.epsilon()
        if exploring:
            action=previous if learner.rng.random()<.5 else int(learner.rng.integers(cfg.action_dim));predicted=eye
        else:
            plan=learner.plan(np.stack(stack),guard);action=plan.action;predicted=plan.predicted_frame;planned+=1
        frame,reward,done,info=env.step(action);eye=frame_to_eye(frame,cfg);stack.append(eye)
        eyes.append(eye);actions.append(action);rewards.append(reward);dones.append(done);previous=action;hits=info['total_hits']
        publish(workspace,frame,predicted)
        if learning:learner.state['environment_steps']+=1
        progress(workspace,'self-learning' if learning else 'self-watch',episode=learner.state['episodes']+1,
                 decision=decision+1,score=hits,lifetime_experience=learner.state['environment_steps'],
                 training_updates=learner.state['updates'],epsilon=learner.epsilon(),
                 activity='exploring' if exploring else 'planning with learned outcomes',
                 simulated_seconds=info['simulated_seconds'],wall_seconds=time.monotonic()-started,
                 message='Learning vision and dynamics directly from local pixels; no pretrained model or LLM.')
        if replay_delay:time.sleep(replay_delay)
        if done:break
    result={'seed':seed,'steps':len(actions),'hits':hits,'return':float(sum(rewards)),
            'planned_decisions':planned,'learning':learning,'terminated':bool(info['terminated']),
            'truncated':bool(info['truncated']),'epsilon':learner.epsilon()}
    if learning:
        replay.add(np.stack(eyes),np.asarray(actions),np.asarray(rewards),np.asarray(dones),
                   seed=seed,hits=hits,policy='epsilon exploration + learned world-model planning',
                   updates=learner.state['updates'])
        learner.state['episodes']+=1;completed=learner.train(replay,guard,cfg.self_updates_per_episode)
        result['updates_completed']=len(completed)
        milestone=learner.state['updates']==0 or (completed and learner.state['updates']%cfg.self_milestone_updates<len(completed))
        learner.save(milestone=bool(milestone))
    with (workspace/'self_episodes.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(result)+'\n')
    return result


def run_session(cfg:Config,workspace:Path,device:torch.device,guard:Guard,*,episodes:int,
                fresh:bool=False,learning:bool=True,replay_delay:float=0.0)->dict:
    if episodes<1:raise ValueError('episodes must be positive.')
    if fresh and ((workspace/'self_brain.pt').exists() or (workspace/'self_replay').exists()):
        raise FileExistsError('Fresh start never erases another learner; choose an empty workspace.')
    learner=SelfLearner(cfg,workspace,device,fresh=fresh);replay=PixelReplay(cfg,workspace,learner.rng)
    if not learning and not learner.path.exists():raise RuntimeError('Watch requires a saved self-learning brain.')
    if fresh:learner.save(milestone=True)
    results=[]
    try:
        for _ in range(episodes):
            seed=cfg.seed+4_000_000_000+learner.state['episodes']
            results.append(run_episode(cfg,workspace,learner,replay,guard,seed=seed,
                                       learning=learning,replay_delay=replay_delay))
    except (Stopped,KeyboardInterrupt):
        if learning:learner.save()
        raise
    summary={'learning':learning,'episodes':len(results),'brain_episodes':learner.state['episodes'],
             'environment_steps':learner.state['environment_steps'],'training_updates':learner.state['updates'],
             'results':results}
    atomic_json(workspace/'self_session.json',summary)
    progress(workspace,'paused' if learning else 'watch-complete',metrics=summary,
             message='Self-learning session saved.' if learning else 'Frozen watch finished without training or replay writes.')
    return summary


def _evaluate_episode(cfg:Config,learner:SelfLearner,guard:Guard,*,seed:int,policy:str)->dict:
    env=AsteroidsAdapter(cfg);frame=env.reset(seed);eye=frame_to_eye(frame,cfg)
    stack=deque([eye.copy() for _ in range(cfg.self_frame_stack)],maxlen=cfg.self_frame_stack)
    rng=np.random.default_rng(seed+771_001);trace=[];latencies=[];hits=0;total=0.0
    permutation=torch.tensor([2,4,0,1,3],device=learner.device)
    for _ in range(cfg.episode_steps):
        tick=time.monotonic()
        if policy=='random':action=int(rng.integers(cfg.action_dim))
        elif policy=='constant_fire':action=4
        else:
            plan=learner.plan(np.stack(stack),guard,
                action_permutation=permutation if policy=='corrupted_action_model' else None)
            action=plan.action
        latencies.append(time.monotonic()-tick);trace.append(action)
        frame,reward,done,info=env.step(action);stack.append(frame_to_eye(frame,cfg))
        hits=info['total_hits'];total+=reward
        if done:break
    return {'seed':seed,'policy':policy,'steps':len(trace),'hits':hits,'return':float(total),
            'terminated':bool(info['terminated']),'truncated':bool(info['truncated']),
            'mean_decision_seconds':float(np.mean(latencies)),'max_decision_seconds':float(np.max(latencies)),
            'actions':trace}


def evaluate(cfg:Config,workspace:Path,device:torch.device,guard:Guard,*,episodes:int=12,
             seed_offset:int=0)->dict:
    """Matched frozen comparison. Evaluation experience is never stored or trained on."""
    if episodes<1:raise ValueError('episodes must be positive.')
    trained=SelfLearner(cfg,workspace,device)
    if not trained.path.exists():raise RuntimeError('A saved self-learning brain is required.')
    # A separately initialized model supplies the initialization control.
    untrained=SelfLearner(cfg,workspace/'evaluation_untrained_fixture',device,fresh=True)
    seeds=[cfg.seed+4_500_000_000+seed_offset+i for i in range(episodes)];results=[]
    policies=('random','constant_fire','untrained_model','trained_model','corrupted_action_model')
    for policy in policies:
        agent=untrained if policy=='untrained_model' else trained
        effective='trained_model' if policy=='trained_model' else policy
        for seed in seeds:
            guard.check(force=True)
            results.append(_evaluate_episode(cfg,agent,guard,seed=seed,policy=effective))
            progress(workspace,'self-evaluation',policy=policy,trial=len(results),
                     total_trials=len(policies)*episodes,message='Frozen matched-seed evaluation; no replay or updates.')
    summaries={}
    for policy in policies:
        name='trained_model' if policy=='trained_model' else policy
        rows=[r for r in results if r['policy']==name]
        summaries[policy]={'episodes':len(rows),'mean_hits':float(np.mean([r['hits'] for r in rows])),
            'mean_return':float(np.mean([r['return'] for r in rows])),
            'termination_rate':float(np.mean([r['terminated'] for r in rows])),
            'mean_steps':float(np.mean([r['steps'] for r in rows])),
            'mean_decision_seconds':float(np.mean([r['mean_decision_seconds'] for r in rows]))}
    report={'protocol':'preliminary matched frozen episodes; evaluation data excluded from replay/training',
            'checkpoint_updates':trained.state['updates'],'checkpoint_experience':trained.state['environment_steps'],
            'seed_offset':seed_offset,'seeds':seeds,'summaries':summaries,'results':results}
    atomic_json(workspace/'self_evaluation.json',report)
    progress(workspace,'self-evaluated',metrics=summaries,
             message='Frozen matched evaluation complete. Results may be negative or inconclusive.')
    return report
