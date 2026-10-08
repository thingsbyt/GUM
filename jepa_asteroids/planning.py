"""Eqs. 13-14: rank a fixed vocabulary by terminal squared latent distance.

This module deliberately has no environment import or simulation capability.
The driving vocabulary is replaced with a fixed Asteroids button-sequence bank.
No learned/simulator reward, policy, value, survival score or scripted aim is used.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F


def build_vocabulary(count:int=256,horizon:int=8,actions:int=5,seed:int=7213)->torch.Tensor:
    if count<actions or count>actions**horizon:
        raise ValueError('Invalid vocabulary size.')
    rng=np.random.default_rng(seed)
    rows=[tuple([a]*horizon) for a in range(actions)]
    seen=set(rows)
    while len(rows)<count:
        row=rng.integers(actions,size=horizon)
        # Action persistence only generates fixed candidates, not a steering rule.
        if rng.random()<0.5:
            for t in range(1,horizon):
                if rng.random()<0.5: row[t]=row[t-1]
        key=tuple(int(a) for a in row)
        if key not in seen:
            seen.add(key);rows.append(key)
    return torch.tensor(rows,dtype=torch.long)


@dataclass
class Plan:
    index:int
    actions:torch.Tensor
    costs:torch.Tensor


def rollout(predictor,context:torch.Tensor,past_actions:torch.Tensor,
            candidates:torch.Tensor,action_dim:int,check=None)->torch.Tensor:
    """Return terminal embeddings, one per candidate. Context batch must be one."""
    if context.shape[0]!=1:
        raise ValueError('Planner accepts one observed scene at a time.')
    n,f=candidates.shape
    h=context.shape[1]
    if past_actions.shape!=(1,h-1,action_dim):
        raise ValueError('Past actions must connect the H context frames, shape [1,H-1,A].')
    current=context.expand(n,*context.shape[1:])
    a_future=F.one_hot(candidates.to(context.device),action_dim).float()
    all_actions=torch.cat((past_actions.expand(n,-1,-1),a_future),dim=1)
    for step in range(f):
        if check is not None:check()
        prediction=predictor(current,all_actions[:,step:step+h])
        current=torch.cat((current[:,1:],prediction[:,-1:]),dim=1)
    return current[:,-1]


@torch.no_grad()
def plan_to_goal(predictor,context:torch.Tensor,past_actions:torch.Tensor,goal:torch.Tensor,
                 vocabulary:torch.Tensor,action_dim:int=5,chunk:int=16,check=None)->Plan:
    if chunk<1 or len(vocabulary)<1:
        raise ValueError('Nonempty vocabulary and positive chunk size required.')
    if goal.shape!=context[:,-1:].shape:
        raise ValueError('Goal must be [1,1,H,W,D] using the SAME encoder and projector.')
    was_training=predictor.training
    predictor.eval()
    scores=[]
    try:
        for candidates in vocabulary.split(chunk):
            if check is not None:check()
            last=rollout(predictor,context,past_actions,candidates,action_dim,check=check)
            # SUM of squared errors, not cosine, normalized feature, reward or value.
            scores.append((last-goal[:,0]).square().flatten(1).sum(1))
    finally:
        predictor.train(was_training)
    costs=torch.cat(scores)
    if not torch.isfinite(costs).all():
        raise FloatingPointError('Non-finite planning costs; refusing to choose a control.')
    index=int(torch.argmin(costs))
    return Plan(index,vocabulary[index].detach().cpu(),costs.detach().cpu())
