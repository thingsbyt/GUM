"""Learned task-to-strategy ranking from neutral interaction signatures."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from .runtime import atomic_json
from .state_memory import perceptual_state_key


def probe_features(factory,seed:int,steps:int=64)->np.ndarray:
    """Summarize pixels and outcomes from policy-neutral random interaction."""
    env=factory();obs=env.reset(seed);rng=np.random.default_rng(seed+4471)
    rewards=[];diffs=[];keys=[];values=[];done_count=0
    for _ in range(steps):
        values.append(np.asarray(obs,dtype=np.float32));keys.append(perceptual_state_key(obs))
        nxt,reward,done,_=env.step(int(rng.integers(env.action_dim)))
        diffs.append(float(np.mean(np.abs(np.asarray(nxt,dtype=np.float32)-np.asarray(obs,dtype=np.float32)))/255.0))
        rewards.append(float(reward));obs=nxt
        if done:
            done_count+=1;obs=env.reset(int(rng.integers(2**31)))
    sample=np.concatenate([x.reshape(-1) for x in values[:min(8,len(values))]])/255.0
    reward=np.asarray(rewards,dtype=np.float64)
    shape=np.asarray(values[0].shape,dtype=np.float64)
    return np.asarray([env.action_dim/10.0,getattr(env,'horizon',steps)/512.0,
        values[0].ndim/4.0,shape[-2]/256.0,shape[-1]/512.0,
        float(sample.mean()),float(sample.std()),float(np.mean(diffs)),
        float(reward.mean()),float(reward.std()),float(np.mean(reward>0)),
        float(np.mean(reward!=0)),len(set(keys))/max(1,len(keys)),done_count/max(1,steps)],dtype=np.float64)


class LearnedStrategySelector:
    """Nearest-centroid selector trained on successful empirical strategy trials."""
    def __init__(self,centroids:dict[str,list[float]],scale:list[float]):
        self.centroids={k:np.asarray(v,dtype=np.float64) for k,v in centroids.items()}
        self.scale=np.maximum(np.asarray(scale,dtype=np.float64),1e-4)
    @classmethod
    def fit(cls,rows:list[tuple[np.ndarray,str]]):
        matrix=np.stack([x for x,_ in rows]);scale=matrix.std(0)
        labels=sorted(set(label for _,label in rows))
        centroids={label:matrix[[y==label for _,y in rows]].mean(0).tolist() for label in labels}
        return cls(centroids,scale.tolist())
    def predict_features(self,features:np.ndarray,available:list[str]|None=None)->list[tuple[str,float]]:
        names=[x for x in self.centroids if available is None or x in available]
        scored=[(name,float(np.mean(((features-self.centroids[name])/self.scale)**2))) for name in names]
        return sorted(scored,key=lambda item:item[1])
    def rank(self,factory,seed:int,available:list[str])->list[tuple[str,float]]:
        return self.predict_features(probe_features(factory,seed),available)
    def save(self,path:Path)->None:
        atomic_json(path,{'format':'wailah-learned-strategy-selector-v1',
            'centroids':{k:v.tolist() for k,v in self.centroids.items()},'scale':self.scale.tolist()})
    @classmethod
    def load(cls,path:Path):
        value=json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(value['centroids'],value['scale'])


class AdaptiveRewardMemory:
    """Online contextual action learner that rapidly unlearns stale mappings."""
    def __init__(self,action_dim:int):
        self.action_dim=action_dim;self.values={};self.counts={};self.pending=None
    def act(self,observation:np.ndarray)->int:
        key=perceptual_state_key(observation)
        values=self.values.setdefault(key,np.zeros(self.action_dim,dtype=np.float64))
        counts=self.counts.setdefault(key,np.zeros(self.action_dim,dtype=np.int64))
        unseen=np.flatnonzero(counts==0)
        action=int(unseen[0]) if len(unseen) else int(np.argmax(values+.12/np.sqrt(counts)))
        self.pending=(key,action);return action
    def observe(self,reward:float)->None:
        if self.pending is None:return
        key,action=self.pending;self.counts[key][action]+=1
        # Strong recency lets a formerly good action lose authority after one clear
        # contradiction, while successful actions remain preferred.
        self.values[key][action]=.25*self.values[key][action]+.75*float(reward)


class TemporalRewardMemory:
    """Contextual learner whose state includes recent visual history."""
    def __init__(self,action_dim:int,history:int=4):
        self.action_dim=action_dim;self.history=deque(maxlen=history)
        self.values={};self.counts={};self.pending=None
    def act(self,observation:np.ndarray)->int:
        self.history.append(perceptual_state_key(observation));context=tuple(self.history)
        values=self.values.setdefault(context,np.zeros(self.action_dim,dtype=np.float64))
        counts=self.counts.setdefault(context,np.zeros(self.action_dim,dtype=np.int64))
        unseen=np.flatnonzero(counts==0)
        action=int(unseen[0]) if len(unseen) else int(np.argmax(values+.12/np.sqrt(counts)))
        self.pending=(context,action);return action
    def observe(self,reward:float)->None:
        if self.pending is None or reward==0:return
        context,action=self.pending;self.counts[context][action]+=1
        self.values[context][action]=.25*self.values[context][action]+.75*float(reward)
