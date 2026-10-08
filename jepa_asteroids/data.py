"""RGB/action-only experience, frozen-feature caching, and episode-safe windows."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import Config
from .engine import Asteroids, GameConfig
from .rendering import png_bytes
from .runtime import Guard, atomic_json, atomic_bytes, disk_budget, replace_file

FORMAT='ad-e2e-jepa-asteroids-rgb-v1'
SPLIT_OFFSETS={'train':0,'validation':1_000_000_000,'test':2_000_000_000}


def game_config(cfg:Config)->GameConfig:
    return GameConfig(difficulty=cfg.difficulty,max_steps=cfg.episode_steps,
        frames_per_action=cfg.frames_per_action,image_width=cfg.image_width,
        image_height=cfg.image_height)


def domain(cfg:Config)->dict:
    return {'format':FORMAT,'game':asdict(game_config(cfg)),
            'observation':'RGB only, no HUD','action_count':cfg.action_dim,
            'action_semantics':['coast','left','right','thrust','fire']}


def fingerprint(value:dict)->str:
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def progress(workspace:Path,stage:str,**items)->None:
    atomic_json(workspace/'status.json',{'stage':stage,**items})


def load_manifest(workspace:Path,cfg:Config)->dict:
    path=workspace/'dataset.json'
    if not path.exists():raise RuntimeError('No recorded data. Run collect first.')
    value=json.loads(path.read_text(encoding='utf-8'))
    if value.get('format')!=FORMAT or value.get('domain')!=domain(cfg):
        raise RuntimeError('Dataset domain/config differs. Use a new workspace rather than mixing data.')
    for entry in value['episodes']:
        for key in ('raw','cache'):
            if key in entry and Path(entry[key]).name!=entry[key]:
                raise ValueError('Invalid episode filename in manifest.')
    return value


def collect(cfg:Config,workspace:Path,guard:Guard)->dict:
    workspace.mkdir(parents=True,exist_ok=True);rawdir=workspace/'raw';rawdir.mkdir(exist_ok=True)
    desired={s:getattr(cfg,f'{s}_episodes') for s in SPLIT_OFFSETS}
    if (workspace/'dataset.json').exists():
        manifest=load_manifest(workspace,cfg)
        if manifest.get('seed')!=cfg.seed or manifest.get('requested')!=desired:
            raise RuntimeError('Collection settings changed. Use a new workspace to avoid silent split changes.')
    else:
        manifest={'format':FORMAT,'domain':domain(cfg),'seed':cfg.seed,'requested':desired,
                  'episodes':[],'collection_policy':'random buttons, 50% persistence; no aiming rules'}
        atomic_json(workspace/'dataset.json',manifest)
    existing={e['id'] for e in manifest['episodes']};total=sum(desired.values())
    for split,count in desired.items():
        for index in range(count):
            guard.check(force=True)
            identity=f'{split}_{index:05d}'
            if identity in existing:continue
            seed=cfg.seed+SPLIT_OFFSETS[split]+index
            game=Asteroids(game_config(cfg),seed)
            rng=np.random.default_rng(seed+40_000_000)
            frames=[game.observe()];actions=[];action=int(rng.integers(cfg.action_dim))
            for _ in range(cfg.episode_steps):
                guard.check()
                if rng.random()<0.5:action=int(rng.integers(cfg.action_dim))
                frame,_,ended,timeout,_=game.step(action)  # rewards/telemetry discarded
                frames.append(frame);actions.append(action)
                if ended or timeout:break
            payload=np.stack(frames)
            disk_budget(workspace,payload.nbytes,cfg)  # conservative pre-compression check
            name=identity+'.npz';tmp=rawdir/(name+'.tmp')
            with tmp.open('wb') as handle:
                np.savez_compressed(handle,frames=payload,actions=np.asarray(actions,dtype=np.int64))
            replace_file(tmp,rawdir/name)
            manifest['episodes'].append({'id':identity,'split':split,'seed':seed,'raw':name,
                'frames':len(frames),'terminated':bool(game.terminated),'truncated':bool(game.truncated)})
            atomic_json(workspace/'dataset.json',manifest)
            atomic_bytes(workspace/'current.png',png_bytes(frames[-1]))
            progress(workspace,'collect',episodes=len(manifest['episodes']),total_episodes=total,
                     message=f'Recorded {identity}; RGB frames and button IDs only.')
    progress(workspace,'collected',episodes=len(manifest['episodes']),message='Next: encode RGB frames with the real frozen DINOv3.')
    return manifest


def validate_raw(payload,entry:dict,cfg:Config)->None:
    if set(payload.files)!={'frames','actions'}:raise ValueError('Raw episode must contain only frames and actions.')
    frames,actions=payload['frames'],payload['actions']
    if frames.dtype!=np.uint8 or frames.shape!=(entry['frames'],cfg.image_height,cfg.image_width,3):
        raise ValueError('Invalid RGB episode shape or dtype.')
    if actions.shape!=(len(frames)-1,) or not np.issubdtype(actions.dtype,np.integer):
        raise ValueError('Misaligned actions: action[i] must connect frame[i] to frame[i+1].')
    if len(actions) and (actions.min()<0 or actions.max()>=cfg.action_dim):raise ValueError('Invalid action IDs.')


def encode_dataset(cfg:Config,workspace:Path,encoder,guard:Guard)->dict:
    """Encoder dependency is explicit for unit testing. Production always passes FrozenDINOv3."""
    manifest=load_manifest(workspace,cfg);cachedir=workspace/'features';cachedir.mkdir(exist_ok=True)
    encoding={'encoder':encoder.identity,'storage_dtype':cfg.feature_dtype,'domain':domain(cfg)}
    if 'encoding' in manifest and manifest['encoding']!=encoding:
        raise RuntimeError('Feature cache encoder/precision differs. Use a new workspace.')
    manifest['encoding']=encoding
    for number,entry in enumerate(manifest['episodes']):
        guard.check(force=True)
        shape=(entry['frames'],cfg.encoder_dim,*cfg.grid)
        name=entry['id']+'.npy';out=cachedir/name
        if entry.get('cache')==name and out.exists():
            arr=np.load(out,mmap_mode='r',allow_pickle=False)
            if arr.shape!=shape or arr.dtype!=np.dtype(cfg.feature_dtype):raise ValueError('Invalid existing feature cache.')
            del arr
            continue
        size=int(np.prod(shape))*np.dtype(cfg.feature_dtype).itemsize
        disk_budget(workspace,size,cfg)
        with np.load(workspace/'raw'/entry['raw'],allow_pickle=False) as raw:
            validate_raw(raw,entry,cfg);frames=raw['frames']
            temp=out.with_suffix('.tmp.npy')
            cache=np.lib.format.open_memmap(temp,mode='w+',dtype=cfg.feature_dtype,shape=shape)
            try:
                for start in range(0,len(frames),cfg.encoder_batch):
                    guard.check()
                    batch=torch.from_numpy(frames[start:start+cfg.encoder_batch])
                    with torch.no_grad():features=encoder(batch).float().cpu().numpy()
                    if features.shape!=(len(batch),*shape[1:]) or not np.isfinite(features).all():
                        raise ValueError('Encoder produced invalid patch features.')
                    cache[start:start+len(batch)]=features
                    progress(workspace,'encode',episode=entry['id'],episode_index=number+1,
                        total_episodes=len(manifest['episodes']),frame=start+len(batch),frames=len(frames),
                        message='Caching frozen DINOv3 spatial patches, not training the encoder.')
                cache.flush()
            finally:
                del cache
            replace_file(temp,out)
        entry['cache']=name
        atomic_json(workspace/'dataset.json',manifest)
    progress(workspace,'encoded',message='Frozen feature cache complete. Next: train the projector and predictor.')
    return manifest


class EpisodeWindows(Dataset):
    def __init__(self,cfg:Config,workspace:Path,split:str,*,window_frames:int|None=None):
        self.cfg=cfg;self.workspace=workspace;self.manifest=load_manifest(workspace,cfg)
        self.window=window_frames or cfg.sequence_frames
        self.windows=[];self.entries={};self._maps={};self._actions={}
        for entry in self.manifest['episodes']:
            if entry['split']!=split:continue
            if 'cache' not in entry:raise RuntimeError('Feature cache incomplete. Run encode first.')
            self.entries[entry['id']]=entry
            self.windows.extend((entry['id'],start) for start in range(max(0,entry['frames']-self.window+1)))
        if not self.windows:raise RuntimeError(f'No usable {split} windows. Record longer or more episodes.')

    def __len__(self):return len(self.windows)

    def __getitem__(self,index):
        name,start=self.windows[index]
        if name not in self._maps:
            entry=self.entries[name]
            self._maps[name]=np.load(self.workspace/'features'/entry['cache'],mmap_mode='r',allow_pickle=False)
            with np.load(self.workspace/'raw'/entry['raw'],allow_pickle=False) as raw:
                self._actions[name]=raw['actions'].copy()
        features=torch.from_numpy(np.array(self._maps[name][start:start+self.window],copy=True))
        ids=torch.from_numpy(self._actions[name][start:start+self.window-1].copy()).long()
        actions=torch.nn.functional.one_hot(ids,self.cfg.action_dim).float()
        return features,actions
