"""Explicit, real-weight integration check to run after the official download."""
from __future__ import annotations
from pathlib import Path
import time
import torch
from .config import Config
from .data import game_config,progress
from .encoder import FrozenDINOv3
from .engine import Asteroids
from .model import WorldModel
from .runtime import Guard,atomic_json


def verify_real_encoder(cfg:Config,workspace:Path,device:torch.device,guard:Guard)->dict:
    progress(workspace,'verify',message='Loading the actual official DINOv3 checkpoint. No fallback is permitted.')
    start=time.monotonic();encoder=FrozenDINOv3(cfg,device);guard.check(force=True)
    game=Asteroids(game_config(cfg),seed=cfg.seed)
    frames=[game.observe()]
    for _ in range(cfg.history-1):
        frame,_,ended,timeout,_=game.step(0);frames.append(frame)
        if ended or timeout:raise RuntimeError('Smoke-test game ended during context warm-up.')
    import numpy as np
    frames=torch.from_numpy(np.stack(frames))
    model=WorldModel(cfg).to(device).eval();guard.check(force=True)
    with torch.no_grad():
        pieces=[]
        for batch in frames.split(cfg.encoder_batch):
            guard.check();pieces.append(encoder(batch))
        features=torch.cat(pieces).unsqueeze(0)
        z=model.projector(features)
        actions=torch.nn.functional.one_hot(torch.zeros(1,cfg.history,dtype=torch.long,device=device),cfg.action_dim).float()
        predicted=model.predictor(z,actions)
    if not torch.isfinite(predicted).all():raise FloatingPointError('Non-finite integration-check prediction.')
    if z.shape!=(1,cfg.history,*cfg.projected_grid,cfg.projected_dim):raise RuntimeError('Unexpected projected patch geometry.')
    report={'real_encoder_weights_loaded':True,'encoder':encoder.identity,'encoder_frozen':all(not p.requires_grad for p in encoder.parameters()),
        'features_shape':list(features.shape),'projected_shape':list(z.shape),'prediction_shape':list(predicted.shape),
        'finite':True,'device':str(device),'torch':torch.__version__,'seconds':time.monotonic()-start,
        'scope':'Real pretrained visual-encoder integration; projector/predictor are fresh random weights. NOT learned gameplay.'}
    atomic_json(workspace/'real_encoder_verification.json',report)
    progress(workspace,'verified',message='Real DINOv3 load and full-shape forward passed. No game-training claim is made.')
    return report
