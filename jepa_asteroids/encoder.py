"""Actual frozen DINOv3 ViT-L/16 via the official Hugging Face checkpoint.

There is no random, smaller, telemetry or DINOv2 fallback. Only the explicit
'download' command permits network model retrieval. Other commands are offline.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .config import Config
from .runtime import ROOT,atomic_json,privacy_environment

MODEL_URL='https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m'


def checkpoint_location(cfg:Config)->Path:
    if cfg.encoder_path:
        path=Path(cfg.encoder_path).expanduser()
        return path if path.is_absolute() else ROOT/path
    record=ROOT/'encoder_location.json'
    if not record.exists():
        raise FileNotFoundError('DINOv3 is not set up. Run DOWNLOAD_ENCODER.bat after accepting the official model access conditions. No substitute encoder will be used.')
    data=json.loads(record.read_text(encoding='utf-8'))
    if data.get('model_id')!=cfg.encoder_id:
        raise ValueError('Cached encoder model identity does not match the configuration.')
    return Path(data['path'])


def download_encoder(cfg:Config,token:str|None=None)->dict:
    privacy_environment()
    from huggingface_hub import snapshot_download
    path=snapshot_download(repo_id=cfg.encoder_id,revision=cfg.encoder_revision,
        token=token or None,allow_patterns=['*.json','*.safetensors','README.md','LICENSE*'])
    location={'model_id':cfg.encoder_id,'requested_revision':cfg.encoder_revision,
              'path':str(Path(path).resolve()),'resolved_revision':Path(path).name}
    atomic_json(ROOT/'encoder_location.json',location)
    return location


def patch_tokens(last_hidden:torch.Tensor,registers:int,h:int,w:int,dim:int)->torch.Tensor:
    expected=1+registers+h*w
    if last_hidden.shape[1:]!=(expected,dim):
        raise ValueError(f'Unexpected DINOv3 output {tuple(last_hidden.shape)}; expected [B,{expected},{dim}].')
    return last_hidden[:,1+registers:].transpose(1,2).reshape(-1,dim,h,w).contiguous()


class FrozenDINOv3(nn.Module):
    def __init__(self,cfg:Config,device:torch.device):
        super().__init__();privacy_environment()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError('Install requirements.txt in the project environment first.') from exc
        path=checkpoint_location(cfg)
        if not (path/'config.json').exists() or not list(path.glob('*.safetensors')):
            raise FileNotFoundError(f'Expected an official Transformers DINOv3 checkpoint folder at {path}.')
        self.backbone,info=AutoModel.from_pretrained(str(path),local_files_only=True,
            trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32,
            output_loading_info=True)
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise RuntimeError(f'Incomplete encoder load; refusing random initialized weights: {info}')
        actual=self.backbone.config
        if actual.hidden_size!=1024 or actual.patch_size!=16 or actual.num_hidden_layers!=24:
            raise ValueError('Checkpoint is not the required DINOv3 ViT-L/16 architecture.')
        if getattr(actual,'model_type',None)!='dinov3_vit':
            raise ValueError('Checkpoint model_type must be dinov3_vit.')
        self.backbone.requires_grad_(False)
        self.backbone.eval().to(device)
        self.cfg=cfg;self.device=device
        self.registers=actual.num_register_tokens
        self.identity={'model_id':cfg.encoder_id,'snapshot':path.name,
                       'config_sha256':__import__('hashlib').sha256((path/'config.json').read_bytes()).hexdigest(),
                       'height':cfg.image_height,'width':cfg.image_width,
                       'normalization':'imagenet-rgb','features':'normalized patch tokens, CLS/registers removed',
                       'inference_dtype':'float32'}

    def train(self,mode:bool=True):
        super().train(False)
        if hasattr(self,'backbone'):self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self,frames:np.ndarray|torch.Tensor)->torch.Tensor:
        # Input is the actual rendered RGB frame, not an engine state dictionary.
        pixels=torch.as_tensor(frames,device=self.device)
        if pixels.ndim!=4 or tuple(pixels.shape[1:])!=(self.cfg.image_height,self.cfg.image_width,3):
            raise ValueError('Encoder expects RGB [N,H,W,3] at the configured camera resolution.')
        if pixels.dtype!=torch.uint8:
            raise ValueError('Camera input must be uint8, avoiding ambiguous normalization.')
        x=pixels.permute(0,3,1,2).float()/255.0
        mean=x.new_tensor([0.485,0.456,0.406])[None,:,None,None]
        std=x.new_tensor([0.229,0.224,0.225])[None,:,None,None]
        output=self.backbone(pixel_values=(x-mean)/std).last_hidden_state
        h,w=self.cfg.grid
        return patch_tokens(output,self.registers,h,w,self.cfg.encoder_dim)
