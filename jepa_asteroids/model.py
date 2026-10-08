"""Paper Sections 3.2.1-3.2.2: shared spatial projector and action-conditioned predictor.

The two-layer projector structure comes from AD-E2E-JEPA, Eq. 7.
AdaLN/block-causal attention/axial RoPE follow the JEPA-WMs reference family.
The exact unpublished projector kernel and predictor width are choices, not facts:
see FIDELITY.md. No policy, value, reward, death or inverse-action heads exist here.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .config import Config

class SpatialProjector(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        k = cfg.projector_kernel
        p = 1 if k == 3 else 0
        self.convs = nn.Sequential(
            nn.Conv2d(cfg.encoder_dim, cfg.projector_hidden, k, stride=2, padding=p),
            nn.GELU(),
            nn.Conv2d(cfg.projector_hidden, cfg.projected_dim, k, stride=2, padding=p),
        )
        self.frame_chunk = cfg.projection_frame_chunk
        self.checkpointing = cfg.activation_checkpointing

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError('Expected frozen features [B,T,C,H,W].')
        b,t,c,h,w = features.shape
        if h % 4 or w % 4:
            raise ValueError('Feature grid must divide by four on both axes.')
        flat = features.reshape(b*t,c,h,w)
        parts = []
        for chunk in flat.split(self.frame_chunk):
            chunk = chunk.float()
            if self.training and self.checkpointing and torch.is_grad_enabled():
                out = checkpoint(self.convs, chunk, use_reentrant=False)
            else:
                out = self.convs(chunk)
            parts.append(out)
        z = torch.cat(parts, dim=0)
        return z.reshape(b,t,z.shape[1],z.shape[2],z.shape[3]).permute(0,1,3,4,2).contiguous()


def block_causal_mask(t: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    frame = torch.arange(t, device=device).repeat_interleave(h*w)
    # True means allowed in torch SDPA. All patches of this/past frames are visible.
    return frame[None, :] <= frame[:, None]


def rotate_axis(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Conventional adjacent-pair RoPE; positions [N], x [B,heads,N,D_even]."""
    d = x.shape[-1]
    if d == 0:
        return x
    omega = 10000.0 ** (-torch.arange(0,d,2,device=x.device,dtype=torch.float32)/d)
    angles = positions.float()[:,None] * omega[None,:]
    cosine = angles.cos().to(x.dtype)[None,None]
    sine = angles.sin().to(x.dtype)[None,None]
    pairs = x.reshape(*x.shape[:-1],d//2,2)
    even,odd = pairs.unbind(-1)
    return torch.stack((even*cosine-odd*sine, even*sine+odd*cosine),-1).flatten(-2)


def axial_rope(x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
    d = x.shape[-1]
    axis_d = 2*((d//3)//2)
    ids = torch.arange(t*h*w,device=x.device)
    positions = (ids//(h*w), (ids//w)%h, ids%w)
    # Native rectangular coordinates. No flattened 1D spatial position shortcut.
    pieces = [rotate_axis(x[...,i*axis_d:(i+1)*axis_d],positions[i]) for i in range(3)]
    pieces.append(x[...,3*axis_d:])
    return torch.cat(pieces,-1)


class SpatialTemporalAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim,3*dim)
        self.output = nn.Linear(dim,dim)

    def forward(self,x:torch.Tensor,t:int,h:int,w:int,mask:torch.Tensor)->torch.Tensor:
        b,n,d=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        q,k=axial_rope(q,t,h,w),axial_rope(k,t,h,w)
        out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0)
        return self.output(out.transpose(1,2).reshape(b,n,d))


class AdaLNBlock(nn.Module):
    def __init__(self,cfg:Config):
        super().__init__()
        d=cfg.predictor_dim
        self.norm1=nn.LayerNorm(d,eps=1e-6)
        self.norm2=nn.LayerNorm(d,eps=1e-6)
        self.attention=SpatialTemporalAttention(d,cfg.predictor_heads)
        self.mlp=nn.Sequential(nn.Linear(d,d*cfg.mlp_ratio),nn.GELU(),nn.Linear(d*cfg.mlp_ratio,d))
        self.modulation=nn.Sequential(nn.SiLU(),nn.Linear(d,6*d))

    def forward(self,x:torch.Tensor,action:torch.Tensor,t:int,h:int,w:int,mask:torch.Tensor)->torch.Tensor:
        parameters=self.modulation(action).repeat_interleave(h*w,dim=1)
        shift_a,scale_a,gate_a,shift_m,scale_m,gate_m=parameters.chunk(6,-1)
        v=self.norm1(x)*(1+scale_a)+shift_a
        x=x+gate_a*self.attention(v,t,h,w,mask)
        return x+gate_m*self.mlp(self.norm2(x)*(1+scale_m)+shift_m)


class ActionConditionedPredictor(nn.Module):
    def __init__(self,cfg:Config):
        super().__init__()
        self.input=nn.Linear(cfg.projected_dim,cfg.predictor_dim)
        self.action=nn.Linear(cfg.action_dim,cfg.predictor_dim)
        self.blocks=nn.ModuleList(AdaLNBlock(cfg) for _ in range(cfg.predictor_depth))
        self.norm=nn.LayerNorm(cfg.predictor_dim,eps=1e-6)
        self.output=nn.Linear(cfg.predictor_dim,cfg.projected_dim)
        self.checkpointing=cfg.activation_checkpointing
        self.action_dim=cfg.action_dim
        self.apply(self._initialize)
        with torch.no_grad():
            for i,block in enumerate(self.blocks,1):
                block.attention.output.weight.div_(math.sqrt(2*i))
                block.mlp[-1].weight.div_(math.sqrt(2*i))
                nn.init.trunc_normal_(block.modulation[-1].weight,std=0.02*cfg.adaln_init_scale)
                nn.init.zeros_(block.modulation[-1].bias)

    @staticmethod
    def _initialize(m:nn.Module)->None:
        if isinstance(m,nn.Linear):
            nn.init.trunc_normal_(m.weight,std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.LayerNorm):
            nn.init.ones_(m.weight);nn.init.zeros_(m.bias)

    def forward(self,z:torch.Tensor,actions:torch.Tensor)->torch.Tensor:
        if z.ndim!=5:
            raise ValueError('Projected embeddings must be [B,T,H,W,D].')
        b,t,h,w,d=z.shape
        if actions.shape!=(b,t,self.action_dim):
            raise ValueError(f'Expected actions {(b,t,self.action_dim)}, got {tuple(actions.shape)}.')
        x=self.input(z).reshape(b,t*h*w,-1)
        a=self.action(actions.float())
        mask=block_causal_mask(t,h,w,x.device)
        for block in self.blocks:
            if self.training and self.checkpointing and torch.is_grad_enabled():
                x=checkpoint(block,x,a,t,h,w,mask,use_reentrant=False)
            else:
                x=block(x,a,t,h,w,mask)
        return self.output(self.norm(x)).reshape(b,t,h,w,d)


class WorldModel(nn.Module):
    def __init__(self,cfg:Config):
        super().__init__()
        self.projector=SpatialProjector(cfg)
        self.predictor=ActionConditionedPredictor(cfg)
        self.cfg=cfg

    def forward(self,features:torch.Tensor,actions:torch.Tensor)->torch.Tensor:
        return self.predictor(self.projector(features),actions)
