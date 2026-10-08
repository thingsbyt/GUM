"""Direct implementation of AD-E2E-JEPA Eqs. 9-12 and 15-23.

Frame convention: z[:,0:H] is context; z[:,H+j] is future step j+1.
a[:,i] is the action connecting z[:,i] to z[:,i+1]. No rewards are accepted.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .config import Config
from .sigreg import PatchSIGReg


def target_mse(prediction:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    return F.mse_loss(prediction,target.detach())


@dataclass
class LossTerms:
    total:torch.Tensor
    teacher_forcing:torch.Tensor
    rollout:torch.Tensor
    sigreg:torch.Tensor
    prediction:torch.Tensor

    def scalars(self)->dict:
        return {k:float(getattr(self,k).detach()) for k in self.__dataclass_fields__}


def loss_from_embeddings(predictor,z:torch.Tensor,actions:torch.Tensor,cfg:Config,
                         regularizer:PatchSIGReg,directions:torch.Tensor|None=None,check=None)->LossTerms:
    h=cfg.history
    f=cfg.horizon if cfg.rollout_training else 1
    if z.shape[1]!=h+f or actions.shape[:2]!=(z.shape[0],h+f-1):
        raise ValueError(f'Expected {h+f} consecutive frames and {h+f-1} aligned actions.')
    # Eq. 16: supervise EVERY output of the first window.
    if check is not None:check()
    first=predictor(z[:,:h],actions[:,:h])
    initial=target_mse(first,z[:,1:h+1])
    zero=z.new_zeros(())
    if not cfg.rollout_training:
        reg=regularizer(z,directions)
        return LossTerms(initial+cfg.sigreg_weight*reg,initial,zero,reg,initial)

    # Eqs. 17-18: later teacher-forced windows supervise ONLY their last output.
    weighted_tf=h*initial
    for k in range(1,f):
        if check is not None:check()
        pred=predictor(z[:,k:k+h],actions[:,k:k+h])
        weighted_tf=weighted_tf+target_mse(pred[:,-1],z[:,h+k])
    # K+F == H+F-1, not H+F and not F.
    tf=weighted_tf/(h+f-1)

    # Eqs. 19-23: reuse first TF call. Stop the entire AR context on later calls.
    context=torch.cat((z[:,1:h],first[:,-1:]),dim=1)
    ar_sum=zero
    for k in range(1,f):
        if check is not None:check()
        pred=predictor(context.detach(),actions[:,k:k+h])
        final=pred[:,-1:]
        ar_sum=ar_sum+target_mse(final[:,0],z[:,h+k])
        context=torch.cat((context[:,1:],final),dim=1)
    # Eq. 12 averages the TF component plus F-1 AR horizon losses over F.
    prediction=(tf+ar_sum)/f
    reg=regularizer(z,directions)  # all H+F observed frames, no detach
    return LossTerms(prediction+cfg.sigreg_weight*reg,tf,ar_sum,reg,prediction)
