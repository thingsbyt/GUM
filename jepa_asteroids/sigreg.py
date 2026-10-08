"""AD-E2E-JEPA Eq. 10 and Appendix A.1, using the LeWM Epps-Pulley convention.

Across-batch empirical characteristic functions are computed separately at EACH
(time, patch) location. Never pool batch/time/patch into one sample axis.
17 trapezoidal knots in [0,3], 1024 unit directions by default.

The custom first-order autograd operation recomputes bounded site/direction chunks
in backward. It preserves the full-batch statistic, unlike batch micro-averaging.
It is checked against the direct expression for both values and gradients.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.autograd.function import once_differentiable


def direct_statistic(z:torch.Tensor,directions:torch.Tensor,knots:int=17)->torch.Tensor:
    """Slow, transparent reference: z [sites,batch,dim], directions [dim,M]."""
    t=torch.linspace(0,3,knots,device=z.device,dtype=z.dtype)
    phi=torch.exp(-0.5*t.square())
    trapezoid=torch.full_like(t,6.0/(knots-1))
    trapezoid[0]=trapezoid[-1]=3.0/(knots-1)
    weights=trapezoid*phi
    phase=(z@directions).unsqueeze(-1)*t
    error=(phase.cos().mean(1)-phi).square()+phase.sin().mean(1).square()
    return ((error*weights).sum(-1)*z.shape[1]).mean()


class _ChunkedStatistic(torch.autograd.Function):
    @staticmethod
    def forward(ctx,z,directions,t,phi,weights,site_chunk,direction_chunk):
        sites,batch,_=z.shape
        m=directions.shape[1]
        total=z.new_zeros(())
        for si in range(0,sites,site_chunk):
            block=z[si:si+site_chunk]
            for mi in range(0,m,direction_chunk):
                a=directions[:,mi:mi+direction_chunk]
                phase=(block@a).unsqueeze(-1)*t
                real=phase.cos().mean(1)
                imag=phase.sin().mean(1)
                total+=(((real-phi).square()+imag.square())*weights).sum()*batch
        ctx.save_for_backward(z,directions,t,phi,weights)
        ctx.site_chunk=site_chunk;ctx.direction_chunk=direction_chunk
        return total/(sites*m)

    @staticmethod
    @once_differentiable
    def backward(ctx,grad_output):
        z,directions,t,phi,weights=ctx.saved_tensors
        sites,batch,_=z.shape
        m=directions.shape[1]
        grad=torch.zeros_like(z)
        for si in range(0,sites,ctx.site_chunk):
            block=z[si:si+ctx.site_chunk]
            for mi in range(0,m,ctx.direction_chunk):
                a=directions[:,mi:mi+ctx.direction_chunk]
                phase=(block@a).unsqueeze(-1)*t
                c,s=phase.cos(),phase.sin()
                real=c.mean(1)-phi
                imag=s.mean(1)
                # Batch multiplier in the Epps-Pulley statistic cancels the 1/B
                # from differentiating the empirical characteristic function.
                dy=(2*weights*t*(-real[:,None]*s+imag[:,None]*c)).sum(-1)
                grad[si:si+ctx.site_chunk].add_(dy@a.T)
        grad*=grad_output/(sites*m)
        return grad,None,None,None,None,None,None


class PatchSIGReg(nn.Module):
    def __init__(self,directions:int=1024,knots:int=17,site_chunk:int=8,direction_chunk:int=64):
        super().__init__()
        if directions<1 or knots<2 or site_chunk<1 or direction_chunk<1:
            raise ValueError('Invalid SIGReg settings.')
        self.directions=directions;self.knots=knots
        self.site_chunk=site_chunk;self.direction_chunk=direction_chunk

    def forward(self,z:torch.Tensor,directions:torch.Tensor|None=None)->torch.Tensor:
        if z.ndim!=5 or z.shape[0]<2:
            raise ValueError('SIGReg expects [B,T,H,W,D] and B >= 2.')
        b,t,h,w,d=z.shape
        # Float64 is retained for numerical tests; production computes in FP32.
        dtype=torch.float64 if z.dtype==torch.float64 else torch.float32
        sites=z.permute(1,2,3,0,4).reshape(t*h*w,b,d).to(dtype)
        if directions is None:
            a=torch.randn(d,self.directions,device=z.device,dtype=dtype)
            a=a/a.norm(dim=0,keepdim=True).clamp_min(torch.finfo(dtype).eps)
        else:
            a=directions.to(device=z.device,dtype=dtype).detach()
            if a.shape!=(d,self.directions):
                raise ValueError('Directions must have shape [embedding_dim,M].')
            if not torch.allclose(a.norm(dim=0),torch.ones(self.directions,device=z.device,dtype=dtype),atol=1e-5):
                raise ValueError('SIGReg projection directions must have unit norm.')
        knots=torch.linspace(0,3,self.knots,device=z.device,dtype=dtype)
        phi=torch.exp(-knots.square()/2)
        weights=torch.full_like(knots,6/(self.knots-1))
        weights[0]=weights[-1]=3/(self.knots-1)
        weights=weights*phi
        return _ChunkedStatistic.apply(sites,a,knots,phi,weights,self.site_chunk,self.direction_chunk)
