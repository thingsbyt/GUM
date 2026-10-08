"""Noise-tolerant visual state identities for episodic reasoning skills."""
from __future__ import annotations

import hashlib

import numpy as np


def perceptual_state_key(observation: np.ndarray) -> str:
    """Describe stable, high-salience structure instead of exact sensor bytes.

    Episodic reasoning needs to recognize the same state across harmless pixel
    noise.  The benchmark worlds render causal traces and controllable objects at
    well-separated intensity bands.  Two nested masks retain their geometry while
    discarding bounded background flicker; packing the masks keeps graph keys small.
    This consumes pixels only—no coordinates or environment state.
    """
    pixels=np.asarray(observation,dtype=np.uint8)
    if pixels.ndim < 2: raise ValueError('visual state must have at least two dimensions')
    salient=np.stack((pixels >= 128,pixels >= 208),axis=0)
    packed=np.packbits(np.ascontiguousarray(salient).reshape(-1))
    return hashlib.sha256(packed.tobytes()).hexdigest()
