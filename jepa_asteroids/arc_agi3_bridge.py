"""Read-only WAILAH policy bridge for the external ARC-AGI-3 runtime.

The current WAILAH environment uses Python 3.10/PyTorch while the official
ARC-AGI-3 SDK requires Python 3.12.  This tiny JSON-lines bridge keeps those
runtimes isolated.  It receives pixels and the currently available action IDs;
it never receives a game name, source code, rules, title, or baseline solution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .living import LivingSystem


def _eye(frame_layers: list, shape: tuple[int, int, int]) -> np.ndarray:
    """Turn one or more categorical ARC layers into WAILAH's grayscale eye."""
    layers = np.asarray(frame_layers, dtype=np.int16)
    if layers.ndim == 2:
        layers = layers[None]
    if layers.ndim != 3:
        raise ValueError(f"expected ARC frame layers, got shape {layers.shape}")
    # Later non-zero layers visually cover earlier layers in the official
    # renderer.  Preserve category boundaries instead of smoothing them.
    composed = np.zeros(layers.shape[1:], dtype=np.int16)
    for layer in layers:
        composed = np.where(layer != 0, layer, composed)
    scaled = np.clip(composed, 0, 15).astype(np.uint8) * 17
    _, height, width = shape
    resized = np.asarray(
        Image.fromarray(scaled, mode="L").resize((width, height), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    if shape[0] == 1:
        return resized[None]
    return np.repeat(resized[None], shape[0], axis=0)


def serve(workspace: Path) -> int:
    living = LivingSystem(Path(workspace), "cpu")
    expected_shape = tuple(living.experts[0].spec.observation_shape)
    for raw in sys.stdin:
        try:
            message = json.loads(raw)
            action_ids = [int(value) for value in message["available_actions"] if int(value) != 0]
            observation = _eye(message["frame"], expected_shape)
            compatible = [expert.task_id for expert in living.experts
                          if expert.spec.action_dim == len(action_ids)]
            if not action_ids:
                answer = {"error": "no non-reset action available"}
            elif compatible:
                index, evidence = living.act(observation, len(action_ids), top_k=2)
                answer = {"action_id": action_ids[index], "policy": "frozen-wailah",
                          "compatible_experts": compatible, "evidence": evidence}
            else:
                # This is an explicit failure mode, not a hidden random policy.
                answer = {"error": f"no expert supports {len(action_ids)} actions",
                          "compatible_experts": []}
        except Exception as exc:  # keep the external harness alive for audit output
            answer = {"error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(answer, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    return serve(parser.parse_args(argv).workspace)


if __name__ == "__main__":
    raise SystemExit(main())
