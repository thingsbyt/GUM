"""Controlled A→remapped-B→A proof for color-invariant causal roles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .concept_memory import object_concept
from .role_schema import CausalRoleSchema


def _object(color, area_bin, width, height, x, y):
    return {"color": int(color), "area": int(2 ** area_bin), "w": int(width), "h": int(height),
            "cx": float(x), "cy": float(y), "sig": (color, area_bin, width, height)}


def _world(seed: int, palette: list[int]):
    rng = np.random.default_rng(seed)
    specs = [(0, 1, 1), (3, 5, 2), (2, 2, 5), (4, 6, 6), (1, 3, 1),
             (5, 8, 5), (2, 1, 4), (4, 3, 7), (1, 4, 1), (3, 2, 6)]
    order = rng.permutation(len(specs)); positions = rng.choice(np.arange(4, 58), (len(specs), 2), replace=False)
    objects = []
    for index, spec_index in enumerate(order):
        area, width, height = specs[int(spec_index)]
        color = palette[index % len(palette)]
        x, y = positions[index]
        objects.append((int(spec_index), _object(color, area, width, height, x, y)))
    # The role structure is invariant: spec 0 is a disappearing prerequisite;
    # spec 1 is the terminal. Colors, locations, and candidate ordering change.
    grid = np.zeros((64, 64), dtype=np.uint8)
    for _, obj in objects:
        x, y = int(obj["cx"]), int(obj["cy"]); w, h = obj["w"], obj["h"]
        grid[max(0, y-h//2):min(64, y+(h+1)//2),
             max(0, x-w//2):min(64, x+(w+1)//2)] = obj["color"]
    return objects, hashlib.sha256(grid.tobytes()).hexdigest()


def _solve(schema: CausalRoleSchema, seed: int, palette: list[int]) -> dict:
    objects, visual_hash = _world(seed, palette); rng = np.random.default_rng(seed + 9001)
    remaining = list(objects); prerequisite_done = False; steps = 0; schema.reset_episode()
    while remaining and steps < 32:
        scored = []
        for role_id, obj in remaining:
            concept = object_concept(obj)
            score = schema.priority(concept)
            scored.append((score, float(rng.random()), role_id, obj, concept))
        _, _, role_id, obj, concept = max(scored)
        steps += 1
        if role_id == 0:
            remaining.remove((role_id, obj))
            prerequisite_done = True; schema.note_contact(concept)
        elif role_id == 1 and prerequisite_done:
            remaining.remove((role_id, obj))
            schema.note_contact(concept); schema.learn_from_progress()
            return {"success": True, "steps": steps, "visual_hash": visual_hash,
                    "colors": sorted({obj["color"] for _, obj in objects})}
        elif role_id != 1:
            remaining.remove((role_id, obj))
    return {"success": False, "steps": steps, "visual_hash": visual_hash,
            "colors": sorted({obj["color"] for _, obj in objects})}


def run(output: Path, variants: int = 128) -> dict:
    low_palette, high_palette = [1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]
    shared = CausalRoleSchema()
    a_train = _solve(shared, 610_001, low_palette)
    if not a_train["success"]: raise RuntimeError("training analogue did not terminate")
    transfer_rows, scratch_rows = [], []
    for index in range(variants):
        seed = 620_000 + index
        transfer_rows.append(_solve(shared, seed, high_palette))
        scratch_rows.append(_solve(CausalRoleSchema(), seed, high_palette))
    a_revisit = _solve(shared, 610_001, low_palette)
    transfer_steps = np.asarray([row["steps"] for row in transfer_rows], dtype=float)
    scratch_steps = np.asarray([row["steps"] for row in scratch_rows], dtype=float)
    report = {
        "format": "wailah-structural-transfer-v1",
        "protocol": {"sequence": "A -> 128 independently remapped B variants -> A",
                     "transferred": "color-invariant prerequisite/terminal role prototypes only",
                     "changed": ["all colors", "all positions", "candidate order", "visual frame hash"],
                     "not_transferred": ["colors", "coordinates", "action sequence", "layout", "solution"]},
        "a_training": a_train,
        "b_transfer": {"variants": variants, "successes": sum(row["success"] for row in transfer_rows),
                       "mean_steps": float(transfer_steps.mean()), "median_steps": float(np.median(transfer_steps)),
                       "rows": transfer_rows},
        "b_scratch": {"variants": variants, "successes": sum(row["success"] for row in scratch_rows),
                      "mean_steps": float(scratch_steps.mean()), "median_steps": float(np.median(scratch_steps)),
                      "rows": scratch_rows},
        "positive_transfer": {"mean_step_reduction": float(scratch_steps.mean() - transfer_steps.mean()),
                              "relative_step_reduction": float(1.0 - transfer_steps.mean() / scratch_steps.mean()),
                              "paired_variants_faster": int(np.sum(transfer_steps < scratch_steps))},
        "a_revisit": a_revisit,
        "retention": {"successful": bool(a_revisit["success"]),
                      "steps_before": a_train["steps"], "steps_after": a_revisit["steps"]},
        "role_schema": shared.status(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=128); args = parser.parse_args(argv)
    report = run(args.output.resolve(), args.variants)
    print(json.dumps({"a_training": report["a_training"], "b_transfer": {
        k: v for k, v in report["b_transfer"].items() if k != "rows"}, "b_scratch": {
        k: v for k, v in report["b_scratch"].items() if k != "rows"},
        "positive_transfer": report["positive_transfer"], "retention": report["retention"]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
