"""Sealed frozen object-centric transfer audit on Warehouse Spill."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from .freeform_lab import run_object_episode
from .object_centric_agent import ObjectCentricAgent
from .warehouse_spill import WarehouseSpill


FROZEN_SHA = "D5B33EBDF56C4F49585AFEDEB87BB41CD277CE6DAF9C2DBE768D3FE27A6C1444"
SEALED_SEEDS = [710001, 713019, 716037, 719055, 722073, 725091,
                728109, 731127, 734145, 737163, 740181, 743199]


def agent_hash():
    return hashlib.sha256(Path(__file__).with_name("object_centric_agent.py").read_bytes()).hexdigest().upper()


def random_episode(game, budget, seed):
    rng = np.random.default_rng(seed); used = episodes = damage = 0; success = False
    while used < budget and not success:
        game.reset(); episodes += 1
        for _ in range(min(1500, budget - used)):
            _, _, done, info = game.step(int(rng.integers(5))); used += 1
            if done:
                damage += int(info["protected_damaged"]); success = bool(info["success"]); break
    return {"success": success, "interactions": used, "episodes": episodes, "protected_damaged": damage}


def summary(rows):
    return {"worlds": len(rows), "successes": sum(row["success"] for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "zero_damage_worlds": sum(row["protected_damaged"] == 0 for row in rows),
            "mean_interactions": float(np.mean([row["interactions"] for row in rows]))}


def run_object_transfer_audit(output, seeds=SEALED_SEEDS, budget=5000):
    before = agent_hash()
    if before != FROZEN_SHA: raise RuntimeError(f"frozen object agent hash mismatch: {before}")
    mind = ObjectCentricAgent(); learned = []; fresh = []; random_rows = []
    for seed in seeds:
        learned.append(run_object_episode(mind, WarehouseSpill(seed), budget, 1500))
        fresh.append(run_object_episode(ObjectCentricAgent(), WarehouseSpill(seed), budget, 1500))
        random_rows.append(random_episode(WarehouseSpill(seed), budget, seed + 151))
    revisits = [run_object_episode(mind, WarehouseSpill(seed), 1200, 1200) for seed in seeds]
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    brain = output.with_name("OBJECT_V17_WAREHOUSE_MIND.json"); mind.save(brain)
    restored = ObjectCentricAgent.load(brain)
    reload_rows = [run_object_episode(restored, WarehouseSpill(seed), 1200, 1200) for seed in seeds[:6]]
    after = agent_hash(); initial = summary(learned); revisit = summary(revisits)
    speedup = initial["mean_interactions"] / max(1.0, revisit["mean_interactions"])
    passed = (initial["successes"] >= 10 and initial["zero_damage_worlds"] >= initial["successes"]
              and summary(random_rows)["successes"] <= 1 and speedup >= 2
              and summary(reload_rows)["successes"] >= 5 and before == after)
    report = {"format": "wailah-object-v17-warehouse-transfer-audit-v1",
        "classification": "precommitted frozen object-centric transfer without the tiled visual schema",
        "frozen_agent": {"expected_sha256": FROZEN_SHA, "before_sha256": before,
                         "after_sha256": after, "unchanged": before == after},
        "protocol": {"seeds": list(seeds), "budget": budget, "revisit_budget": 1200,
                     "raw_observation_shape": [400, 540, 3],
                     "agent_not_given": ["segmentation", "effector identity", "controls", "object labels",
                                         "active pair", "coordinates", "private state"]},
        "aggregate": {"object_growing_mind": initial, "fresh_object_mind": summary(fresh),
                      "random_controls": summary(random_rows), "revisit": revisit,
                      "reloaded_memory": summary(reload_rows), "revisit_speedup": speedup},
        "precommitted_pass": passed, "growth": mind.status(), "per_world": learned,
        "persistent_brain": str(brain)}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8"); return report
