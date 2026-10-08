"""Sealed evaluation of the byte-frozen v14 learner on Crystal Garden."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from .crystal_garden import CrystalGardenGame
from .universal_goal_state import UniversalGoalStateAgent


FROZEN_SHA256 = "F20B968908C1CB3AB40CECE3B9157F5E4A838538A37350CA07C5E8A1E8118586"
SEALED_SEEDS = [420001, 422132, 424263, 426394, 428525, 430656,
                432787, 434918, 437049, 439180, 441311, 443442]


def agent_hash():
    return hashlib.sha256(Path(__file__).with_name("universal_goal_state.py").read_bytes()).hexdigest().upper()


def run_frozen_episode(agent, game, budget=3000, horizon=300):
    used = episodes = damage = 0; success = False; final = {}; trace = []
    while used < budget and not success:
        frame = game.reset(); agent.begin(frame); episodes += 1
        for _ in range(min(horizon, budget - used)):
            action, evidence = agent.act(frame); nxt, reward, done, info = game.step(action)
            learned = agent.observe(frame, action, nxt, reward, done); used += 1; frame = nxt; final = info
            if evidence["reason"] in ("test-two-object-composition", "apply-treatment-under-goal-constraints"):
                trace.append({"interaction": used, **evidence, "learned": learned})
            if done:
                damage += int(info["protected_damaged"]); success = bool(info["success"]); break
    return {"success": success, "interactions": used, "episodes": episodes,
            "protected_damaged": damage, "threat_remaining": int(final.get("threat_remaining", -1)),
            "recipe_learned": agent.memory.get("successful_pair") is not None,
            "controls_grounded": len(agent.memory["controls"]) == 4 and agent.memory["interact"] is not None,
            "trace": trace[-24:]}


def run_random(game, budget, seed):
    rng = np.random.default_rng(seed); used = episodes = damage = 0; success = False
    while used < budget and not success:
        game.reset(); episodes += 1
        for _ in range(min(300, budget - used)):
            _, _, done, info = game.step(int(rng.integers(5))); used += 1
            if done:
                damage += int(info["protected_damaged"]); success = bool(info["success"]); break
    return {"success": success, "interactions": used, "episodes": episodes, "protected_damaged": damage}


def summary(rows):
    return {"worlds": len(rows), "successes": sum(row["success"] for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "zero_damage_worlds": sum(row["protected_damaged"] == 0 for row in rows),
            "mean_interactions": float(np.mean([row["interactions"] for row in rows]))}


def run_frozen_transfer_audit(output, *, seeds=SEALED_SEEDS, budget=3000):
    before = agent_hash()
    if before != FROZEN_SHA256: raise RuntimeError(f"frozen v14 hash mismatch: {before}")
    mind = UniversalGoalStateAgent(); learned = []; fresh = []; random_rows = []
    for seed in seeds:
        learned.append(run_frozen_episode(mind, CrystalGardenGame(seed), budget))
        fresh.append(run_frozen_episode(UniversalGoalStateAgent(), CrystalGardenGame(seed), budget))
        random_rows.append(run_random(CrystalGardenGame(seed), budget, seed + 71))
    revisits = [run_frozen_episode(mind, CrystalGardenGame(seed), 800) for seed in seeds]
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    brain = output.with_name("FROZEN_V14_TRANSFER_MIND.json"); mind.save(brain)
    restored = UniversalGoalStateAgent.load(brain)
    reload_rows = [run_frozen_episode(restored, CrystalGardenGame(seed), 800) for seed in seeds[:6]]
    after = agent_hash(); initial = summary(learned); revisit = summary(revisits)
    speedup = initial["mean_interactions"] / max(1.0, revisit["mean_interactions"])
    passed = (initial["successes"] >= 10 and initial["zero_damage_worlds"] == len(seeds)
              and summary(random_rows)["successes"] <= 1 and speedup >= 2.0 and before == after)
    report = {"format": "wailah-frozen-v14-held-out-transfer-audit-v1",
        "classification": "precommitted frozen held-out transfer within the tiled visual interface",
        "frozen_agent": {"expected_sha256": FROZEN_SHA256, "before_sha256": before,
                         "after_sha256": after, "unchanged": before == after},
        "protocol": {"seeds": list(seeds), "budget": budget, "revisit_budget": 800,
                     "inputs": ["RGB pixels", "five anonymous controls", "reward", "done"],
                     "agent_not_given": ["semantic labels", "controls", "active pair", "coordinates",
                                         "private simulator state"]},
        "aggregate": {"frozen_growing_mind": initial, "fresh_frozen_mind": summary(fresh),
                      "random_controls": summary(random_rows), "revisit": revisit,
                      "reloaded_memory": summary(reload_rows), "revisit_speedup": speedup},
        "precommitted_pass": passed, "growth": mind.status(), "per_world": learned,
        "persistent_brain": str(brain)}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8"); return report
