"""Precommitted frozen adaptive-agent audit on Data Recovery Console."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from .adaptive_visual_agent import AdaptiveUniversalAgent
from .data_recovery_console import DataRecoveryConsole


ADAPTER_SHA = "EB97E5927255A249899C95422C898A015E7CE5938C070C82D42956DCC55DE31A"
CORE_SHA = "F20B968908C1CB3AB40CECE3B9157F5E4A838538A37350CA07C5E8A1E8118586"
SEALED_SEEDS = [510001, 512729, 515457, 518185, 520913, 523641,
                526369, 529097, 531825, 534553, 537281, 540009]


def hashes():
    root = Path(__file__).parent
    return {"adapter": hashlib.sha256((root / "adaptive_visual_agent.py").read_bytes()).hexdigest().upper(),
            "core": hashlib.sha256((root / "universal_goal_state.py").read_bytes()).hexdigest().upper()}


def run_episode(agent, game, budget=4000, horizon=350):
    used = episodes = damage = 0; success = False; final = {}; trace = []
    while used < budget and not success:
        frame = game.reset(); agent.begin(frame); episodes += 1
        for _ in range(min(horizon, budget - used)):
            action, evidence = agent.act(frame); nxt, reward, done, info = game.step(action)
            learned = agent.observe(frame, action, nxt, reward, done); frame = nxt; used += 1; final = info
            if evidence["reason"] in ("test-two-object-composition", "apply-treatment-under-goal-constraints"):
                trace.append({"interaction": used, **evidence, "learned": learned})
            if done:
                damage += int(info["protected_damaged"]); success = bool(info["success"]); break
    return {"success": success, "interactions": used, "episodes": episodes,
            "protected_damaged": damage, "threat_remaining": int(final.get("threat_remaining", -1)),
            "recipe_learned": agent.memory.get("successful_pair") is not None,
            "controls_grounded": len(agent.memory["controls"]) == 4 and agent.memory["interact"] is not None,
            "adapter_geometry": list(agent.adapter.geometry) if agent.adapter.geometry else None,
            "trace": trace[-24:]}


def run_random(game, budget, seed):
    rng = np.random.default_rng(seed); used = episodes = damage = 0; success = False
    while used < budget and not success:
        game.reset(); episodes += 1
        for _ in range(min(350, budget - used)):
            _, _, done, info = game.step(int(rng.integers(5))); used += 1
            if done:
                damage += int(info["protected_damaged"]); success = bool(info["success"]); break
    return {"success": success, "interactions": used, "episodes": episodes, "protected_damaged": damage}


def summary(rows):
    return {"worlds": len(rows), "successes": sum(row["success"] for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "zero_damage_worlds": sum(row["protected_damaged"] == 0 for row in rows),
            "mean_interactions": float(np.mean([row["interactions"] for row in rows]))}


def run_adaptive_transfer_audit(output, seeds=SEALED_SEEDS, budget=4000):
    before = hashes()
    if before != {"adapter": ADAPTER_SHA, "core": CORE_SHA}: raise RuntimeError(f"frozen hash mismatch: {before}")
    mind = AdaptiveUniversalAgent(); learned = []; fresh = []; random_rows = []
    for seed in seeds:
        learned.append(run_episode(mind, DataRecoveryConsole(seed), budget))
        fresh.append(run_episode(AdaptiveUniversalAgent(), DataRecoveryConsole(seed), budget))
        random_rows.append(run_random(DataRecoveryConsole(seed), budget, seed + 101))
    revisits = [run_episode(mind, DataRecoveryConsole(seed), 900) for seed in seeds]
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    brain = output.with_name("ADAPTIVE_V16_DATA_CONSOLE_MIND.json"); mind.save(brain)
    restored = AdaptiveUniversalAgent.load(brain)
    reload_rows = [run_episode(restored, DataRecoveryConsole(seed), 900) for seed in seeds[:6]]
    after = hashes(); initial = summary(learned); revisit = summary(revisits)
    speedup = initial["mean_interactions"] / max(1.0, revisit["mean_interactions"])
    passed = (initial["successes"] >= 10 and initial["zero_damage_worlds"] == len(seeds)
              and summary(random_rows)["successes"] <= 1 and speedup >= 2 and before == after)
    report = {"format": "wailah-adaptive-v16-data-console-audit-v1",
        "classification": "precommitted frozen transfer through a learned layout adapter",
        "frozen_hashes": {"expected": {"adapter": ADAPTER_SHA, "core": CORE_SHA},
                          "before": before, "after": after, "unchanged": before == after},
        "protocol": {"seeds": list(seeds), "budget": budget, "revisit_budget": 900,
                     "raw_observation_shape": [440, 620, 3],
                     "agent_not_given": ["viewport scale", "viewport rectangle", "semantic labels",
                                         "controls", "active pair", "coordinates", "private state"]},
        "aggregate": {"adaptive_growing_mind": initial, "fresh_adaptive_mind": summary(fresh),
                      "random_controls": summary(random_rows), "revisit": revisit,
                      "reloaded_memory": summary(reload_rows), "revisit_speedup": speedup},
        "precommitted_pass": passed, "growth": mind.status(), "per_world": learned,
        "persistent_brain": str(brain)}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8"); return report
