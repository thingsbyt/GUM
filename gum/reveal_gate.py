"""Sealed 20-world reveal gate for one byte-frozen visual learning agent.

The evaluator and its worlds are outside the learner.  The learner receives
only RGB pixels, five anonymous actions, scalar reward, and termination.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from jepa_asteroids.adaptive_visual_agent import AdaptiveUniversalAgent
from jepa_asteroids.crystal_garden import CrystalGardenGame
from jepa_asteroids.data_recovery_console import DataRecoveryConsole

from .lineage import HashLedger, TransitionTrace, file_sha256
from .mind import observation_id


FROZEN = {
    "adaptive_visual_agent.py": "EB97E5927255A249899C95422C898A015E7CE5938C070C82D42956DCC55DE31A",
    "universal_goal_state.py": "F20B968908C1CB3AB40CECE3B9157F5E4A838538A37350CA07C5E8A1E8118586",
}
CRYSTAL_SEEDS = tuple(8_101_001 + index * 4_099 for index in range(10))
CONSOLE_SEEDS = tuple(8_201_003 + index * 4_127 for index in range(10))


def implementation_hashes():
    root = Path(__file__).parents[1] / "jepa_asteroids"
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest().upper() for name in FROZEN}


def _worlds():
    rows = []
    for index in range(10):
        rows.append(("crystal-garden", CRYSTAL_SEEDS[index], lambda seed=CRYSTAL_SEEDS[index]: CrystalGardenGame(seed), 300))
        rows.append(("data-recovery-console", CONSOLE_SEEDS[index], lambda seed=CONSOLE_SEEDS[index]: DataRecoveryConsole(seed), 350))
    return rows


def _run(agent, factory, *, budget, horizon, seed, trace=None, condition="lifelong"):
    used = episodes = damage = 0; success = False; final = {}; rng = np.random.default_rng(seed + 991)
    while used < budget and not success:
        game = factory(); frame = game.reset(); episodes += 1
        if agent is not None: agent.begin(frame)
        for _ in range(min(horizon, budget - used)):
            before = observation_id(frame)
            if agent is None:
                action = int(rng.integers(5)); evidence = {"reason": "random"}
            else:
                action, evidence = agent.act(frame)
            nxt, reward, done, info = game.step(action)
            learned = {} if agent is None else agent.observe(frame, action, nxt, reward, done)
            used += 1
            if trace is not None:
                safe = {key: info[key] for key in ("success", "protected_damaged", "threat_remaining") if key in info}
                trace.append("transition", {"condition": condition, "seed": int(seed), "interaction": used,
                    "observation": before, "action": int(action), "reward": float(reward),
                    "next_observation": observation_id(nxt), "terminated": bool(done),
                    "public_info": safe, "reason": evidence.get("reason")})
            frame = nxt; final = info
            if done:
                damage += int(info.get("protected_damaged", info.get("healthy_damaged", 0)))
                success = bool(info.get("success")); break
    return {"success": success, "interactions": used, "episodes": episodes,
        "protected_damaged": damage, "threat_remaining": int(final.get("threat_remaining", -1))}


def _summary(rows):
    return {"worlds": len(rows), "successes": sum(row["success"] for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "zero_damage_worlds": sum(row["protected_damaged"] == 0 for row in rows),
        "mean_interactions": float(np.mean([row["interactions"] for row in rows])),
        "median_interactions": float(np.median([row["interactions"] for row in rows]))}


def run_reveal_gate(output_dir: Path, *, budget=4000, revisit_budget=900, ledger_path: Path | None = None):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    before = implementation_hashes()
    if before != FROZEN: raise RuntimeError(f"frozen implementation mismatch: {before}")
    evaluator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    trace = TransitionTrace(output_dir / "LIFELONG_TRANSITIONS.jsonl")
    ledger = HashLedger(ledger_path) if ledger_path else None
    run_id = f"reveal-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if ledger: ledger.append("reveal-gate-started", {"frozen": before, "evaluator_sha256": evaluator_hash,
        "worlds": 20, "inputs": ["RGB pixels", "five anonymous actions", "reward", "termination"]}, run_id=run_id)

    mind = AdaptiveUniversalAgent(); lifelong = []; fresh = []; random_rows = []
    worlds = _worlds()
    for family, seed, factory, horizon in worlds:
        result = _run(mind, factory, budget=budget, horizon=horizon, seed=seed, trace=trace)
        lifelong.append({"family": family, "seed": seed, **result})
        fresh_result = _run(AdaptiveUniversalAgent(), factory, budget=budget, horizon=horizon, seed=seed,
                            condition="fresh")
        fresh.append({"family": family, "seed": seed, **fresh_result})
        random_result = _run(None, factory, budget=budget, horizon=horizon, seed=seed, condition="random")
        random_rows.append({"family": family, "seed": seed, **random_result})

    revisits = []
    for family, seed, factory, horizon in worlds:
        result = _run(mind, factory, budget=revisit_budget, horizon=horizon, seed=seed,
                      trace=trace, condition="revisit")
        revisits.append({"family": family, "seed": seed, **result})

    brain = output_dir / "FROZEN_GUM_MIND.json"; mind.save(brain)
    restored = AdaptiveUniversalAgent.load(brain); reload_rows = []
    for family, seed, factory, horizon in worlds[::3][:8]:
        result = _run(restored, factory, budget=revisit_budget, horizon=horizon, seed=seed,
                      condition="reloaded")
        reload_rows.append({"family": family, "seed": seed, **result})

    after = implementation_hashes(); initial = _summary(lifelong); cold = _summary(fresh)
    revisit = _summary(revisits); random_summary = _summary(random_rows); reload_summary = _summary(reload_rows)
    speedup = initial["mean_interactions"] / max(1.0, revisit["mean_interactions"])
    transfer_advantage = 1.0 - initial["mean_interactions"] / max(1.0, cold["mean_interactions"])
    passed = bool(initial["successes"] >= 18 and initial["zero_damage_worlds"] == 20
                  and random_summary["successes"] <= 2 and revisit["successes"] >= 18
                  and speedup >= 2.0 and reload_summary["successes"] >= 7
                  and transfer_advantage >= 0.10 and before == after)
    report = {"format": "gum-reveal-gate-v1",
        "classification": "sealed 20-world, two-interface test of one byte-frozen growing visual mind",
        "run_id": run_id, "frozen_implementation": {"expected": FROZEN, "before": before,
            "after": after, "unchanged": before == after},
        "evaluator_sha256": evaluator_hash,
        "protocol": {"worlds": 20, "families": ["crystal-garden", "data-recovery-console"],
            "seeds": {"crystal-garden": list(CRYSTAL_SEEDS), "data-recovery-console": list(CONSOLE_SEEDS)},
            "budget": budget, "revisit_budget": revisit_budget,
            "learner_inputs": ["RGB pixels", "five anonymous actions", "scalar reward", "termination"],
            "learner_not_given": ["game name", "rules", "semantic labels", "control meanings",
                "object coordinates", "recipe", "private simulator state"]},
        "precommitted_thresholds": {"lifelong_successes": 18, "zero_damage_worlds": 20,
            "maximum_random_successes": 2, "revisit_successes": 18, "minimum_revisit_speedup": 2.0,
            "minimum_reloaded_successes_of_8": 7, "minimum_transfer_advantage": 0.10,
            "frozen_source_unchanged": True},
        "aggregate": {"lifelong": initial, "fresh_per_world": cold, "random": random_summary,
            "revisit": revisit, "reloaded": reload_summary,
            "revisit_speedup": speedup, "transfer_advantage_fraction": transfer_advantage},
        "precommitted_pass": passed, "mind_growth": mind.status(), "lifelong_worlds": lifelong,
        "fresh_worlds": fresh, "random_worlds": random_rows, "revisit_worlds": revisits,
        "reloaded_worlds": reload_rows, "persistent_brain": {"path": str(brain), "sha256": file_sha256(brain)},
        "transition_trace": {"path": str(trace.path), **trace.verify()}}
    report_path = output_dir / "REVEAL_GATE_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if ledger: ledger.append("reveal-gate-completed", {"report": str(report_path),
        "report_sha256": file_sha256(report_path), "pass": passed, "aggregate": report["aggregate"],
        "trace": report["transition_trace"]}, run_id=run_id)
    return report
