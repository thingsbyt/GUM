"""Untouched second gate: repaired single mind versus two-mind collective."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from jepa_asteroids.crystal_garden import CrystalGardenGame
from jepa_asteroids.data_recovery_console import DataRecoveryConsole

from .experience_collective import ExperienceCollective
from .lineage import HashLedger, TransitionTrace, file_sha256
from .resilient_generalist import ResilientGeneralist
from .reveal_gate import _run, _summary


FROZEN = {
    "gum/resilient_generalist.py": "2DC018E81F21D9BB79CF6C1F7C43412B2CE5F6156D4888F8BA46EA1CF07DD562",
    "gum/experience_collective.py": "9050CA3E25AC5CEF40376954C9541EC56238C26584B90559D002CC5CE4F888AE",
    "jepa_asteroids/adaptive_visual_agent.py": "EB97E5927255A249899C95422C898A015E7CE5938C070C82D42956DCC55DE31A",
    "jepa_asteroids/universal_goal_state.py": "F20B968908C1CB3AB40CECE3B9157F5E4A838538A37350CA07C5E8A1E8118586",
}
CRYSTAL_SEEDS = tuple(9_101_009 + index * 5_003 for index in range(10))
CONSOLE_SEEDS = tuple(9_201_011 + index * 5_009 for index in range(10))
RELOAD_INDICES = (0, 3, 6, 9, 10, 13, 16, 19)


def implementation_hashes():
    root = Path(__file__).parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest().upper() for name in FROZEN}


def worlds():
    rows = []
    for index in range(10):
        crystal = CRYSTAL_SEEDS[index]; console = CONSOLE_SEEDS[index]
        rows.append(("crystal-garden", crystal, lambda seed=crystal: CrystalGardenGame(seed), 300))
        rows.append(("data-recovery-console", console, lambda seed=console: DataRecoveryConsole(seed), 350))
    return rows


def _tag(result, family, seed, **extra): return {"family": family, "seed": seed, **extra, **result}


def run_dual_gate(output_dir: Path, *, budget=4000, revisit_budget=900, ledger_path: Path | None = None):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    before = implementation_hashes()
    if before != FROZEN: raise RuntimeError(f"frozen implementation mismatch: {before}")
    evaluator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    single_trace = TransitionTrace(output_dir / "SINGLE_TRANSITIONS.jsonL")
    collective_trace = TransitionTrace(output_dir / "COLLECTIVE_TRANSITIONS.jsonL")
    ledger = HashLedger(ledger_path) if ledger_path else None
    run_id = f"dual-gate-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if ledger: ledger.append("dual-reveal-gate-started", {"frozen": before,
        "evaluator_sha256": evaluator_hash, "worlds": 20, "members": [1, 2]}, run_id=run_id)

    rows = worlds(); single = ResilientGeneralist(); single_rows = []
    for family, seed, factory, horizon in rows:
        single_rows.append(_tag(_run(single, factory, budget=budget, horizon=horizon, seed=seed,
            trace=single_trace, condition="single-first"), family, seed))
    single_revisit = []
    for family, seed, factory, horizon in rows:
        single_revisit.append(_tag(_run(single, factory, budget=revisit_budget, horizon=horizon, seed=seed,
            trace=single_trace, condition="single-revisit"), family, seed))
    single_path = output_dir / "SINGLE_MIND.json"; single.save(single_path)
    single_loaded = ResilientGeneralist.load(single_path); single_reload = []
    for index in RELOAD_INDICES:
        family, seed, factory, horizon = rows[index]
        single_reload.append(_tag(_run(single_loaded, factory, budget=revisit_budget, horizon=horizon,
            seed=seed, condition="single-reloaded"), family, seed))

    collective = ExperienceCollective(); collective_rows = []
    for index, (family, seed, factory, horizon) in enumerate(rows):
        member = index % 2
        result = _run(collective.members[member], factory, budget=budget, horizon=horizon, seed=seed,
                      trace=collective_trace, condition=f"collective-member-{member}-first")
        collective_rows.append(_tag(result, family, seed, member=member)); collective.publish(member)
    cross_revisit = []
    for index, (family, seed, factory, horizon) in enumerate(rows):
        original = index % 2; member = 1 - original
        result = _run(collective.members[member], factory, budget=revisit_budget, horizon=horizon, seed=seed,
                      trace=collective_trace, condition=f"cross-owner-member-{member}-revisit")
        cross_revisit.append(_tag(result, family, seed, original_member=original, evaluating_member=member))
    collective_path = output_dir / "COLLECTIVE_MIND"; collective.save(collective_path)
    collective_loaded = ExperienceCollective.load(collective_path); collective_reload = []
    for index in RELOAD_INDICES:
        family, seed, factory, horizon = rows[index]; member = 1 - (index % 2)
        result = _run(collective_loaded.members[member], factory, budget=revisit_budget, horizon=horizon,
                      seed=seed, condition=f"collective-reloaded-member-{member}")
        collective_reload.append(_tag(result, family, seed, member=member))

    fresh_rows = []; random_rows = []
    for family, seed, factory, horizon in rows:
        fresh_rows.append(_tag(_run(ResilientGeneralist(), factory, budget=budget, horizon=horizon,
            seed=seed, condition="fresh"), family, seed))
        random_rows.append(_tag(_run(None, factory, budget=budget, horizon=horizon,
            seed=seed, condition="random"), family, seed))

    s_first, s_revisit, s_reload = _summary(single_rows), _summary(single_revisit), _summary(single_reload)
    c_first, c_cross, c_reload = _summary(collective_rows), _summary(cross_revisit), _summary(collective_reload)
    fresh, random = _summary(fresh_rows), _summary(random_rows)
    single_speedup = s_first["mean_interactions"] / max(1.0, s_revisit["mean_interactions"])
    cross_owner_speedup = fresh["mean_interactions"] / max(1.0, c_cross["mean_interactions"])
    single_total = sum(row["interactions"] for row in single_rows)
    collective_parallel_rounds = sum(max(collective_rows[i]["interactions"], collective_rows[i + 1]["interactions"])
                                     for i in range(0, len(collective_rows), 2))
    parallel_speedup = single_total / max(1, collective_parallel_rounds)
    single_pass = bool(s_first["successes"] >= 19 and s_first["zero_damage_worlds"] == 20
        and s_revisit["successes"] >= 19 and single_speedup >= 2 and s_reload["successes"] == 8
        and s_first["mean_interactions"] <= fresh["mean_interactions"] * 1.5 and random["successes"] <= 2)
    collective_pass = bool(c_first["successes"] >= 19 and c_first["zero_damage_worlds"] == 20
        and c_cross["successes"] >= 19 and c_reload["successes"] == 8
        and cross_owner_speedup >= 2 and parallel_speedup >= 1.5 and collective.status()["exchanges"] == 20)
    after = implementation_hashes(); frozen = before == after == FROZEN
    report = {"format": "gum-dual-reveal-gate-v1", "run_id": run_id,
        "classification": "untouched second gate for a repaired single mind and two-mind experience collective",
        "frozen_implementation": {"expected": FROZEN, "before": before, "after": after, "unchanged": frozen},
        "evaluator_sha256": evaluator_hash,
        "protocol": {"worlds": 20, "families": ["crystal-garden", "data-recovery-console"],
            "seeds": {"crystal-garden": list(CRYSTAL_SEEDS), "data-recovery-console": list(CONSOLE_SEEDS)},
            "learner_inputs": ["RGB pixels", "five anonymous actions", "scalar reward", "termination"],
            "learner_not_given": ["game name", "rules", "semantic labels", "control meanings",
                "object coordinates", "recipe", "private simulator state"],
            "collective_exchange": ["learned per-context memory", "learned visual-form role counts"],
            "collective_not_exchanged": ["simulator objects", "hidden controls", "hidden recipe", "audit state"]},
        "thresholds": {"minimum_first_successes": 19, "required_zero_damage": 20,
            "minimum_revisit_successes": 19, "minimum_revisit_speedup": 2.0,
            "required_reload_successes_of_8": 8, "single_maximum_overhead_versus_fresh": 1.5,
            "maximum_random_successes": 2, "collective_minimum_cross_owner_speedup": 2.0,
            "collective_minimum_parallel_speedup": 1.5},
        "aggregate": {"single_first": s_first, "single_revisit": s_revisit, "single_reloaded": s_reload,
            "single_revisit_speedup": single_speedup, "collective_first": c_first,
            "collective_cross_owner_revisit": c_cross, "collective_reloaded": c_reload,
            "collective_cross_owner_speedup_over_fresh": cross_owner_speedup,
            "collective_parallel_acquisition_speedup_over_single": parallel_speedup,
            "fresh_per_world": fresh, "random": random},
        "verdict": {"single_repair_pass": bool(single_pass and frozen),
            "two_mind_collective_pass": bool(collective_pass and frozen),
            "both_pass": bool(single_pass and collective_pass and frozen)},
        "single_status": single.status(), "collective_status": collective.status(),
        "single_worlds": single_rows, "single_revisits": single_revisit, "single_reloads": single_reload,
        "collective_worlds": collective_rows, "cross_owner_revisits": cross_revisit,
        "collective_reloads": collective_reload, "fresh_worlds": fresh_rows, "random_worlds": random_rows,
        "artifacts": {"single_mind": str(single_path), "single_mind_sha256": file_sha256(single_path),
            "collective_mind": str(collective_path),
            "single_trace": {"path": str(single_trace.path), **single_trace.verify()},
            "collective_trace": {"path": str(collective_trace.path), **collective_trace.verify()}}}
    report_path = output_dir / "DUAL_REVEAL_GATE_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if ledger: ledger.append("dual-reveal-gate-completed", {"report": str(report_path),
        "report_sha256": file_sha256(report_path), "verdict": report["verdict"],
        "aggregate": report["aggregate"], "traces": {"single": report["artifacts"]["single_trace"],
        "collective": report["artifacts"]["collective_trace"]}}, run_id=run_id)
    return report
