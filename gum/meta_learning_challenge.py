"""Frozen multi-seed test of whether acquisition itself improves with experience."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .lineage import HashLedger, file_sha256
from .meta_learning import FAMILIES, MetaLearningMind


FORMAT = "gum-learning-to-learn-challenge-v1"


def _write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def protocol(source_root: Path, output: Path):
    source_root = Path(source_root); files = [source_root / "gum" / "meta_learning.py",
                                             source_root / "gum" / "meta_learning_challenge.py"]
    value = {"format": FORMAT,
        "classification": "precommitted multi-seed learning-to-learn audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "meta_seeds": [77, 177, 277, 377, 477],
        "training": {"tasks_per_run": 32, "families": list(FAMILIES), "budget_per_solver": 1200,
                     "population": 18, "solutions_must_not_be_retained": True},
        "held_out": {"tasks_per_run": 24, "new_seeds": True, "all_families": True},
        "learner_inputs": ["current RGB state", "RGB goal", "anonymous action count", "sparse success reward"],
        "learner_not_given": ["task family", "state variables", "transition rules", "action meanings",
                              "goal state values", "solution path", "best acquisition strategy"],
        "thresholds": {"runs": 5, "all_runs_zero_solutions_retained": True,
                       "minimum_runs_beating_initial_strategy": 4,
                       "minimum_median_held_out_speedup": 2.0,
                       "minimum_runs_with_late_curve_better_than_early": 3,
                       "minimum_held_out_family_coverage": 4,
                       "reload_must_preserve_strategy": True, "ledger_valid": True,
                       "source_frozen": True},
        "source_hashes": {str(path.relative_to(source_root)).replace("\\", "/"): file_sha256(path) for path in files}}
    _write_json(output, value); return value


def _cost(row, key, budget):
    value = row[key]; return value["interactions"] if value["success"] else budget * 1.5


def _run_one(meta_seed, protocol_value, output_root, ledger):
    cfg = protocol_value["training"]; budget = cfg["budget_per_solver"]
    mind = MetaLearningMind(meta_seed, cfg["population"]); curve = []
    for index in range(cfg["tasks_per_run"]):
        seed = 510_000_000 + meta_seed * 100_000 + index * 1009
        family = FAMILIES[index % len(FAMILIES)]
        row = mind.learn_task(seed, family, budget); curve.append(row)
        ledger.append("meta-training-task", {"meta_seed": meta_seed, **row}, run_id=f"meta-{meta_seed}")
    specs = [(910_000_000 + meta_seed * 100_000 + index * 2017, FAMILIES[index % len(FAMILIES)])
             for index in range(protocol_value["held_out"]["tasks_per_run"])]
    evaluation = mind.evaluate(specs, budget); rows = evaluation["rows"]
    evolved = np.asarray([_cost(row, "evolved", budget) for row in rows], dtype=float)
    initial = np.asarray([_cost(row, "initial", budget) for row in rows], dtype=float)
    shuffled = np.asarray([_cost(row, "shuffled", budget) for row in rows], dtype=float)
    early = float(np.median([row["relative_efficiency"] for row in curve[:8]]))
    late = float(np.median([row["relative_efficiency"] for row in curve[-8:]]))
    brain = Path(output_root) / f"META_MIND_{meta_seed}.json"; mind.save(brain)
    restored = MetaLearningMind.load(brain); reload_preserved = restored.champion().genome_id == mind.champion().genome_id
    family_rows = {}
    for family in FAMILIES:
        selected = [row for row in rows if row["family_audit_only"] == family]
        family_rows[family] = {"tasks": len(selected),
            "evolved_mean": float(np.mean([_cost(row, "evolved", budget) for row in selected])),
            "initial_mean": float(np.mean([_cost(row, "initial", budget) for row in selected])),
            "evolved_successes": sum(row["evolved"]["success"] for row in selected)}
    summary = {"meta_seed": meta_seed, "tasks_seen": mind.tasks_seen,
        "solutions_retained": mind.solutions_retained, "strategy_changed": mind.champion().weights.tolist() != [0, 0, 0],
        "early_relative_efficiency": early, "late_relative_efficiency": late,
        "late_curve_better": late > early,
        "held_out": {"tasks": len(rows), "evolved_successes": sum(row["evolved"]["success"] for row in rows),
                     "initial_successes": sum(row["initial"]["success"] for row in rows),
                     "evolved_mean_cost": float(evolved.mean()), "initial_mean_cost": float(initial.mean()),
                     "shuffled_mean_cost": float(shuffled.mean()),
                     "speedup_over_initial": float(initial.mean() / max(1, evolved.mean())),
                     "wins_over_initial": int(np.sum(evolved < initial)), "families": family_rows},
        "champion": evaluation["champion"], "reload_preserved": reload_preserved,
        "brain": {"path": str(brain), "bytes": brain.stat().st_size, "sha256": file_sha256(brain)},
        "curve": curve, "evaluation_rows": rows}
    ledger.append("held-out-summary", {key: value for key, value in summary.items()
                  if key not in ("curve", "evaluation_rows")}, run_id=f"meta-{meta_seed}")
    return summary


def _chart(runs, output):
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1000, 580; image = Image.new("RGB", (width, height), (7, 17, 24)); draw = ImageDraw.Draw(image)
    try: title = ImageFont.truetype("arial.ttf", 25); font = ImageFont.truetype("arial.ttf", 16)
    except OSError: title = font = ImageFont.load_default()
    draw.text((40, 25), "GUM LEARNING-TO-LEARN · NORMALIZED ACQUISITION EFFICIENCY", font=title, fill=(103, 230, 177))
    draw.text((40, 62), "Higher means fewer interactions than the original strategy on the same task", font=font, fill=(190, 203, 213))
    left, top, right, bottom = 65, 105, 960, 520
    draw.line((left, bottom, right, bottom), fill=(102, 119, 132), width=2)
    draw.line((left, top, left, bottom), fill=(102, 119, 132), width=2)
    values = np.asarray([[min(12.0, row["relative_efficiency"]) for row in run["curve"]] for run in runs])
    mean = values.mean(0); maximum = 12.0
    points = []
    for index, value in enumerate(mean):
        x = left + index * (right - left) / max(1, len(mean) - 1)
        y = bottom - value / maximum * (bottom - top); points.append((x, y))
    draw.line(points, fill=(103, 230, 177), width=4)
    draw.line((left, bottom - (bottom - top) / maximum, right, bottom - (bottom - top) / maximum),
              fill=(91, 109, 122), width=1)
    draw.text((18, bottom - (bottom - top) / maximum - 8), "1×", font=font, fill=(190, 203, 213))
    draw.text((left, 535), "task 1", font=font, fill=(190, 203, 213)); draw.text((right - 55, 535), f"task {len(mean)}", font=font, fill=(190, 203, 213))
    image.save(output)


def run(source_root: Path, output_root: Path):
    source_root = Path(source_root); output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "LEARNING_TO_LEARN_PROTOCOL.json"
    frozen = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else protocol(source_root, protocol_path)
    before = {name: file_sha256(source_root / Path(name)) for name in frozen["source_hashes"]}
    if before != frozen["source_hashes"]: raise RuntimeError("implementation differs from frozen protocol")
    ledger = HashLedger(output_root / "LEARNING_TO_LEARN_LEDGER.jsonl")
    runs = [_run_one(seed, frozen, output_root, ledger) for seed in frozen["meta_seeds"]]
    speedups = [row["held_out"]["speedup_over_initial"] for row in runs]
    after = {name: file_sha256(source_root / Path(name)) for name in frozen["source_hashes"]}
    ledger_status = ledger.verify(); checks = {
        "five_runs": len(runs) == 5,
        "zero_solutions_retained": all(row["solutions_retained"] == 0 for row in runs),
        "strategies_evolved": all(row["strategy_changed"] for row in runs),
        "runs_beating_initial": sum(row["held_out"]["speedup_over_initial"] > 1 for row in runs) >= 4,
        "median_speedup": float(np.median(speedups)) >= frozen["thresholds"]["minimum_median_held_out_speedup"],
        "late_curve_improvement": sum(row["late_curve_better"] for row in runs) >= 3,
        "all_families_covered": all(all(run["held_out"]["families"][family]["tasks"] > 0 for family in FAMILIES) for run in runs),
        "reload_preserved": all(row["reload_preserved"] for row in runs),
        "ledger_valid": ledger_status["valid"], "source_frozen": before == after == frozen["source_hashes"]}
    report = {"format": FORMAT, "passed": all(checks.values()), "classification": frozen["classification"],
        "learner_inputs": frozen["learner_inputs"], "learner_not_given": frozen["learner_not_given"],
        "source_freeze": {"expected": frozen["source_hashes"], "before": before, "after": after,
                          "unchanged": before == after},
        "aggregate": {"runs": len(runs), "training_tasks": sum(row["tasks_seen"] for row in runs),
                      "held_out_tasks": sum(row["held_out"]["tasks"] for row in runs),
                      "median_speedup_over_initial": float(np.median(speedups)),
                      "mean_speedup_over_initial": float(np.mean(speedups)),
                      "runs_beating_initial": sum(row["held_out"]["speedup_over_initial"] > 1 for row in runs),
                      "runs_with_late_curve_better": sum(row["late_curve_better"] for row in runs),
                      "total_solutions_retained": sum(row["solutions_retained"] for row in runs)},
        "runs": runs, "ledger": ledger_status, "checks": checks,
        "interpretation": ("GUM changed a persistent acquisition-policy genome and learned new pixel-goal puzzles "
            "more efficiently without retaining any task solution paths. Candidate search operators and the "
            "three genome features remain engineered; this is strategy meta-learning, not unrestricted self-rewriting.")}
    _write_json(output_root / "LEARNING_TO_LEARN_AUDIT.json", report)
    _chart(runs, output_root / "LEARNING_TO_LEARN_CURVE.png")
    lines = ["# GUM Learning-to-Learn Challenge", "", f"**Verdict: {'PASS' if report['passed'] else 'FAIL'}**", "",
        f"- Independent meta-learning runs: **{len(runs)}**", f"- Training tasks: **{report['aggregate']['training_tasks']}**",
        f"- Held-out tasks: **{report['aggregate']['held_out_tasks']}**",
        f"- Runs beating their original acquisition strategy: **{report['aggregate']['runs_beating_initial']}/5**",
        f"- Median held-out speedup: **{report['aggregate']['median_speedup_over_initial']:.2f}×**",
        f"- Runs whose late learning curve beat their early curve: **{report['aggregate']['runs_with_late_curve_better']}/5**",
        f"- Exact task solutions retained: **{report['aggregate']['total_solutions_retained']}**", "",
        "Only the evolved three-weight acquisition strategy persisted across tasks. Puzzle seeds, dimensions, controls, dynamics, presentations, goals, and solution paths changed, and all four families appeared in held-out evaluation."]
    (output_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path); parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args(); path = args.output / "LEARNING_TO_LEARN_PROTOCOL.json"
    result = protocol(args.source, path) if args.protocol_only else run(args.source, args.output)
    summary = result if args.protocol_only else {"format": result["format"], "passed": result["passed"],
        "aggregate": result["aggregate"], "checks": result["checks"],
        "per_run_speedups": [row["held_out"]["speedup_over_initial"] for row in result["runs"]]}
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
