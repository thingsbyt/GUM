"""Development-only rehearsal through the real GUM School lifecycle."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gum.lineage import HashLedger, canonical, file_sha256
from gum.storage import atomic_write_json

from .engine import SchoolEngine
from .training import (
    TrainingLaneConfig,
    initialize_school_learner,
    make_development_rehearsal_evaluator,
    make_rehearsal_trainer,
)


DEFAULT_CURRICULUM = Path(__file__).resolve().parents[2] / "curriculum" / "gum-school-v1.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "gum" / "protocol.py",
        root / "gum" / "school" / "engine.py",
        root / "gum" / "school" / "evaluation.py",
        root / "gum" / "school" / "learner.py",
        root / "gum" / "school" / "rehearsal.py",
        root / "gum" / "school" / "training.py",
        root / "gum" / "school" / "worlds.py",
    ]
    return {
        path.relative_to(root).as_posix(): f"sha256:{file_sha256(path)}"
        for path in paths
    }


def run_nonpromoting_rehearsal(
    workspace: Path,
    *,
    curriculum_path: Path = DEFAULT_CURRICULUM,
    config: TrainingLaneConfig = TrainingLaneConfig(),
    learner_seed: int = 8_620_001,
) -> dict[str, Any]:
    """Train one disposable candidate and prove that development evidence cannot promote it."""
    config.validate()
    workspace = Path(workspace)
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(f"rehearsal workspace is not empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)

    engine = SchoolEngine(workspace, Path(curriculum_path))
    bootstrap = workspace / "bootstrap"
    initialize_school_learner(bootstrap, seed=learner_seed)
    initial_pointer = engine.initialize_promoted(bootstrap)
    lesson = engine.next_lesson()
    if lesson is None:
        raise RuntimeError("the rehearsal curriculum contains no lesson")

    decision = engine.run_lesson(
        make_rehearsal_trainer(config),
        make_development_rehearsal_evaluator(config),
        lesson_id=lesson["lesson_id"],
    )
    if decision["outcome"] != "quarantine":
        raise RuntimeError("development-only rehearsal unexpectedly escaped quarantine")

    final_pointer = engine.snapshots.promoted()
    progress = _read_json(workspace / "PROGRESS.json")
    if canonical(initial_pointer) != canonical(final_pointer):
        raise RuntimeError("rehearsal changed the promoted learner pointer")
    if progress["promoted_lessons"]:
        raise RuntimeError("rehearsal recorded an official curriculum promotion")

    run_id = decision["run_id"]
    run_directory = workspace / "runs" / run_id
    training = _read_json(run_directory / "TRAINING.json")
    evaluation = _read_json(run_directory / "EVALUATION.json")
    failed_gates = sorted(
        gate for gate, result in decision["gates"]["gates"].items()
        if not result["passed"]
    )
    if "sealed-performance" not in failed_gates or "evidence-integrity" not in failed_gates:
        raise RuntimeError("rehearsal did not fail both required non-promotion safeguards")

    artifacts = {}
    for path in sorted(item for item in run_directory.rglob("*") if item.is_file()):
        relative = path.relative_to(workspace).as_posix()
        artifacts[relative] = f"sha256:{file_sha256(path)}"
    ledger = HashLedger(workspace / "SCHOOL_LEDGER.jsonl").verify()
    if not ledger["valid"]:
        raise RuntimeError(f"rehearsal ledger is invalid: {ledger['errors']}")

    report = {
        "format": "gum-school-training-lane-rehearsal-v1",
        "official_curriculum_run": False,
        "candidate_rehearsal_training_performed": True,
        "development_only": True,
        "sealed_data_used": False,
        "scripted_admission_controllers_used_for_training": False,
        "lesson_id": lesson["lesson_id"],
        "run_id": run_id,
        "learner_seed": learner_seed,
        "config": {
            "max_training_interactions": config.max_training_interactions,
            "training_episodes_per_seed": config.training_episodes_per_seed,
            "development_trials": config.development_trials,
            "replay_trial_index": config.replay_trial_index,
        },
        "result": {
            "outcome": decision["outcome"],
            "failed_gates": failed_gates,
            "training_interactions": training["interactions"],
            "development_trials": evaluation["sealed"]["trials"],
            "candidate_development_success_rate": evaluation["sealed"]["success_rate"],
            "fresh_development_success_rate": evaluation["sealed"]["fresh_success_rate"],
            "replay_verified": evaluation["replay"]["verified"],
            "input_boundary_verified": evaluation["input_boundary"]["verified"],
            "promoted_snapshot_unchanged": True,
            "promoted_lessons": progress["promoted_lessons"],
        },
        "interpretation": (
            "This is a development-only systems rehearsal, not a sealed evaluation or a "
            "learning claim. Its candidate was deliberately quarantined."
        ),
        "source_hashes": _source_hashes(),
        "artifact_hashes": artifacts,
        "ledger": ledger,
    }
    atomic_write_json(
        workspace / "REHEARSAL_REPORT.json", report, backup=False, sort_keys=True
    )
    return report
