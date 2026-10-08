"""Post-freeze sealed examination for the first GUM School lesson.

The source and protocol are frozen before seed selection. Training completes
without access to the seed manifest; only then does a system-random evaluator
select the withheld trials, commit to them, and run candidate and controls.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import psutil

from gum.lineage import HashLedger, canonical, file_sha256
from gum.mind import observation_id
from gum.storage import atomic_write_json

from .engine import SchoolEngine
from .learner import CrossSeedSchoolLearner
from .training import (
    LEARNER_FILENAME,
    TrainingLaneConfig,
    _directory_megabytes,
    _paired_advantage_interval,
    _run_episode,
    _summarize,
    _wilson,
    initialize_school_learner,
    make_official_trainer,
)
from .validation import DEFAULT_CURRICULUM
from .worlds import FOUNDATIONAL_LOADERS, create_foundational_world


SHIFTED_MECHANISM = "occlusion-shifted"
SELECTION_PROTOCOL = "post-freeze-independent-v1"
SEED_MINIMUM = 1_000_000_000
SEED_MAXIMUM = 2_000_000_000


class SealedExamError(RuntimeError):
    pass


@dataclass(frozen=True)
class SealedExamConfig:
    training: TrainingLaneConfig = TrainingLaneConfig(
        max_training_interactions=600,
        training_episodes_per_seed=8,
        development_trials=32,
        max_replicas=4,
    )
    trials: int = 32
    replay_trial_index: int = 0

    def validate(self) -> None:
        self.training.validate()
        if isinstance(self.trials, bool) or not isinstance(self.trials, int) or self.trials < 1:
            raise SealedExamError("trials must be a positive integer")
        if not 0 <= self.replay_trial_index < self.trials:
            raise SealedExamError("replay_trial_index is outside the sealed trials")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def frozen_source_hashes(root: Path | None = None) -> dict[str, str]:
    root = _repo_root() if root is None else Path(root)
    paths = [
        root / "curriculum" / "gum-school-v1.json",
        *sorted((root / "curriculum" / "schemas").glob("gum-school-*.schema.json")),
        root / "gum" / "lineage.py",
        root / "gum" / "mind.py",
        root / "gum" / "protocol.py",
        root / "gum" / "storage.py",
        *sorted((root / "gum" / "school").glob("*.py")),
    ]
    return {
        path.relative_to(root).as_posix(): f"sha256:{file_sha256(path)}"
        for path in paths
    }


def _protocol_definition(examination: dict[str, Any], *, trials: int) -> dict[str, Any]:
    return {
        "format": "gum-school-sealed-protocol-v1",
        "examination_id": examination["examination_id"],
        "selection_protocol": SELECTION_PROTOCOL,
        "curriculum_selection_commitment": examination["selection"][
            "sealed_seed_commitment"
        ],
        "seed_selection": "system-random-unique-integers",
        "seed_range": [SEED_MINIMUM, SEED_MAXIMUM],
        "trials": trials,
        "mechanism": SHIFTED_MECHANISM,
        "controls": ["matched-fresh", "random"],
        "candidate_learning_during_evaluation": False,
        "manifest_selected_after_training": True,
        "manifest_revealed_with_completed_evidence": True,
    }


def create_source_freeze(
    path: Path,
    *,
    curriculum: dict[str, Any],
    lesson: dict[str, Any],
    trials: int,
) -> dict[str, Any]:
    examinations = {
        row["examination_id"]: row for row in curriculum["examinations"]
    }
    examination_id = lesson["evaluation"]["examination_ids"][0]
    examination = examinations[examination_id]
    if examination["selection"]["protocol"] != SELECTION_PROTOCOL:
        raise SealedExamError("curriculum does not name the supported sealed protocol")
    protocol = _protocol_definition(examination, trials=trials)
    value = {
        "format": "gum-school-source-freeze-v1",
        "created_at_utc": _utc_now(),
        "git_head": _git_head(_repo_root()),
        "curriculum_id": curriculum["curriculum_id"],
        "lesson_id": lesson["lesson_id"],
        "examination_id": examination_id,
        "source_hashes": frozen_source_hashes(),
        "protocol": protocol,
        "protocol_sha256": f"sha256:{hashlib.sha256(canonical(protocol)).hexdigest()}",
    }
    atomic_write_json(path, value, backup=False, sort_keys=True)
    return value


def _verify_source_freeze(value: dict[str, Any]) -> bool:
    expected_fields = {
        "format", "created_at_utc", "git_head", "curriculum_id", "lesson_id",
        "examination_id", "source_hashes", "protocol", "protocol_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        return False
    if value["format"] != "gum-school-source-freeze-v1":
        return False
    protocol_hash = f"sha256:{hashlib.sha256(canonical(value['protocol'])).hexdigest()}"
    return (
        protocol_hash == value["protocol_sha256"]
        and value["source_hashes"] == frozen_source_hashes()
    )


def select_seed_manifest(
    path: Path,
    *,
    freeze: dict[str, Any],
    lesson: dict[str, Any],
) -> dict[str, Any]:
    """Draw the sealed seeds once, after training, from operating-system entropy."""
    if not _verify_source_freeze(freeze):
        raise SealedExamError("source freeze changed before seed selection")
    if Path(path).exists():
        raise FileExistsError(f"sealed seed manifest already exists: {path}")
    trials = int(freeze["protocol"]["trials"])
    public = {
        int(seed)
        for section in ("training", "evaluation", "retention")
        for seed in (
            lesson[section].get("seeds", [])
            if section == "training"
            else lesson[section].get("development_seeds", [])
            if section == "evaluation"
            else lesson[section].get("evaluation_seeds", [])
        )
    }
    generator = secrets.SystemRandom()
    selected: set[int] = set()
    while len(selected) < trials:
        seed = generator.randrange(SEED_MINIMUM, SEED_MAXIMUM)
        if seed not in public:
            selected.add(seed)
    payload = {
        "format": "gum-school-sealed-seed-manifest-v1",
        "selected_at_utc": _utc_now(),
        "selected_after_source_freeze": True,
        "selected_after_training": True,
        "authority": "independent-evaluator-process",
        "examination_id": freeze["examination_id"],
        "source_freeze_sha256": f"sha256:{hashlib.sha256(canonical(freeze)).hexdigest()}",
        "nonce": secrets.token_hex(32),
        "seeds": sorted(selected),
    }
    manifest = dict(payload)
    manifest["seed_manifest_commitment"] = (
        f"sha256:{hashlib.sha256(canonical(payload)).hexdigest()}"
    )
    atomic_write_json(path, manifest, backup=False, sort_keys=True)
    return manifest


def _verify_seed_manifest(
    manifest: dict[str, Any],
    *,
    freeze: dict[str, Any],
    lesson: dict[str, Any],
) -> bool:
    commitment = manifest.get("seed_manifest_commitment")
    payload = {key: value for key, value in manifest.items() if key != "seed_manifest_commitment"}
    if commitment != f"sha256:{hashlib.sha256(canonical(payload)).hexdigest()}":
        return False
    if payload.get("format") != "gum-school-sealed-seed-manifest-v1":
        return False
    if payload.get("examination_id") != freeze["examination_id"]:
        return False
    if payload.get("source_freeze_sha256") != (
        f"sha256:{hashlib.sha256(canonical(freeze)).hexdigest()}"
    ):
        return False
    seeds = payload.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != freeze["protocol"]["trials"]
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or any(not SEED_MINIMUM <= seed < SEED_MAXIMUM for seed in seeds)
    ):
        return False
    public = set(lesson["training"]["seeds"])
    public.update(lesson["evaluation"]["development_seeds"])
    public.update(lesson["retention"]["evaluation_seeds"])
    return not public.intersection(seeds)


def _run_random_episode(world, *, seed: int, policy_seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(policy_seed)
    observation = world.reset(seed)
    rows = []
    total = 0.0
    unnecessary = 0
    success = False
    for step in range(world.public_spec().horizon):
        action = int(rng.integers(world.public_spec().action_count))
        transition = world.step(action)
        event = transition.public_info.get("event")
        if event in {"blocked", "no-change", "rejected"}:
            unnecessary += 1
        rows.append({
            "step": step,
            "observation": observation_id(observation),
            "action": action,
            "reward": transition.reward,
            "next_observation": observation_id(transition.observation),
            "terminated": transition.terminated,
            "truncated": transition.truncated,
            "public_info": transition.public_info,
        })
        observation = transition.observation
        total += transition.reward
        if transition.terminated or transition.truncated:
            success = bool(transition.public_info.get("success"))
            break
    return {
        "seed": seed,
        "success": success,
        "interactions": len(rows),
        "return": total,
        "uncertain_actions": 0,
        "unnecessary_actions": unnecessary,
        "trajectory": rows,
    }


def make_sealed_evaluator(
    *,
    config: SealedExamConfig,
    freeze: dict[str, Any],
    manifest: dict[str, Any],
):
    config.validate()

    def evaluator(candidate_directory: Path, lesson: dict, evidence_directory: Path) -> dict:
        started = time.perf_counter()
        if not _verify_source_freeze(freeze):
            raise SealedExamError("frozen source changed before evaluation")
        if not _verify_seed_manifest(manifest, freeze=freeze, lesson=lesson):
            raise SealedExamError("sealed seed manifest failed verification")
        candidate_path = Path(candidate_directory) / LEARNER_FILENAME
        candidate = CrossSeedSchoolLearner.load(candidate_path)
        fresh = CrossSeedSchoolLearner(
            candidate.seed,
            max_replicas=candidate.max_replicas,
            initial_replicas=candidate.replica_count,
        )
        worlds_root = Path(evidence_directory) / "sealed-worlds"
        worlds_root.mkdir()
        candidate_rows: list[dict[str, Any]] = []
        fresh_rows: list[dict[str, Any]] = []
        random_rows: list[dict[str, Any]] = []
        replay_package: Path | None = None
        replay_seed: int | None = None
        for index, seed in enumerate(manifest["seeds"]):
            package = create_foundational_world(
                worlds_root, lesson["adapter"], seed=seed, mechanism=SHIFTED_MECHANISM
            )
            candidate_world = FOUNDATIONAL_LOADERS[lesson["adapter"]](package)
            fresh_world = FOUNDATIONAL_LOADERS[lesson["adapter"]](package)
            random_world = FOUNDATIONAL_LOADERS[lesson["adapter"]](package)
            candidate_rows.append(_run_episode(
                candidate, candidate_world, seed=seed, training=False,
                interaction_limit=candidate_world.public_spec().horizon,
            ))
            fresh_rows.append(_run_episode(
                fresh, fresh_world, seed=seed, training=False,
                interaction_limit=fresh_world.public_spec().horizon,
            ))
            random_rows.append(_run_random_episode(
                random_world, seed=seed, policy_seed=seed ^ 0x5EED5EED,
            ))
            if index == config.replay_trial_index:
                replay_package, replay_seed = package, seed

        if replay_package is None or replay_seed is None:
            raise SealedExamError("sealed replay trial was not selected")
        replay_learner = CrossSeedSchoolLearner.load(candidate_path)
        replay_world = FOUNDATIONAL_LOADERS[lesson["adapter"]](replay_package)
        replay_row = _run_episode(
            replay_learner, replay_world, seed=replay_seed, training=False,
            interaction_limit=replay_world.public_spec().horizon,
        )
        expected_replay = candidate_rows[config.replay_trial_index]
        replay_verified = canonical(replay_row) == canonical(expected_replay)
        candidate_summary = _summarize(candidate_rows)
        fresh_summary = _summarize(fresh_rows)
        random_summary = _summarize(random_rows)
        lower, upper = _wilson(candidate_summary["successes"], candidate_summary["trials"])
        advantage_lower, advantage_upper = _paired_advantage_interval(
            [row["success"] for row in candidate_rows],
            [row["success"] for row in fresh_rows],
        )
        evidence = {
            "format": "gum-school-sealed-examination-evidence-v1",
            "official_curriculum_run": True,
            "development_only": False,
            "lesson_id": lesson["lesson_id"],
            "examination_id": freeze["examination_id"],
            "source_freeze": freeze,
            "seed_manifest": manifest,
            "candidate_trials": candidate_rows,
            "matched_fresh_trials": fresh_rows,
            "random_trials": random_rows,
            "candidate_summary": candidate_summary,
            "matched_fresh_summary": fresh_summary,
            "random_summary": random_summary,
            "replay": {
                "trial_index": config.replay_trial_index,
                "expected": expected_replay,
                "replayed": replay_row,
                "deterministic": replay_verified,
            },
        }
        replay_path = Path(evidence_directory) / "sealed-examination.json"
        atomic_write_json(replay_path, evidence, backup=False, sort_keys=True)
        shutil.rmtree(worlds_root)

        training_summary = json.loads(
            (Path(evidence_directory).parent / "TRAINING.json").read_text(encoding="utf-8")
        )
        evaluation_interactions = sum(
            summary["interactions"]
            for summary in (candidate_summary, fresh_summary, random_summary)
        ) + replay_row["interactions"]
        wall = max(0.001, time.perf_counter() - started)
        peak = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        workspace = Path(evidence_directory).parent.parent.parent
        ledger_verified = HashLedger(workspace / "SCHOOL_LEDGER.jsonl").verify()["valid"]
        sources_verified = _verify_source_freeze(freeze)
        protocol_verified = sources_verified and _verify_seed_manifest(
            manifest, freeze=freeze, lesson=lesson
        )
        return {
            "format": "gum-school-evaluation-v1",
            "sealed": {
                "protocol_verified": protocol_verified,
                "trials": candidate_summary["trials"],
                "success_rate": candidate_summary["success_rate"],
                "fresh_success_rate": fresh_summary["success_rate"],
                "uncertainty_rate": candidate_summary["uncertainty_rate"],
                "unnecessary_action_rate": candidate_summary["unnecessary_action_rate"],
                "same_perception": True,
                "same_resource_limits": True,
                "confidence_level": lesson["promotion"]["confidence_level"],
                "success_interval": {"lower": lower, "upper": upper, "method": "wilson"},
                "fresh_advantage_interval": {
                    "lower": advantage_lower,
                    "upper": advantage_upper,
                    "method": "paired-bootstrap",
                },
            },
            "retention": [],
            "evidence": {
                "source_hashes_verified": sources_verified,
                "protocol_hashes_verified": protocol_verified,
                "trajectory_verified": replay_verified,
                "ledger_verified": ledger_verified,
            },
            "resources": {
                "wall_clock_seconds": training_summary["wall_clock_seconds"] + wall,
                "peak_memory_mb": max(training_summary["peak_memory_mb"], peak),
                "artifact_storage_mb": max(
                    training_summary["artifact_storage_mb"],
                    _directory_megabytes(evidence_directory),
                ),
                "interactions": training_summary["interactions"] + evaluation_interactions,
            },
            "input_boundary": {
                "verified": not candidate.boundary_violations and not fresh.boundary_violations,
                "violations": (
                    list(candidate.boundary_violations) + list(fresh.boundary_violations)
                ),
            },
            "replay": {
                "verified": replay_verified,
                "deterministic": replay_verified,
                "artifact_path": "evidence/sealed-examination.json",
                "artifact_sha256": f"sha256:{file_sha256(replay_path)}",
            },
            "transfer_results": [],
        }

    return evaluator


def run_official_sealed_exam(
    workspace: Path,
    *,
    curriculum_path: Path = DEFAULT_CURRICULUM,
    config: SealedExamConfig = SealedExamConfig(),
    learner_seed: int = 8_620_001,
) -> dict[str, Any]:
    """Run one official lesson, selecting sealed seeds only after training."""
    config.validate()
    workspace = Path(workspace)
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(f"sealed exam workspace is not empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    engine = SchoolEngine(workspace, Path(curriculum_path))
    bootstrap = workspace / "bootstrap"
    initialize_school_learner(
        bootstrap, seed=learner_seed, max_replicas=config.training.max_replicas
    )
    initial_pointer = engine.initialize_promoted(bootstrap)
    lesson = engine.next_lesson()
    if lesson is None:
        raise SealedExamError("curriculum contains no lesson to examine")
    started = engine.begin_lesson(lesson["lesson_id"])
    run_id = started["run_id"]
    run_directory = workspace / "runs" / run_id
    freeze = create_source_freeze(
        run_directory / "SOURCE_FREEZE.json",
        curriculum=engine.curriculum,
        lesson=lesson,
        trials=config.trials,
    )
    try:
        trainer = make_official_trainer(config.training)
        summary = trainer(started["candidate_directory"], lesson, run_directory / "evidence")
        engine.mark_training_complete(run_id, summary)
        manifest = select_seed_manifest(
            run_directory / "SEALED_SEED_MANIFEST.json", freeze=freeze, lesson=lesson
        )
        evaluator = make_sealed_evaluator(
            config=config, freeze=freeze, manifest=manifest
        )
        evaluation = evaluator(
            engine.candidate_snapshot_directory(run_id), lesson, run_directory / "evidence"
        )
        engine.record_evaluation(run_id, evaluation)
        decision = engine.finalize(run_id)
    except Exception as error:
        state = engine.status()["active"]
        if state["status"] == "active":
            engine.abort(run_id, f"{type(error).__name__}: {error}")
        raise

    evaluation = json.loads((run_directory / "EVALUATION.json").read_text(encoding="utf-8"))
    sealed_evidence = json.loads(
        (run_directory / "evidence" / "sealed-examination.json").read_text(encoding="utf-8")
    )
    final_pointer = engine.snapshots.promoted()
    artifacts = {
        path.relative_to(workspace).as_posix(): f"sha256:{file_sha256(path)}"
        for path in sorted(item for item in run_directory.rglob("*") if item.is_file())
    }
    report = {
        "format": "gum-school-sealed-exam-report-v1",
        "official_curriculum_run": True,
        "development_only": False,
        "sealed_data_used_for_training": False,
        "lesson_id": lesson["lesson_id"],
        "examination_id": freeze["examination_id"],
        "run_id": run_id,
        "source_freeze_git_head": freeze["git_head"],
        "seed_manifest_commitment": manifest["seed_manifest_commitment"],
        "seed_manifest_revealed": True,
        "result": {
            "outcome": decision["outcome"],
            "all_gates_passed": decision["gates"]["all_passed"],
            "failed_gates": [
                gate for gate, row in decision["gates"]["gates"].items()
                if not row["passed"]
            ],
            "candidate_success_rate": evaluation["sealed"]["success_rate"],
            "matched_fresh_success_rate": evaluation["sealed"]["fresh_success_rate"],
            "random_success_rate": sealed_evidence["random_summary"]["success_rate"],
            "success_interval": evaluation["sealed"]["success_interval"],
            "training_interactions": summary["interactions"],
            "total_interactions": evaluation["resources"]["interactions"],
            "replay_verified": evaluation["replay"]["verified"],
            "input_boundary_verified": evaluation["input_boundary"]["verified"],
            "source_and_protocol_verified": (
                evaluation["evidence"]["source_hashes_verified"]
                and evaluation["evidence"]["protocol_hashes_verified"]
            ),
            "promoted_snapshot_changed": final_pointer != initial_pointer,
            "promoted_snapshot_id": final_pointer["snapshot_id"],
        },
        "gates": decision["gates"],
        "artifact_hashes": artifacts,
        "ledger": HashLedger(workspace / "SCHOOL_LEDGER.jsonl").verify(),
    }
    atomic_write_json(workspace / "SEALED_EXAM_REPORT.json", report, backup=False, sort_keys=True)
    return report
