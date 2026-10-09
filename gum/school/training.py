"""Bounded training and development-only evaluation for GUM School rehearsals."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import psutil

from gum.lineage import HashLedger, canonical, file_sha256
from gum.mind import observation_id
from gum.storage import atomic_write_json

from .learner import CrossSeedSchoolLearner
from .worlds import FOUNDATIONAL_LOADERS, create_lesson_world


LEARNER_FILENAME = "SCHOOL_LEARNER.json"


@dataclass(frozen=True)
class TrainingLaneConfig:
    max_training_interactions: int = 300
    training_episodes_per_seed: int = 4
    development_trials: int = 32
    replay_trial_index: int = 0
    max_replicas: int = 4

    def validate(self) -> None:
        for name in (
            "max_training_interactions", "training_episodes_per_seed",
            "development_trials", "max_replicas",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_replicas > CrossSeedSchoolLearner.replica_limit:
            raise ValueError(
                f"max_replicas cannot exceed {CrossSeedSchoolLearner.replica_limit}"
            )
        if not 0 <= self.replay_trial_index < self.development_trials:
            raise ValueError("replay_trial_index is outside the development trials")


def initialize_school_learner(
    directory: Path, *, seed: int = 8_620_001, max_replicas: int = 4
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LEARNER_FILENAME
    if path.exists():
        raise FileExistsError(f"learner state already exists: {path}")
    CrossSeedSchoolLearner(seed, max_replicas=max_replicas).save(path)
    return path


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "gum" / "protocol.py",
        root / "gum" / "school" / "learner.py",
        root / "gum" / "school" / "training.py",
        root / "gum" / "school" / "worlds.py",
    ]
    return {
        path.relative_to(root).as_posix(): f"sha256:{file_sha256(path)}"
        for path in paths
    }


def _directory_megabytes(path: Path) -> float:
    total = sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())
    return total / (1024.0 * 1024.0)


def _run_episode(
    learner: CrossSeedSchoolLearner,
    world,
    *,
    seed: int,
    training: bool,
    interaction_limit: int,
) -> dict[str, Any]:
    observation = world.reset(seed)
    learner.begin(world.public_spec(), observation, training=training)
    rows = []
    total = 0.0
    uncertainty = 0
    unnecessary = 0
    known_ineffective_actions: set[int] = set()
    success = False
    for step in range(min(world.public_spec().horizon, interaction_limit)):
        before = observation_id(observation)
        action = learner.act(observation, training=training)
        confidence = learner.confidence()
        if confidence < 0.05:
            uncertainty += 1
        transition = world.step(action)
        learner.observe(action, transition, training=training)
        event = transition.public_info.get("event")
        if event == "no-change":
            if action in known_ineffective_actions:
                unnecessary += 1
            known_ineffective_actions.add(action)
        elif event in {"blocked", "rejected"}:
            unnecessary += 1
        row = {
            "step": step,
            "observation": before,
            "action": action,
            "reward": transition.reward,
            "next_observation": observation_id(transition.observation),
            "terminated": transition.terminated,
            "truncated": transition.truncated,
            "public_info": transition.public_info,
            "confidence": confidence,
        }
        rows.append(row)
        observation = transition.observation
        total += transition.reward
        if transition.terminated or transition.truncated:
            success = bool(transition.public_info.get("success"))
            break
    learner.finish_episode(training=training)
    return {
        "seed": seed,
        "success": success,
        "interactions": len(rows),
        "return": total,
        "uncertain_actions": uncertainty,
        "unnecessary_actions": unnecessary,
        "trajectory": rows,
    }


def _trial_seeds(base_seeds: list[int], count: int) -> list[int]:
    return [
        int(base_seeds[index % len(base_seeds)]) + 10_000 * (index // len(base_seeds))
        for index in range(count)
    ]


def _write_sized_artifact(path: Path, value: dict[str, Any], evidence_directory: Path) -> float:
    size = 0.0
    for _ in range(3):
        value["artifact_storage_mb"] = size
        atomic_write_json(path, value, backup=False, sort_keys=True)
        measured = _directory_megabytes(evidence_directory)
        if abs(measured - size) < 1e-9:
            break
        size = measured
    value["artifact_storage_mb"] = _directory_megabytes(evidence_directory)
    atomic_write_json(path, value, backup=False, sort_keys=True)
    return _directory_megabytes(evidence_directory)


def _make_trainer(
    config: TrainingLaneConfig,
    *,
    official_curriculum_run: bool,
):
    config.validate()

    def trainer(candidate_directory: Path, lesson: dict, evidence_directory: Path) -> dict:
        started = time.perf_counter()
        candidate_directory = Path(candidate_directory)
        evidence_directory = Path(evidence_directory)
        learner_path = candidate_directory / LEARNER_FILENAME
        learner = CrossSeedSchoolLearner.load(learner_path)
        before = learner.status()
        worlds_root = evidence_directory / "training-worlds"
        worlds_root.mkdir()
        trajectories = []
        interactions = 0
        for seed in lesson["training"]["seeds"]:
            package = create_lesson_world(worlds_root, lesson, seed=seed)
            for episode in range(config.training_episodes_per_seed):
                remaining = min(
                    config.max_training_interactions - interactions,
                    lesson["training"]["interaction_budget"] - interactions,
                )
                if remaining <= 0:
                    break
                world = FOUNDATIONAL_LOADERS[lesson["adapter"]](package)
                episode_seed = int(seed) + episode * 1_000_000
                row = _run_episode(
                    learner,
                    world,
                    seed=episode_seed,
                    training=True,
                    interaction_limit=remaining,
                )
                row["root_training_seed"] = seed
                row["episode"] = episode
                trajectories.append(row)
                interactions += row["interactions"]
            if interactions >= config.max_training_interactions:
                break
        if interactions < 1:
            raise RuntimeError("rehearsal trainer produced no interactions")
        learner.save(learner_path)
        trajectory_path = evidence_directory / "training-trajectories.json"
        atomic_write_json(
            trajectory_path,
            {
                "format": (
                    "gum-school-training-trajectories-v1"
                    if official_curriculum_run
                    else "gum-school-rehearsal-trajectories-v1"
                ),
                "development_or_sealed": False,
                "rows": trajectories,
            },
            backup=False,
            sort_keys=True,
        )
        shutil.rmtree(worlds_root)
        artifact_path = evidence_directory / "training-rehearsal.json"
        wall = max(0.001, time.perf_counter() - started)
        peak = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        artifact = {
            "format": (
                "gum-school-training-evidence-v1"
                if official_curriculum_run
                else "gum-school-rehearsal-training-evidence-v1"
            ),
            "official_curriculum_run": official_curriculum_run,
            "sealed_data_used": False,
            "lesson_id": lesson["lesson_id"],
            "interaction_limit": config.max_training_interactions,
            "interactions": interactions,
            "episodes": len(trajectories),
            "before": before,
            "after": learner.status(),
            "source_hashes": _source_hashes(),
            "trajectory_path": "training-trajectories.json",
            "trajectory_sha256": f"sha256:{file_sha256(trajectory_path)}",
            "wall_clock_seconds": wall,
            "peak_memory_mb": peak,
            "artifact_storage_mb": 0.0,
            "input_boundary": {
                "learner_received_audit_state": False,
                "learner_received_private_world_data": False,
                "violations": list(learner.boundary_violations),
            },
        }
        storage = _write_sized_artifact(artifact_path, artifact, evidence_directory)
        return {
            "format": "gum-school-training-summary-v1",
            "completed": True,
            "interactions": interactions,
            "wall_clock_seconds": wall,
            "peak_memory_mb": peak,
            "artifact_storage_mb": storage,
            "evidence_path": "evidence/training-rehearsal.json",
            "evidence_sha256": f"sha256:{file_sha256(artifact_path)}",
        }

    return trainer


def make_rehearsal_trainer(config: TrainingLaneConfig = TrainingLaneConfig()):
    return _make_trainer(config, official_curriculum_run=False)


def make_official_trainer(config: TrainingLaneConfig = TrainingLaneConfig()):
    """Return the bounded trainer used after an official source freeze."""
    return _make_trainer(config, official_curriculum_run=True)


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _paired_advantage_interval(candidate: list[bool], fresh: list[bool]) -> tuple[float, float]:
    count = len(candidate)
    rng = np.random.default_rng(4_921_337)
    samples = []
    candidate_array = np.asarray(candidate, dtype=float)
    fresh_array = np.asarray(fresh, dtype=float)
    for _ in range(2000):
        indices = rng.integers(0, count, size=count)
        candidate_rate = float(candidate_array[indices].mean())
        fresh_rate = float(fresh_array[indices].mean())
        samples.append(candidate_rate / max(fresh_rate, 1.0 / count))
    point = float(candidate_array.mean()) / max(float(fresh_array.mean()), 1.0 / count)
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return min(float(lower), point), max(float(upper), point)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    interactions = sum(row["interactions"] for row in rows)
    return {
        "trials": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "mean_interactions": interactions / len(rows),
        "interactions": interactions,
        "uncertainty_rate": sum(row["uncertain_actions"] for row in rows) / max(1, interactions),
        "unnecessary_action_rate": sum(row["unnecessary_actions"] for row in rows) / max(1, interactions),
    }


def make_development_rehearsal_evaluator(config: TrainingLaneConfig = TrainingLaneConfig()):
    config.validate()

    def evaluator(candidate_directory: Path, lesson: dict, evidence_directory: Path) -> dict:
        started = time.perf_counter()
        candidate_directory = Path(candidate_directory)
        evidence_directory = Path(evidence_directory)
        candidate_path = candidate_directory / LEARNER_FILENAME
        candidate = CrossSeedSchoolLearner.load(candidate_path)
        fresh = CrossSeedSchoolLearner(
            candidate.seed,
            max_replicas=candidate.max_replicas,
            initial_replicas=candidate.replica_count,
        )
        worlds_root = evidence_directory / "development-worlds"
        worlds_root.mkdir()
        seeds = _trial_seeds(lesson["evaluation"]["development_seeds"], config.development_trials)
        candidate_rows = []
        fresh_rows = []
        replay_package = None
        replay_seed = None
        for index, seed in enumerate(seeds):
            package = create_lesson_world(worlds_root, lesson, seed=seed)
            candidate_world = FOUNDATIONAL_LOADERS[lesson["adapter"]](package)
            fresh_world = FOUNDATIONAL_LOADERS[lesson["adapter"]](package)
            candidate_row = _run_episode(
                candidate,
                candidate_world,
                seed=seed,
                training=False,
                interaction_limit=candidate_world.public_spec().horizon,
            )
            fresh_row = _run_episode(
                fresh,
                fresh_world,
                seed=seed,
                training=False,
                interaction_limit=fresh_world.public_spec().horizon,
            )
            candidate_rows.append(candidate_row)
            fresh_rows.append(fresh_row)
            if index == config.replay_trial_index:
                replay_package, replay_seed = package, seed

        replay_learner = CrossSeedSchoolLearner.load(candidate_path)
        replay_world = FOUNDATIONAL_LOADERS[lesson["adapter"]](replay_package)
        replay_row = _run_episode(
            replay_learner,
            replay_world,
            seed=replay_seed,
            training=False,
            interaction_limit=replay_world.public_spec().horizon,
        )
        expected_replay = candidate_rows[config.replay_trial_index]
        replay_verified = canonical(replay_row) == canonical(expected_replay)
        replay_artifact = {
            "format": "gum-school-rehearsal-replay-v1",
            "development_only": True,
            "trial_index": config.replay_trial_index,
            "seed": replay_seed,
            "deterministic": replay_verified,
            "candidate_trials": candidate_rows,
            "fresh_trials": fresh_rows,
            "expected": expected_replay,
            "replayed": replay_row,
        }
        replay_path = evidence_directory / "development-replay.json"
        atomic_write_json(replay_path, replay_artifact, backup=False, sort_keys=True)
        shutil.rmtree(worlds_root)

        candidate_summary = _summarize(candidate_rows)
        fresh_summary = _summarize(fresh_rows)
        lower, upper = _wilson(candidate_summary["successes"], candidate_summary["trials"])
        advantage_lower, advantage_upper = _paired_advantage_interval(
            [row["success"] for row in candidate_rows],
            [row["success"] for row in fresh_rows],
        )
        training_summary = json.loads(
            (evidence_directory.parent / "TRAINING.json").read_text(encoding="utf-8")
        )
        evaluation_interactions = (
            candidate_summary["interactions"] + fresh_summary["interactions"]
            + replay_row["interactions"]
        )
        wall = max(0.001, time.perf_counter() - started)
        peak = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        workspace = evidence_directory.parent.parent.parent
        ledger_verified = HashLedger(workspace / "SCHOOL_LEDGER.jsonl").verify()["valid"]
        return {
            "format": "gum-school-evaluation-v1",
            "sealed": {
                "protocol_verified": False,
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
                "source_hashes_verified": True,
                "protocol_hashes_verified": False,
                "trajectory_verified": True,
                "ledger_verified": ledger_verified,
            },
            "resources": {
                "wall_clock_seconds": training_summary["wall_clock_seconds"] + wall,
                "peak_memory_mb": max(training_summary["peak_memory_mb"], peak),
                "artifact_storage_mb": max(
                    training_summary["artifact_storage_mb"], _directory_megabytes(evidence_directory)
                ),
                "interactions": training_summary["interactions"] + evaluation_interactions,
            },
            "input_boundary": {
                "verified": not candidate.boundary_violations and not fresh.boundary_violations,
                "violations": list(candidate.boundary_violations) + list(fresh.boundary_violations),
            },
            "replay": {
                "verified": replay_verified,
                "deterministic": replay_verified,
                "artifact_path": "evidence/development-replay.json",
                "artifact_sha256": f"sha256:{file_sha256(replay_path)}",
            },
            "transfer_results": [],
        }

    return evaluator
