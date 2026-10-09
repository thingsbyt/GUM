"""Official post-freeze promotion path for recurrent causal composition."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import time
from typing import Any

import psutil

from gum.lineage import HashLedger, canonical, file_sha256
from gum.storage import atomic_write_json

from .cumulative import (
    BASE_FILENAME,
    CAUSAL_FILENAME,
    BUNDLE_FILENAME,
    CumulativeSchoolLearner,
)
from .engine import SchoolEngine
from .recurrent_meta import RecurrentCausalLearner
from .sealed import (
    SEED_MAXIMUM,
    SEED_MINIMUM,
    _RGBShiftedWorld,
    _run_random_episode,
    frozen_source_hashes,
)
from .symmetry_uncertainty import (
    MINIMUM_INITIAL_UNCERTAINTY_RATE,
    summarize_symmetry_aware_uncertainty,
)
from .training import _paired_advantage_interval, _run_episode, _summarize, _wilson
from .validation import DEFAULT_CURRICULUM
from .worlds import (
    CAUSAL_WORKSHOP_ADAPTER,
    FOUNDATIONAL_LOADERS,
    create_foundational_world,
    create_lesson_world,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_CONTROLS_WORKSPACE = (
    ROOT / "evidence" / "gum-school" / "sealed-recurrent-official-v2"
)
DEFAULT_POLICY = (
    ROOT / "evidence" / "gum-school" / "research"
    / "causal-composition-reward-outcome-v2"
    / "COMPOSITION_REWARD_OUTCOME_POLICY.pt"
)
DEFAULT_TRAINING_REPORT = DEFAULT_POLICY.with_name(
    "COMPOSITION_REWARD_OUTCOME_REPORT.json"
)
INHERITED_LESSONS = [
    "object-laboratory.occlusion.001",
    "object-laboratory.functional-category.002",
    "causal-workshop.controls.001",
]
LESSON_ID = "causal-workshop.composition.002"
INTERACTION_LIMIT = 32


class CompositionOfficialError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if len(value) != 40:
        raise CompositionOfficialError("git did not return a full source revision")
    return value


def _git_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _source_hashes(runner_path: Path) -> dict[str, str]:
    hashes = frozen_source_hashes(ROOT)
    runner = Path(runner_path).resolve()
    hashes[runner.relative_to(ROOT).as_posix()] = f"sha256:{file_sha256(runner)}"
    return dict(sorted(hashes.items()))


def _all_public_seeds(curriculum: dict[str, Any]) -> set[int]:
    seeds: set[int] = set()
    for lesson in curriculum["lessons"]:
        seeds.update(int(seed) for seed in lesson["training"]["seeds"])
        seeds.update(int(seed) for seed in lesson["evaluation"]["development_seeds"])
        seeds.update(int(seed) for seed in lesson["retention"]["evaluation_seeds"])
    return seeds


def _revealed_seed_state() -> tuple[set[int], dict[str, str]]:
    seeds: set[int] = set()
    hashes: dict[str, str] = {}
    for path in sorted((ROOT / "evidence" / "gum-school").rglob("SEALED_SEED_MANIFEST.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value.get("seeds")
        if not isinstance(rows, list) or not all(
            isinstance(seed, int) and not isinstance(seed, bool) for seed in rows
        ):
            raise CompositionOfficialError(f"invalid prior sealed seed manifest: {path}")
        seeds.update(rows)
        hashes[path.relative_to(ROOT).as_posix()] = f"sha256:{file_sha256(path)}"
    return seeds, hashes


def _verified_lesson_three_state(
    official_workspace: Path,
) -> tuple[Path, dict[str, Any]]:
    official_workspace = Path(official_workspace).resolve()
    report_path = official_workspace / "OFFICIAL_RECURRENT_PROMOTION.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pointer = json.loads(
        (official_workspace / "PROMOTED.json").read_text(encoding="utf-8")
    )
    progress = json.loads(
        (official_workspace / "PROGRESS.json").read_text(encoding="utf-8")
    )
    if (
        report.get("qualification") != "pass"
        or report.get("lesson_id") != INHERITED_LESSONS[-1]
        or report.get("promoted_snapshot_id") != pointer.get("snapshot_id")
        or progress.get("promoted_lessons", [])[:3] != INHERITED_LESSONS
    ):
        raise CompositionOfficialError("official Lesson 3 inheritance is not verified")
    state = (
        official_workspace / "snapshots" / pointer["snapshot_id"] / "state"
    )
    CumulativeSchoolLearner.load_bundle(state)
    evidence = {
        "source_official_report": report_path.relative_to(ROOT).as_posix(),
        "source_official_report_sha256": f"sha256:{file_sha256(report_path)}",
        "source_promoted_snapshot_id": pointer["snapshot_id"],
        "source_bundle_components": {
            filename: f"sha256:{file_sha256(state / filename)}"
            for filename in (BASE_FILENAME, CAUSAL_FILENAME, BUNDLE_FILENAME)
        },
        "inherited_lessons": INHERITED_LESSONS,
        "verified": True,
    }
    return state, evidence


def _validate_public_candidate(
    policy_path: Path,
    training_report_path: Path,
    *,
    starting_snapshot_id: str,
) -> dict[str, Any]:
    report = json.loads(training_report_path.read_text(encoding="utf-8"))
    policy_hash = f"sha256:{file_sha256(policy_path)}"
    if (
        report.get("format")
        != "gum-school-recurrent-composition-reward-outcome-v2"
        or report.get("strict_pass") is not True
        or report.get("sealed_data_used") is not False
        or report.get("development_only") is not True
        or report.get("official_curriculum_run") is not False
        or report.get("lesson_id") != LESSON_ID
        or report.get("starting_official_snapshot") != starting_snapshot_id
        or report.get("source_hashes", {}).get("trained_policy") != policy_hash
        or report.get("training", {}).get("interactions") != 18_000
    ):
        raise CompositionOfficialError("public composition evidence is incomplete")
    RecurrentCausalLearner.load(policy_path)
    return report


def _create_freeze(
    path: Path,
    *,
    runner_path: Path,
    engine: SchoolEngine,
    lesson: dict[str, Any],
    inherited_state: Path,
    policy_path: Path,
    training_report_path: Path,
    trials: int,
) -> dict[str, Any]:
    excluded, prior_hashes = _revealed_seed_state()
    protocol = {
        "format": "gum-school-official-composition-protocol-v1",
        "evaluation_format": "gum-school-evaluation-v2",
        "lesson_id": lesson["lesson_id"],
        "trials": trials,
        "seed_selection": "post-freeze operating-system entropy",
        "exclude_all_public_and_prior_revealed_sealed_seeds": True,
        "sealed_world_shift": "composition plus deterministic spatial RGB transform",
        "per_trial_interaction_limit": INTERACTION_LIMIT,
        "candidate_learning_during_evaluation": False,
        "confidence_threshold": 0.05,
        "minimum_initial_uncertainty_rate": MINIMUM_INITIAL_UNCERTAINTY_RATE,
        "maximum_post_causal_uncertainty_rate": lesson["promotion"][
            "maximum_uncertainty_rate"
        ],
        "positive_causal_evidence": "an earlier action produced scalar reward > 0",
        "event_labels_used_by_candidate": False,
        "hidden_state_used_by_candidate": False,
        "resources_interactions_semantics": "imported public training interactions",
        "inherited_lessons": INHERITED_LESSONS,
    }
    value = {
        "format": "gum-school-official-composition-source-freeze-v1",
        "created_at_utc": _utc_now(),
        "git_head": _git_head(),
        "git_clean_before_workspace_creation": True,
        "curriculum_sha256": f"sha256:{engine.curriculum_sha256}",
        "source_hashes": _source_hashes(runner_path),
        "inherited_bundle_hashes": {
            filename: f"sha256:{file_sha256(inherited_state / filename)}"
            for filename in (BASE_FILENAME, CAUSAL_FILENAME, BUNDLE_FILENAME)
        },
        "candidate_policy_sha256": f"sha256:{file_sha256(policy_path)}",
        "public_training_report_sha256": f"sha256:{file_sha256(training_report_path)}",
        "prior_sealed_manifest_hashes": prior_hashes,
        "excluded_prior_sealed_seeds": sorted(excluded),
        "protocol": protocol,
        "protocol_sha256": f"sha256:{hashlib.sha256(canonical(protocol)).hexdigest()}",
    }
    atomic_write_json(path, value, backup=False, sort_keys=True)
    return value


def _verify_freeze(
    freeze: dict[str, Any],
    *,
    runner_path: Path,
    engine: SchoolEngine,
    inherited_state: Path,
    policy_path: Path,
    training_report_path: Path,
) -> bool:
    return (
        freeze.get("format") == "gum-school-official-composition-source-freeze-v1"
        and freeze.get("git_head") == _git_head()
        and freeze.get("curriculum_sha256") == f"sha256:{engine.curriculum_sha256}"
        and freeze.get("source_hashes") == _source_hashes(runner_path)
        and freeze.get("inherited_bundle_hashes") == {
            filename: f"sha256:{file_sha256(inherited_state / filename)}"
            for filename in (BASE_FILENAME, CAUSAL_FILENAME, BUNDLE_FILENAME)
        }
        and freeze.get("candidate_policy_sha256")
        == f"sha256:{file_sha256(policy_path)}"
        and freeze.get("public_training_report_sha256")
        == f"sha256:{file_sha256(training_report_path)}"
        and freeze.get("protocol_sha256")
        == f"sha256:{hashlib.sha256(canonical(freeze.get('protocol'))).hexdigest()}"
    )


def _select_manifest(
    path: Path,
    *,
    freeze: dict[str, Any],
    verified: bool,
    curriculum: dict[str, Any],
) -> dict[str, Any]:
    if not verified:
        raise CompositionOfficialError("source freeze changed before seed selection")
    if path.exists():
        raise FileExistsError(f"official seed manifest already exists: {path}")
    excluded = _all_public_seeds(curriculum).union(
        int(seed) for seed in freeze["excluded_prior_sealed_seeds"]
    )
    seeds: set[int] = set()
    generator = secrets.SystemRandom()
    while len(seeds) < int(freeze["protocol"]["trials"]):
        seed = generator.randrange(SEED_MINIMUM, SEED_MAXIMUM)
        if seed not in excluded:
            seeds.add(seed)
    payload = {
        "format": "gum-school-official-composition-seeds-v1",
        "selected_at_utc": _utc_now(),
        "selected_after_source_candidate_and_protocol_freeze": True,
        "selected_after_training": True,
        "prior_revealed_and_public_seeds_excluded": True,
        "authority": "operating-system-entropy",
        "source_freeze_sha256": f"sha256:{hashlib.sha256(canonical(freeze)).hexdigest()}",
        "nonce": secrets.token_hex(32),
        "seeds": sorted(seeds),
    }
    value = dict(payload)
    value["commitment"] = f"sha256:{hashlib.sha256(canonical(payload)).hexdigest()}"
    atomic_write_json(path, value, backup=False, sort_keys=True)
    return value


def _verify_manifest(
    manifest: dict[str, Any],
    *,
    freeze: dict[str, Any],
    curriculum: dict[str, Any],
) -> bool:
    payload = {key: value for key, value in manifest.items() if key != "commitment"}
    seeds = payload.get("seeds")
    excluded = _all_public_seeds(curriculum).union(
        int(seed) for seed in freeze["excluded_prior_sealed_seeds"]
    )
    return (
        manifest.get("commitment")
        == f"sha256:{hashlib.sha256(canonical(payload)).hexdigest()}"
        and payload.get("format") == "gum-school-official-composition-seeds-v1"
        and payload.get("selected_after_source_candidate_and_protocol_freeze") is True
        and payload.get("selected_after_training") is True
        and payload.get("prior_revealed_and_public_seeds_excluded") is True
        and payload.get("source_freeze_sha256")
        == f"sha256:{hashlib.sha256(canonical(freeze)).hexdigest()}"
        and isinstance(seeds, list)
        and len(seeds) == freeze["protocol"]["trials"]
        and len(set(seeds)) == len(seeds)
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
        and all(SEED_MINIMUM <= seed < SEED_MAXIMUM for seed in seeds)
        and not excluded.intersection(seeds)
    )


def _create_sealed_world(root: Path, lesson: dict[str, Any], *, seed: int) -> Path:
    return create_foundational_world(
        root,
        CAUSAL_WORKSHOP_ADAPTER,
        seed=seed,
        mechanism="composition",
    )


def _load_sealed_world(package: Path, lesson: dict[str, Any], *, seed: int):
    world = FOUNDATIONAL_LOADERS[lesson["adapter"]](package)
    return _RGBShiftedWorld(world, seed=seed, translate=True)


def _retention(
    candidate_directory: Path,
    base_directory: Path,
    *,
    current_lesson: dict[str, Any],
    prior_lessons: list[dict[str, Any]],
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evidence_rows = []
    all_rows = []
    baseline_all = []
    current_all = []
    by_lesson = []
    for prior in prior_lessons:
        baseline = CumulativeSchoolLearner.load_bundle(base_directory)
        current = CumulativeSchoolLearner.load_bundle(candidate_directory)
        baseline_rows = []
        current_rows = []
        seeds = current_lesson["retention"]["evaluation_seeds"]
        trials = int(current_lesson["retention"]["trials"])
        for index in range(trials):
            seed = int(seeds[index % len(seeds)]) + 10_000 * (index // len(seeds))
            slug = prior["lesson_id"].replace(".", "-")
            baseline_package = create_lesson_world(root / slug / "baseline", prior, seed=seed)
            current_package = create_lesson_world(root / slug / "current", prior, seed=seed)
            baseline_world = FOUNDATIONAL_LOADERS[prior["adapter"]](baseline_package)
            current_world = FOUNDATIONAL_LOADERS[prior["adapter"]](current_package)
            limit = 11 if prior["lesson_id"] == "causal-workshop.controls.001" else baseline_world.public_spec().horizon
            baseline_rows.append(_run_episode(
                baseline,
                baseline_world,
                seed=seed,
                training=False,
                interaction_limit=limit,
            ))
            current_rows.append(_run_episode(
                current,
                current_world,
                seed=seed,
                training=False,
                interaction_limit=limit,
            ))
        baseline_summary = _summarize(baseline_rows)
        current_summary = _summarize(current_rows)
        baseline_all.extend(baseline_rows)
        current_all.extend(current_rows)
        all_rows.extend(baseline_rows + current_rows)
        by_lesson.append({
            "lesson_id": prior["lesson_id"],
            "baseline_summary": baseline_summary,
            "current_summary": current_summary,
            "baseline_trials": baseline_rows,
            "current_trials": current_rows,
        })
        for skill in prior["capabilities"]:
            evidence_rows.append({
                "skill": skill,
                "baseline_success_rate": baseline_summary["success_rate"],
                "current_success_rate": current_summary["success_rate"],
                "evidence_verified": True,
            })
    return evidence_rows, all_rows, {
        "by_lesson": by_lesson,
        "baseline_summary": _summarize(baseline_all),
        "current_summary": _summarize(current_all),
    }


def run_official_composition_promotion(
    workspace: Path,
    *,
    runner_path: Path,
    curriculum_path: Path = DEFAULT_CURRICULUM,
    official_controls_workspace: Path = DEFAULT_OFFICIAL_CONTROLS_WORKSPACE,
    policy_path: Path = DEFAULT_POLICY,
    training_report_path: Path = DEFAULT_TRAINING_REPORT,
    trials: int = 64,
) -> dict[str, Any]:
    if trials < 32:
        raise CompositionOfficialError("official composition promotion requires at least 32 trials")
    workspace = Path(workspace).resolve()
    if workspace.exists():
        raise FileExistsError(f"official composition workspace already exists: {workspace}")
    if not _git_clean():
        raise CompositionOfficialError("commit the examiner and protocol before running")

    inherited_state, inheritance = _verified_lesson_three_state(
        official_controls_workspace
    )
    policy_path = Path(policy_path).resolve()
    training_report_path = Path(training_report_path).resolve()
    training_report = _validate_public_candidate(
        policy_path,
        training_report_path,
        starting_snapshot_id=inheritance["source_promoted_snapshot_id"],
    )

    workspace.mkdir(parents=True)
    bootstrap = workspace / "bootstrap"
    shutil.copytree(inherited_state, bootstrap)
    engine = SchoolEngine(workspace, curriculum_path)
    engine.initialize_promoted(
        bootstrap,
        inherited_lessons=INHERITED_LESSONS,
        inheritance_evidence=inheritance,
    )
    started = engine.begin_lesson(LESSON_ID)
    run_id = started["run_id"]
    run_directory = workspace / "runs" / run_id
    lesson = started["lesson"]
    base_evaluation = run_directory / "base-evaluation-state"
    engine.snapshots.materialize(
        engine._run_state(run_id)["base_snapshot_id"], base_evaluation
    )
    freeze = _create_freeze(
        run_directory / "SOURCE_FREEZE.json",
        runner_path=runner_path,
        engine=engine,
        lesson=lesson,
        inherited_state=inherited_state,
        policy_path=policy_path,
        training_report_path=training_report_path,
        trials=trials,
    )
    try:
        training_started = time.perf_counter()
        CumulativeSchoolLearner.replace_causal_policy(
            started["candidate_directory"], causal_path=policy_path
        )
        training_artifact = {
            "format": "gum-school-composition-training-import-v1",
            "official_curriculum_run": True,
            "lesson_id": lesson["lesson_id"],
            "sealed_data_used": False,
            "policy_weights_changed_during_import": False,
            "public_training_interactions": int(training_report["training"]["interactions"]),
            "public_training_report_sha256": f"sha256:{file_sha256(training_report_path)}",
            "candidate_policy_sha256": f"sha256:{file_sha256(policy_path)}",
            "source_freeze_sha256": f"sha256:{file_sha256(run_directory / 'SOURCE_FREEZE.json')}",
        }
        training_artifact_path = run_directory / "evidence" / "training-import.json"
        atomic_write_json(training_artifact_path, training_artifact, backup=False, sort_keys=True)
        training_summary = {
            "format": "gum-school-training-summary-v1",
            "completed": True,
            "interactions": int(training_report["training"]["interactions"]),
            "wall_clock_seconds": max(0.001, time.perf_counter() - training_started),
            "peak_memory_mb": psutil.Process().memory_info().rss / (1024.0 * 1024.0),
            "artifact_storage_mb": sum(
                path.stat().st_size
                for path in (run_directory / "evidence").rglob("*")
                if path.is_file()
            ) / (1024.0 * 1024.0),
            "evidence_path": "evidence/training-import.json",
            "evidence_sha256": f"sha256:{file_sha256(training_artifact_path)}",
        }
        engine.mark_training_complete(run_id, training_summary)
        freeze_verified = _verify_freeze(
            freeze,
            runner_path=runner_path,
            engine=engine,
            inherited_state=inherited_state,
            policy_path=policy_path,
            training_report_path=training_report_path,
        )
        manifest = _select_manifest(
            run_directory / "SEALED_SEED_MANIFEST.json",
            freeze=freeze,
            verified=freeze_verified,
            curriculum=engine.curriculum,
        )
        manifest_verified = _verify_manifest(
            manifest, freeze=freeze, curriculum=engine.curriculum
        )
        if not manifest_verified:
            raise CompositionOfficialError("official seed manifest verification failed")

        evaluation_started = time.perf_counter()
        candidate_directory = engine.candidate_snapshot_directory(run_id)
        candidate = CumulativeSchoolLearner.load_bundle(candidate_directory)
        base = CumulativeSchoolLearner.load_bundle(base_evaluation)
        fresh = RecurrentCausalLearner(candidate.causal.seed, config=candidate.causal.config)
        worlds_root = run_directory / "evidence" / "sealed-worlds"
        worlds_root.mkdir()
        candidate_rows = []
        base_rows = []
        fresh_rows = []
        random_rows = []
        replay_package = None
        replay_seed = None
        for index, seed in enumerate(manifest["seeds"]):
            package = _create_sealed_world(worlds_root, lesson, seed=seed)
            candidate_rows.append(_run_episode(
                candidate,
                _load_sealed_world(package, lesson, seed=seed),
                seed=seed,
                training=False,
                interaction_limit=INTERACTION_LIMIT,
            ))
            base_rows.append(_run_episode(
                base,
                _load_sealed_world(package, lesson, seed=seed),
                seed=seed,
                training=False,
                interaction_limit=INTERACTION_LIMIT,
            ))
            fresh_rows.append(_run_episode(
                fresh,
                _load_sealed_world(package, lesson, seed=seed),
                seed=seed,
                training=False,
                interaction_limit=INTERACTION_LIMIT,
            ))
            random_rows.append(_run_random_episode(
                _load_sealed_world(package, lesson, seed=seed),
                seed=seed,
                policy_seed=seed ^ 0xC0A10517,
                interaction_limit=INTERACTION_LIMIT,
            ))
            if index == 0:
                replay_package, replay_seed = package, seed
        if replay_package is None or replay_seed is None:
            raise CompositionOfficialError("official replay trial was not selected")
        replay = _run_episode(
            CumulativeSchoolLearner.load_bundle(candidate_directory),
            _load_sealed_world(replay_package, lesson, seed=replay_seed),
            seed=replay_seed,
            training=False,
            interaction_limit=INTERACTION_LIMIT,
        )
        replay_verified = canonical(replay) == canonical(candidate_rows[0])
        candidate_summary = _summarize(candidate_rows)
        base_summary = _summarize(base_rows)
        fresh_summary = _summarize(fresh_rows)
        random_summary = _summarize(random_rows)
        calibration = summarize_symmetry_aware_uncertainty(candidate_rows)
        success_lower, success_upper = _wilson(
            candidate_summary["successes"], candidate_summary["trials"]
        )
        advantage_lower, advantage_upper = _paired_advantage_interval(
            [row["success"] for row in candidate_rows],
            [row["success"] for row in fresh_rows],
        )
        prior_lessons = [engine.lesson_by_id[lesson_id] for lesson_id in INHERITED_LESSONS]
        retention_rows, retention_trial_rows, retention = _retention(
            candidate_directory,
            base_evaluation,
            current_lesson=lesson,
            prior_lessons=prior_lessons,
            root=run_directory / "evidence" / "retention-worlds",
        )
        trajectories = {
            "format": "gum-school-official-composition-trajectories-v1",
            "candidate": candidate_rows,
            "candidate_before_training": base_rows,
            "matched_fresh": fresh_rows,
            "random": random_rows,
            "retention": retention,
        }
        trajectory_path = run_directory / "evidence" / "official-trajectories.json"
        atomic_write_json(trajectory_path, trajectories, backup=False, sort_keys=True)
        trajectory_verified = (
            json.loads(trajectory_path.read_text(encoding="utf-8")) == trajectories
        )
        replay_path = run_directory / "evidence" / "official-replay.json"
        atomic_write_json(
            replay_path,
            {
                "format": "gum-school-official-composition-replay-v1",
                "expected": candidate_rows[0],
                "replayed": replay,
            },
            backup=False,
            sort_keys=True,
        )
        shutil.rmtree(worlds_root)
        shutil.rmtree(run_directory / "evidence" / "retention-worlds")
        evaluation_interactions = sum(
            row["interactions"]
            for row in candidate_rows + base_rows + fresh_rows + random_rows
            + retention_trial_rows + [replay]
        )
        evaluation = {
            "format": "gum-school-evaluation-v2",
            "sealed": {
                "protocol_verified": freeze_verified and manifest_verified,
                "trials": candidate_summary["trials"],
                "success_rate": candidate_summary["success_rate"],
                "fresh_success_rate": fresh_summary["success_rate"],
                "uncertainty_rate": calibration["post_causal_evidence_uncertainty_rate"],
                "unnecessary_action_rate": candidate_summary["unnecessary_action_rate"],
                "same_perception": True,
                "same_resource_limits": True,
                "confidence_level": lesson["promotion"]["confidence_level"],
                "success_interval": {
                    "lower": success_lower,
                    "upper": success_upper,
                    "method": "wilson",
                },
                "fresh_advantage_interval": {
                    "lower": advantage_lower,
                    "upper": advantage_upper,
                    "method": "paired-bootstrap",
                },
            },
            "uncertainty_calibration": calibration,
            "retention": retention_rows,
            "evidence": {
                "source_hashes_verified": freeze_verified,
                "protocol_hashes_verified": manifest_verified,
                "trajectory_verified": trajectory_verified,
                "ledger_verified": HashLedger(workspace / "SCHOOL_LEDGER.jsonl").verify()["valid"],
            },
            "resources": {
                "wall_clock_seconds": training_summary["wall_clock_seconds"]
                + max(0.001, time.perf_counter() - evaluation_started),
                "peak_memory_mb": max(
                    training_summary["peak_memory_mb"],
                    psutil.Process().memory_info().rss / (1024.0 * 1024.0),
                ),
                "artifact_storage_mb": sum(
                    path.stat().st_size
                    for path in (run_directory / "evidence").rglob("*")
                    if path.is_file()
                ) / (1024.0 * 1024.0),
                "interactions": training_summary["interactions"],
            },
            "input_boundary": {
                "verified": not candidate.boundary_violations and not base.boundary_violations,
                "violations": list(candidate.boundary_violations) + list(base.boundary_violations),
            },
            "replay": {
                "verified": replay_verified,
                "deterministic": replay_verified,
                "artifact_path": "evidence/official-replay.json",
                "artifact_sha256": f"sha256:{file_sha256(replay_path)}",
            },
            "transfer_results": [],
        }
        engine.record_evaluation(run_id, evaluation)
        decision = engine.finalize(run_id)
    except Exception as error:
        active = engine.status()["active"]
        if active["status"] == "active":
            engine.abort(run_id, f"{type(error).__name__}: {error}")
        raise

    final_pointer = engine.snapshots.promoted()
    next_lesson = engine.next_lesson()
    report = {
        "format": "gum-school-official-composition-promotion-v1",
        "created_at_utc": _utc_now(),
        "official_curriculum_run": True,
        "development_only": False,
        "sealed_data_used_for_training": False,
        "lesson_id": lesson["lesson_id"],
        "run_id": run_id,
        "source_freeze": freeze,
        "seed_manifest": manifest,
        "candidate": candidate_summary,
        "candidate_before_training": base_summary,
        "matched_fresh": fresh_summary,
        "uniform_random": random_summary,
        "uncertainty_calibration": calibration,
        "retention": retention,
        "evaluation_interactions": evaluation_interactions,
        "decision": decision,
        "promoted_snapshot_id": final_pointer["snapshot_id"],
        "next_lesson_id": None if next_lesson is None else next_lesson["lesson_id"],
        "ledger": HashLedger(workspace / "SCHOOL_LEDGER.jsonl").verify(),
        "qualification": "pass" if decision["outcome"] == "promote" else "fail",
    }
    atomic_write_json(
        workspace / "OFFICIAL_COMPOSITION_PROMOTION.json",
        report,
        backup=False,
        sort_keys=True,
    )
    return report
