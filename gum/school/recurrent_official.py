"""Official engine promotion path for the recurrent causal specialist.

The track inherits the first two verified Object Laboratory promotions, imports
the frozen public-trained recurrent policy into an engine candidate, freezes
source and protocol, draws never-before-revealed sealed seeds, and lets the
ordinary SchoolEngine make the atomic promotion decision from a v2 evaluation.
"""
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

from .cumulative import BASE_FILENAME, CumulativeSchoolLearner
from .engine import SchoolEngine
from .learner import CrossSeedSchoolLearner
from .recurrent_meta import RecurrentCausalLearner
from .sealed import (
    SEED_MAXIMUM,
    SEED_MINIMUM,
    _create_sealed_world,
    _load_sealed_world,
    _run_random_episode,
    frozen_source_hashes,
)
from .symmetry_uncertainty import (
    MINIMUM_INITIAL_UNCERTAINTY_RATE,
    summarize_symmetry_aware_uncertainty,
)
from .training import _paired_advantage_interval, _run_episode, _summarize, _wilson
from .validation import DEFAULT_CURRICULUM
from .worlds import FOUNDATIONAL_LOADERS, create_lesson_world


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_SCHOOL = (
    ROOT / "evidence" / "gum-school" / "sealed" / "object-laboratory-occlusion-v1"
)
DEFAULT_POLICY = (
    ROOT / "evidence" / "gum-school" / "research" / "causal-recurrent-meta-v1"
    / "RECURRENT_META_POLICY.pt"
)
DEFAULT_TRAINING_REPORT = DEFAULT_POLICY.with_name("RECURRENT_META_REPORT.json")
DEFAULT_PUBLIC_VALIDATION = (
    ROOT / "evidence" / "gum-school" / "research" / "causal-recurrent-meta-v2"
    / "SYMMETRY_UNCERTAINTY_VALIDATION.json"
)
DEFAULT_SUPPLEMENTAL_REPORT = (
    ROOT / "evidence" / "gum-school" / "sealed-recheck" / "causal-recurrent-meta-v2"
    / "SEALED_SYMMETRY_CONFIRMATION.json"
)
INHERITED_LESSONS = [
    "object-laboratory.occlusion.001",
    "object-laboratory.functional-category.002",
]


class RecurrentOfficialError(RuntimeError):
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
        raise RecurrentOfficialError("git did not return a full source revision")
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
    runner_path = Path(runner_path).resolve()
    hashes[runner_path.relative_to(ROOT).as_posix()] = f"sha256:{file_sha256(runner_path)}"
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
            raise RecurrentOfficialError(f"invalid prior sealed seed manifest: {path}")
        seeds.update(rows)
        hashes[path.relative_to(ROOT).as_posix()] = f"sha256:{file_sha256(path)}"
    return seeds, hashes


def _verified_lesson_two_state(official_school: Path) -> tuple[Path, dict[str, Any]]:
    official_school = Path(official_school).resolve()
    report = json.loads((official_school / "SCHOOL_REPORT.json").read_text(encoding="utf-8"))
    decisions = {
        row["lesson_id"]: row
        for row in report.get("decisions", [])
        if row.get("outcome") == "promote"
    }
    missing = [lesson for lesson in INHERITED_LESSONS if lesson not in decisions]
    if missing:
        raise RecurrentOfficialError(f"official inheritance evidence is missing {missing}")
    lesson_two = decisions[INHERITED_LESSONS[-1]]
    snapshot_id = lesson_two["candidate_snapshot_id"]
    state = official_school / "snapshots" / snapshot_id / "state"
    base = state / BASE_FILENAME
    if not base.is_file():
        raise RecurrentOfficialError("verified Lesson 2 snapshot has no base learner")
    CrossSeedSchoolLearner.load(base)
    evidence = {
        "source_school_report": official_school.relative_to(ROOT).as_posix() + "/SCHOOL_REPORT.json",
        "source_school_report_sha256": f"sha256:{file_sha256(official_school / 'SCHOOL_REPORT.json')}",
        "inherited_lessons": INHERITED_LESSONS,
        "lesson_two_snapshot_id": snapshot_id,
        "lesson_two_base_sha256": f"sha256:{file_sha256(base)}",
        "verified": True,
    }
    return base, evidence


def _create_freeze(
    path: Path,
    *,
    runner_path: Path,
    engine: SchoolEngine,
    lesson: dict[str, Any],
    base_path: Path,
    policy_path: Path,
    training_report_path: Path,
    public_validation_path: Path,
    supplemental_report_path: Path,
    trials: int,
) -> dict[str, Any]:
    training = json.loads(training_report_path.read_text(encoding="utf-8"))
    validation = json.loads(public_validation_path.read_text(encoding="utf-8"))
    supplemental = json.loads(supplemental_report_path.read_text(encoding="utf-8"))
    policy_hash = f"sha256:{file_sha256(policy_path)}"
    if (
        training.get("strict_pass") is not True
        or training.get("sealed_data_used") is not False
        or validation.get("validation_pass") is not True
        or validation.get("policy_weights_changed") is not False
        or validation.get("artifact_hashes", {}).get("policy") != policy_hash
        or supplemental.get("qualification") != "pass"
        or supplemental.get("candidate_bundle_hashes", {}).get("CAUSAL_META_POLICY.pt")
        != policy_hash
    ):
        raise RecurrentOfficialError("frozen policy evidence is incomplete or inconsistent")
    excluded, prior_hashes = _revealed_seed_state()
    protocol = {
        "format": "gum-school-official-recurrent-protocol-v1",
        "evaluation_format": "gum-school-evaluation-v2",
        "lesson_id": lesson["lesson_id"],
        "trials": trials,
        "seed_selection": "post-training operating-system entropy",
        "exclude_all_prior_revealed_sealed_seeds": True,
        "per_trial_interaction_limit": 11,
        "candidate_learning_during_evaluation": False,
        "confidence_threshold": 0.05,
        "minimum_initial_uncertainty_rate": MINIMUM_INITIAL_UNCERTAINTY_RATE,
        "maximum_post_causal_uncertainty_rate": lesson["promotion"][
            "maximum_uncertainty_rate"
        ],
        "positive_causal_evidence": "an earlier action produced scalar reward > 0",
        "event_labels_used_for_uncertainty": False,
        "hidden_state_used_for_uncertainty": False,
        "resources_interactions_semantics": "candidate-changing training interactions",
        "inherited_lessons": INHERITED_LESSONS,
    }
    value = {
        "format": "gum-school-official-recurrent-source-freeze-v1",
        "created_at_utc": _utc_now(),
        "git_head": _git_head(),
        "git_clean_before_workspace_creation": True,
        "curriculum_sha256": f"sha256:{engine.curriculum_sha256}",
        "source_hashes": _source_hashes(runner_path),
        "base_learner_sha256": f"sha256:{file_sha256(base_path)}",
        "causal_policy_sha256": policy_hash,
        "public_training_report_sha256": f"sha256:{file_sha256(training_report_path)}",
        "public_validation_sha256": f"sha256:{file_sha256(public_validation_path)}",
        "supplemental_confirmation_sha256": f"sha256:{file_sha256(supplemental_report_path)}",
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
    base_path: Path,
    policy_path: Path,
    training_report_path: Path,
    public_validation_path: Path,
    supplemental_report_path: Path,
) -> bool:
    return (
        freeze.get("format") == "gum-school-official-recurrent-source-freeze-v1"
        and freeze.get("git_head") == _git_head()
        and freeze.get("curriculum_sha256") == f"sha256:{engine.curriculum_sha256}"
        and freeze.get("source_hashes") == _source_hashes(runner_path)
        and freeze.get("base_learner_sha256") == f"sha256:{file_sha256(base_path)}"
        and freeze.get("causal_policy_sha256") == f"sha256:{file_sha256(policy_path)}"
        and freeze.get("public_training_report_sha256")
        == f"sha256:{file_sha256(training_report_path)}"
        and freeze.get("public_validation_sha256")
        == f"sha256:{file_sha256(public_validation_path)}"
        and freeze.get("supplemental_confirmation_sha256")
        == f"sha256:{file_sha256(supplemental_report_path)}"
        and freeze.get("protocol_sha256")
        == f"sha256:{hashlib.sha256(canonical(freeze.get('protocol'))).hexdigest()}"
    )


def _select_manifest(
    path: Path,
    *,
    freeze: dict[str, Any],
    verify: bool,
    curriculum: dict[str, Any],
) -> dict[str, Any]:
    if not verify:
        raise RecurrentOfficialError("source freeze changed before official seed selection")
    if path.exists():
        raise FileExistsError(f"official seed manifest already exists: {path}")
    excluded = _all_public_seeds(curriculum).union(
        int(seed) for seed in freeze["excluded_prior_sealed_seeds"]
    )
    seeds: set[int] = set()
    rng = secrets.SystemRandom()
    while len(seeds) < int(freeze["protocol"]["trials"]):
        seed = rng.randrange(SEED_MINIMUM, SEED_MAXIMUM)
        if seed not in excluded:
            seeds.add(seed)
    payload = {
        "format": "gum-school-official-recurrent-seeds-v1",
        "selected_at_utc": _utc_now(),
        "selected_after_source_candidate_and_protocol_freeze": True,
        "selected_after_training_import": True,
        "prior_revealed_seeds_excluded": True,
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
    manifest: dict[str, Any], *, freeze: dict[str, Any], curriculum: dict[str, Any]
) -> bool:
    payload = {key: value for key, value in manifest.items() if key != "commitment"}
    seeds = payload.get("seeds")
    excluded = _all_public_seeds(curriculum).union(
        int(seed) for seed in freeze["excluded_prior_sealed_seeds"]
    )
    return (
        manifest.get("commitment")
        == f"sha256:{hashlib.sha256(canonical(payload)).hexdigest()}"
        and payload.get("format") == "gum-school-official-recurrent-seeds-v1"
        and payload.get("selected_after_source_candidate_and_protocol_freeze") is True
        and payload.get("selected_after_training_import") is True
        and payload.get("prior_revealed_seeds_excluded") is True
        and payload.get("source_freeze_sha256")
        == f"sha256:{hashlib.sha256(canonical(freeze)).hexdigest()}"
        and isinstance(seeds, list)
        and len(seeds) == freeze["protocol"]["trials"]
        and len(set(seeds)) == len(seeds)
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
        and all(SEED_MINIMUM <= seed < SEED_MAXIMUM for seed in seeds)
        and not excluded.intersection(seeds)
    )


def _retention(
    candidate_directory: Path,
    base_path: Path,
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
        baseline = CrossSeedSchoolLearner.load(base_path)
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
            baseline_rows.append(_run_episode(
                baseline,
                baseline_world,
                seed=seed,
                training=False,
                interaction_limit=baseline_world.public_spec().horizon,
            ))
            current_rows.append(_run_episode(
                current,
                current_world,
                seed=seed,
                training=False,
                interaction_limit=current_world.public_spec().horizon,
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


def run_official_recurrent_promotion(
    workspace: Path,
    *,
    runner_path: Path,
    curriculum_path: Path = DEFAULT_CURRICULUM,
    official_school: Path = DEFAULT_OFFICIAL_SCHOOL,
    policy_path: Path = DEFAULT_POLICY,
    training_report_path: Path = DEFAULT_TRAINING_REPORT,
    public_validation_path: Path = DEFAULT_PUBLIC_VALIDATION,
    supplemental_report_path: Path = DEFAULT_SUPPLEMENTAL_REPORT,
    trials: int = 64,
) -> dict[str, Any]:
    if trials < 32:
        raise RecurrentOfficialError("official recurrent promotion requires at least 32 trials")
    workspace = Path(workspace).resolve()
    if workspace.exists():
        raise FileExistsError(f"official recurrent workspace already exists: {workspace}")
    if not _git_clean():
        raise RecurrentOfficialError("commit the official engine and protocol before running")

    base_path, inheritance = _verified_lesson_two_state(official_school)
    training_report_path = Path(training_report_path).resolve()
    public_validation_path = Path(public_validation_path).resolve()
    supplemental_report_path = Path(supplemental_report_path).resolve()
    policy_path = Path(policy_path).resolve()
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))

    workspace.mkdir(parents=True)
    bootstrap = workspace / "bootstrap"
    bootstrap.mkdir()
    shutil.copy2(base_path, bootstrap / BASE_FILENAME)
    engine = SchoolEngine(workspace, curriculum_path)
    engine.initialize_promoted(
        bootstrap,
        inherited_lessons=INHERITED_LESSONS,
        inheritance_evidence=inheritance,
    )
    started = engine.begin_lesson("causal-workshop.controls.001")
    run_id = started["run_id"]
    run_directory = workspace / "runs" / run_id
    lesson = started["lesson"]
    base_evaluation = run_directory / "base-evaluation-state"
    engine.snapshots.materialize(engine._run_state(run_id)["base_snapshot_id"], base_evaluation)
    freeze = _create_freeze(
        run_directory / "SOURCE_FREEZE.json",
        runner_path=runner_path,
        engine=engine,
        lesson=lesson,
        base_path=base_path,
        policy_path=policy_path,
        training_report_path=training_report_path,
        public_validation_path=public_validation_path,
        supplemental_report_path=supplemental_report_path,
        trials=trials,
    )
    try:
        training_started = time.perf_counter()
        CumulativeSchoolLearner.augment_existing_base(
            started["candidate_directory"], causal_path=policy_path
        )
        training_artifact = {
            "format": "gum-school-recurrent-training-import-v1",
            "official_curriculum_run": True,
            "lesson_id": lesson["lesson_id"],
            "sealed_data_used": False,
            "policy_weights_changed_during_import": False,
            "public_training_interactions": int(training_report["training"]["interactions"]),
            "public_training_report_sha256": f"sha256:{file_sha256(training_report_path)}",
            "causal_policy_sha256": f"sha256:{file_sha256(policy_path)}",
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
            base_path=base_path,
            policy_path=policy_path,
            training_report_path=training_report_path,
            public_validation_path=public_validation_path,
            supplemental_report_path=supplemental_report_path,
        )
        manifest = _select_manifest(
            run_directory / "SEALED_SEED_MANIFEST.json",
            freeze=freeze,
            verify=freeze_verified,
            curriculum=engine.curriculum,
        )
        manifest_verified = _verify_manifest(
            manifest, freeze=freeze, curriculum=engine.curriculum
        )
        if not manifest_verified:
            raise RecurrentOfficialError("official sealed manifest verification failed")

        evaluation_started = time.perf_counter()
        candidate_directory = engine.candidate_snapshot_directory(run_id)
        candidate = CumulativeSchoolLearner.load_bundle(candidate_directory)
        base = CrossSeedSchoolLearner.load(base_evaluation / BASE_FILENAME)
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
                interaction_limit=11,
            ))
            base_rows.append(_run_episode(
                base,
                _load_sealed_world(package, lesson, seed=seed),
                seed=seed,
                training=False,
                interaction_limit=11,
            ))
            fresh_rows.append(_run_episode(
                fresh,
                _load_sealed_world(package, lesson, seed=seed),
                seed=seed,
                training=False,
                interaction_limit=11,
            ))
            random_rows.append(_run_random_episode(
                _load_sealed_world(package, lesson, seed=seed),
                seed=seed,
                policy_seed=seed ^ 0x5EED5EED,
                interaction_limit=11,
            ))
            if index == 0:
                replay_package, replay_seed = package, seed
        if replay_package is None or replay_seed is None:
            raise RecurrentOfficialError("official replay trial was not selected")
        replay = _run_episode(
            CumulativeSchoolLearner.load_bundle(candidate_directory),
            _load_sealed_world(replay_package, lesson, seed=replay_seed),
            seed=replay_seed,
            training=False,
            interaction_limit=11,
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
            base_evaluation / BASE_FILENAME,
            current_lesson=lesson,
            prior_lessons=prior_lessons,
            root=run_directory / "evidence" / "retention-worlds",
        )
        trajectories = {
            "format": "gum-school-official-recurrent-trajectories-v1",
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
            {"format": "gum-school-official-recurrent-replay-v1", "expected": candidate_rows[0], "replayed": replay},
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
    report = {
        "format": "gum-school-official-recurrent-promotion-v1",
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
        "next_lesson_id": engine.next_lesson()["lesson_id"],
        "ledger": HashLedger(workspace / "SCHOOL_LEDGER.jsonl").verify(),
        "qualification": "pass" if decision["outcome"] == "promote" else "fail",
    }
    report_path = workspace / "OFFICIAL_RECURRENT_PROMOTION.json"
    atomic_write_json(report_path, report, backup=False, sort_keys=True)
    return report
