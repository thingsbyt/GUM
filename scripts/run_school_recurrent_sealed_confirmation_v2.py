"""Run a fresh sealed exam with symmetry-aware uncertainty calibration.

Version 1 remains immutable.  This version precommits a distinction between
justified uncertainty while anonymous controls are indistinguishable and
residual uncertainty after a prior action has produced positive scalar reward.
It never reuses a seed from an earlier sealed manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import time
from typing import Any

import psutil

from gum.lineage import HashLedger, canonical, file_sha256
from gum.school.cumulative import CumulativeSchoolLearner
from gum.school.evaluation import evaluate_promotion_gates
from gum.school.recurrent_meta import RecurrentCausalLearner
from gum.school.sealed import (
    CAUSAL_CONTROL_SHIFT,
    SEED_MAXIMUM,
    SEED_MINIMUM,
    _create_sealed_world,
    _load_sealed_world,
    _run_random_episode,
)
from gum.school.symmetry_uncertainty import (
    MINIMUM_INITIAL_UNCERTAINTY_RATE,
    evaluate_symmetry_uncertainty_gate,
    summarize_symmetry_aware_uncertainty,
)
from gum.school.training import (
    _paired_advantage_interval,
    _run_episode,
    _summarize,
    _wilson,
)
from gum.school.validation import DEFAULT_CURRICULUM
from gum.storage import atomic_write_json
if __package__:
    from scripts.run_school_recurrent_sealed_confirmation import (
        DEFAULT_CAUSAL,
        DEFAULT_RESEARCH_REPORT,
        DEFAULT_SCHOOL,
        ConfirmationError,
        _all_public_seeds,
        _candidate_hashes,
        _git_clean,
        _git_head,
        _promoted_state,
        _retention,
        _source_paths as _v1_source_paths,
    )
else:
    from run_school_recurrent_sealed_confirmation import (
        DEFAULT_CAUSAL,
        DEFAULT_RESEARCH_REPORT,
        DEFAULT_SCHOOL,
        ConfirmationError,
        _all_public_seeds,
        _candidate_hashes,
        _git_clean,
        _git_head,
        _promoted_state,
        _retention,
        _source_paths as _v1_source_paths,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC_VALIDATION = (
    ROOT / "evidence" / "gum-school" / "research" / "causal-recurrent-meta-v2"
    / "SYMMETRY_UNCERTAINTY_VALIDATION.json"
)
REPORT_FILENAME = "SEALED_SYMMETRY_CONFIRMATION.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_paths() -> list[Path]:
    paths = _v1_source_paths() + [
        ROOT / "gum" / "school" / "symmetry_uncertainty.py",
        Path(__file__).resolve(),
    ]
    return sorted(set(paths))


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): f"sha256:{file_sha256(path)}"
        for path in paths
    }


def _revealed_sealed_seeds() -> tuple[set[int], dict[str, str]]:
    seeds: set[int] = set()
    hashes: dict[str, str] = {}
    evidence_root = ROOT / "evidence" / "gum-school"
    for path in sorted(evidence_root.rglob("SEALED_SEED_MANIFEST.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfirmationError(f"cannot inspect prior sealed manifest {path}: {error}") from error
        manifest_seeds = value.get("seeds")
        if not isinstance(manifest_seeds, list) or not all(
            isinstance(seed, int) and not isinstance(seed, bool)
            for seed in manifest_seeds
        ):
            raise ConfirmationError(f"prior sealed manifest has invalid seeds: {path}")
        seeds.update(manifest_seeds)
        hashes[path.relative_to(ROOT).as_posix()] = f"sha256:{file_sha256(path)}"
    return seeds, hashes


def _create_freeze(
    path: Path,
    *,
    candidate: Path,
    research_report: Path,
    public_validation_path: Path,
    trials: int,
) -> dict[str, Any]:
    public_validation = json.loads(public_validation_path.read_text(encoding="utf-8"))
    if (
        public_validation.get("validation_pass") is not True
        or public_validation.get("sealed_data_used") is not False
        or public_validation.get("policy_weights_changed") is not False
    ):
        raise ConfirmationError("symmetry-aware protocol lacks qualifying public validation")
    policy_hash = _candidate_hashes(candidate)["CAUSAL_META_POLICY.pt"]
    if public_validation.get("artifact_hashes", {}).get("policy") != policy_hash:
        raise ConfirmationError("public validation used a different causal policy")
    excluded_seeds, prior_manifest_hashes = _revealed_sealed_seeds()
    protocol = {
        "format": "gum-school-recurrent-sealed-protocol-v2",
        "lesson_id": "causal-workshop.controls.001",
        "selection": "post-freeze system-random unique integers",
        "seed_range": [SEED_MINIMUM, SEED_MAXIMUM],
        "trials": trials,
        "mechanism": CAUSAL_CONTROL_SHIFT,
        "per_trial_interaction_limit": 11,
        "controls": ["same-untrained-network", "uniform-random"],
        "retention_lessons": [
            "object-laboratory.occlusion.001",
            "object-laboratory.functional-category.002",
        ],
        "candidate_learning_during_evaluation": False,
        "promotion_workspace_mutated": False,
        "uncertainty_measure": "post-causal-evidence decision uncertainty",
        "positive_causal_evidence": "an earlier action produced scalar reward > 0",
        "current_action_consequence_is_prior_evidence": False,
        "confidence_threshold": 0.05,
        "minimum_initial_uncertainty_rate": MINIMUM_INITIAL_UNCERTAINTY_RATE,
        "maximum_post_causal_uncertainty_rate": 0.2,
        "require_positive_confidence_gain": True,
        "event_labels_used_for_uncertainty": False,
        "hidden_state_used_for_uncertainty": False,
        "reuse_prior_sealed_seeds": False,
    }
    value = {
        "format": "gum-school-recurrent-source-freeze-v2",
        "created_at_utc": _utc_now(),
        "git_head": _git_head(),
        "git_clean_before_workspace_creation": True,
        "source_hashes": _hashes(_source_paths()),
        "candidate_hashes": _candidate_hashes(candidate),
        "public_training_evidence_sha256": f"sha256:{file_sha256(research_report)}",
        "public_protocol_validation_sha256": f"sha256:{file_sha256(public_validation_path)}",
        "prior_sealed_manifest_hashes": prior_manifest_hashes,
        "excluded_prior_sealed_seeds": sorted(excluded_seeds),
        "protocol": protocol,
        "protocol_sha256": f"sha256:{hashlib.sha256(canonical(protocol)).hexdigest()}",
    }
    atomic_write_json(path, value, backup=False, sort_keys=True)
    return value


def _verify_freeze(
    freeze: dict[str, Any],
    *,
    candidate: Path,
    research_report: Path,
    public_validation_path: Path,
) -> bool:
    protocol_hash = f"sha256:{hashlib.sha256(canonical(freeze.get('protocol'))).hexdigest()}"
    return (
        freeze.get("format") == "gum-school-recurrent-source-freeze-v2"
        and freeze.get("git_clean_before_workspace_creation") is True
        and freeze.get("source_hashes") == _hashes(_source_paths())
        and freeze.get("candidate_hashes") == _candidate_hashes(candidate)
        and freeze.get("public_training_evidence_sha256")
        == f"sha256:{file_sha256(research_report)}"
        and freeze.get("public_protocol_validation_sha256")
        == f"sha256:{file_sha256(public_validation_path)}"
        and freeze.get("protocol_sha256") == protocol_hash
        and freeze.get("git_head") == _git_head()
    )


def _select_manifest(
    path: Path,
    *,
    freeze: dict[str, Any],
    candidate: Path,
    research_report: Path,
    public_validation_path: Path,
    curriculum: dict[str, Any],
) -> dict[str, Any]:
    if not _verify_freeze(
        freeze,
        candidate=candidate,
        research_report=research_report,
        public_validation_path=public_validation_path,
    ):
        raise ConfirmationError("source or candidate changed before seed selection")
    if path.exists():
        raise FileExistsError(f"sealed seed manifest already exists: {path}")
    excluded = _all_public_seeds(curriculum).union(
        int(seed) for seed in freeze["excluded_prior_sealed_seeds"]
    )
    trials = int(freeze["protocol"]["trials"])
    rng = secrets.SystemRandom()
    seeds: set[int] = set()
    while len(seeds) < trials:
        seed = rng.randrange(SEED_MINIMUM, SEED_MAXIMUM)
        if seed not in excluded:
            seeds.add(seed)
    payload = {
        "format": "gum-school-recurrent-sealed-seeds-v2",
        "selected_at_utc": _utc_now(),
        "selected_after_source_candidate_and_protocol_freeze": True,
        "authority": "operating-system-entropy",
        "source_freeze_sha256": f"sha256:{hashlib.sha256(canonical(freeze)).hexdigest()}",
        "prior_revealed_seeds_excluded": True,
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
    candidate: Path,
    research_report: Path,
    public_validation_path: Path,
    curriculum: dict[str, Any],
) -> bool:
    payload = {key: value for key, value in manifest.items() if key != "commitment"}
    seeds = payload.get("seeds")
    excluded = _all_public_seeds(curriculum).union(
        int(seed) for seed in freeze["excluded_prior_sealed_seeds"]
    )
    return (
        _verify_freeze(
            freeze,
            candidate=candidate,
            research_report=research_report,
            public_validation_path=public_validation_path,
        )
        and manifest.get("commitment")
        == f"sha256:{hashlib.sha256(canonical(payload)).hexdigest()}"
        and payload.get("format") == "gum-school-recurrent-sealed-seeds-v2"
        and payload.get("selected_after_source_candidate_and_protocol_freeze") is True
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--school", type=Path, default=DEFAULT_SCHOOL)
    parser.add_argument("--causal-policy", type=Path, default=DEFAULT_CAUSAL)
    parser.add_argument("--research-report", type=Path, default=DEFAULT_RESEARCH_REPORT)
    parser.add_argument("--public-validation", type=Path, default=DEFAULT_PUBLIC_VALIDATION)
    parser.add_argument("--trials", type=int, default=64)
    args = parser.parse_args(argv)
    if args.trials < 32:
        raise SystemExit("sealed confirmation requires at least 32 trials")
    workspace = args.workspace.resolve()
    if workspace.exists():
        raise FileExistsError(f"sealed workspace already exists: {workspace}")
    if not _git_clean():
        raise ConfirmationError("commit the exact source, protocol, validation, and candidate first")

    research_path = args.research_report.resolve()
    validation_path = args.public_validation.resolve()
    research = json.loads(research_path.read_text(encoding="utf-8"))
    if research.get("strict_pass") is not True or research.get("sealed_data_used") is not False:
        raise ConfirmationError("candidate has no qualifying public-only development result")

    started = time.perf_counter()
    workspace.mkdir(parents=True)
    candidate_directory = workspace / "candidate"
    base_path = _promoted_state(args.school.resolve())
    CumulativeSchoolLearner.create_bundle(
        candidate_directory,
        base_path=base_path,
        causal_path=args.causal_policy.resolve(),
    )
    ledger = HashLedger(workspace / "CONFIRMATION_LEDGER.jsonl")
    ledger.append("candidate-bundled", {"hashes": _candidate_hashes(candidate_directory)})
    curriculum = json.loads(DEFAULT_CURRICULUM.read_text(encoding="utf-8"))
    lesson = next(
        row for row in curriculum["lessons"]
        if row["lesson_id"] == "causal-workshop.controls.001"
    )
    prior_lessons = [row for row in curriculum["lessons"] if row["sequence"] < 3]
    freeze = _create_freeze(
        workspace / "SOURCE_FREEZE.json",
        candidate=candidate_directory,
        research_report=research_path,
        public_validation_path=validation_path,
        trials=args.trials,
    )
    ledger.append("source-and-protocol-frozen", {
        "git_head": freeze["git_head"],
        "protocol_sha256": freeze["protocol_sha256"],
    })
    manifest = _select_manifest(
        workspace / "SEALED_SEED_MANIFEST.json",
        freeze=freeze,
        candidate=candidate_directory,
        research_report=research_path,
        public_validation_path=validation_path,
        curriculum=curriculum,
    )
    ledger.append("fresh-sealed-seeds-selected", {"commitment": manifest["commitment"]})
    protocol_verified = _verify_manifest(
        manifest,
        freeze=freeze,
        candidate=candidate_directory,
        research_report=research_path,
        public_validation_path=validation_path,
        curriculum=curriculum,
    )
    if not protocol_verified:
        raise ConfirmationError("sealed manifest verification failed")

    worlds_root = workspace / "sealed-worlds"
    worlds_root.mkdir()
    candidate = CumulativeSchoolLearner.load_bundle(candidate_directory)
    fresh = RecurrentCausalLearner(candidate.causal.seed, config=candidate.causal.config)
    candidate_rows = []
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
            replay_package = package
            replay_seed = seed
    if replay_package is None or replay_seed is None:
        raise ConfirmationError("no replay trial was selected")
    replay = _run_episode(
        CumulativeSchoolLearner.load_bundle(candidate_directory),
        _load_sealed_world(replay_package, lesson, seed=replay_seed),
        seed=replay_seed,
        training=False,
        interaction_limit=11,
    )
    replay_verified = canonical(replay) == canonical(candidate_rows[0])

    candidate_summary = _summarize(candidate_rows)
    fresh_summary = _summarize(fresh_rows)
    random_summary = _summarize(random_rows)
    symmetry_summary = summarize_symmetry_aware_uncertainty(candidate_rows)
    symmetry_gate = evaluate_symmetry_uncertainty_gate(
        symmetry_summary,
        maximum_post_causal_uncertainty_rate=float(
            lesson["promotion"]["maximum_uncertainty_rate"]
        ),
    )
    success_lower, success_upper = _wilson(
        candidate_summary["successes"], candidate_summary["trials"]
    )
    advantage_lower, advantage_upper = _paired_advantage_interval(
        [row["success"] for row in candidate_rows],
        [row["success"] for row in fresh_rows],
    )
    retention_rows, retention_trials, retention = _retention(
        candidate_directory,
        base_path,
        current_lesson=lesson,
        prior_lessons=prior_lessons,
        root=workspace / "retention-worlds",
    )

    trajectories = {
        "format": "gum-school-recurrent-sealed-trajectories-v2",
        "candidate": candidate_rows,
        "matched_fresh": fresh_rows,
        "random": random_rows,
        "retention": retention,
    }
    trajectory_path = workspace / "TRAJECTORIES.json"
    atomic_write_json(trajectory_path, trajectories, backup=False, sort_keys=True)
    trajectory_sha256 = f"sha256:{file_sha256(trajectory_path)}"
    trajectory_verified = (
        json.loads(trajectory_path.read_text(encoding="utf-8")) == trajectories
        and trajectory_sha256 == f"sha256:{file_sha256(trajectory_path)}"
    )
    replay_path = workspace / "REPLAY.json"
    atomic_write_json(
        replay_path,
        {"format": "gum-school-recurrent-replay-v2", "expected": candidate_rows[0], "replayed": replay},
        backup=False,
        sort_keys=True,
    )
    ledger.append("evaluation-completed", {
        "candidate_successes": candidate_summary["successes"],
        "candidate_trials": candidate_summary["trials"],
        "post_causal_uncertainty_rate": symmetry_summary[
            "post_causal_evidence_uncertainty_rate"
        ],
        "trajectory_sha256": trajectory_sha256,
        "replay_verified": replay_verified,
    })
    ledger_valid = ledger.verify()["valid"]
    total_evaluation_interactions = sum(
        row["interactions"]
        for row in candidate_rows + fresh_rows + random_rows + retention_trials + [replay]
    )
    artifact_mb = sum(
        path.stat().st_size for path in workspace.rglob("*") if path.is_file()
    ) / (1024.0 * 1024.0)
    evaluation = {
        "format": "gum-school-evaluation-v1",
        "sealed": {
            "protocol_verified": protocol_verified,
            "trials": candidate_summary["trials"],
            "success_rate": candidate_summary["success_rate"],
            "fresh_success_rate": fresh_summary["success_rate"],
            "uncertainty_rate": symmetry_summary[
                "post_causal_evidence_uncertainty_rate"
            ],
            "unnecessary_action_rate": candidate_summary["unnecessary_action_rate"],
            "same_perception": True,
            "same_resource_limits": True,
            "confidence_level": 0.95,
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
        "retention": retention_rows,
        "evidence": {
            "source_hashes_verified": _verify_freeze(
                freeze,
                candidate=candidate_directory,
                research_report=research_path,
                public_validation_path=validation_path,
            ),
            "protocol_hashes_verified": protocol_verified,
            "trajectory_verified": trajectory_verified,
            "ledger_verified": ledger_valid,
        },
        "resources": {
            "wall_clock_seconds": max(0.001, time.perf_counter() - started),
            "peak_memory_mb": psutil.Process().memory_info().rss / (1024.0 * 1024.0),
            "artifact_storage_mb": artifact_mb,
            "interactions": int(research["training"]["interactions"]),
        },
        "input_boundary": {
            "verified": not candidate.boundary_violations,
            "violations": candidate.boundary_violations,
        },
        "replay": {
            "verified": replay_verified,
            "deterministic": replay_verified,
            "artifact_path": replay_path.name,
            "artifact_sha256": f"sha256:{file_sha256(replay_path)}",
        },
        "transfer_results": [],
    }
    required_skills = {
        skill for prior in prior_lessons for skill in prior["capabilities"]
    }
    standard_gates = evaluate_promotion_gates(
        lesson,
        evaluation,
        required_protected_skills=required_skills,
    )
    all_passed = bool(standard_gates["all_passed"] and symmetry_gate["passed"])
    gates = {
        "format": "gum-school-symmetry-aware-gate-result-v2",
        "all_passed": all_passed,
        "standard_curriculum_gates": standard_gates,
        "symmetry_uncertainty_gate": symmetry_gate,
    }
    report = {
        "format": "gum-school-recurrent-sealed-confirmation-v2",
        "created_at_utc": _utc_now(),
        "official_curriculum_promotion": False,
        "supplemental_symmetry_aware_qualification": True,
        "development_only": False,
        "sealed_confirmation": True,
        "sealed_data_used_for_training": False,
        "policy_weights_changed_after_public_validation": False,
        "promoted_workspace_mutated": False,
        "prior_failed_confirmation_preserved": True,
        "lesson_id": lesson["lesson_id"],
        "source_freeze": freeze,
        "seed_manifest": manifest,
        "candidate": candidate_summary,
        "matched_fresh": fresh_summary,
        "uniform_random": random_summary,
        "symmetry_aware_uncertainty": symmetry_summary,
        "success_interval": evaluation["sealed"]["success_interval"],
        "retention": retention,
        "evaluation": evaluation,
        "gates": gates,
        "training_interactions": int(research["training"]["interactions"]),
        "evaluation_interactions": total_evaluation_interactions,
        "candidate_bundle_hashes": _candidate_hashes(candidate_directory),
        "qualification": "pass" if all_passed else "fail",
    }
    report_path = workspace / REPORT_FILENAME
    atomic_write_json(report_path, report, backup=False, sort_keys=True)
    shutil.rmtree(worlds_root)
    shutil.rmtree(workspace / "retention-worlds")
    print(json.dumps({
        "qualification": report["qualification"],
        "all_gates_passed": all_passed,
        "candidate_success_rate": candidate_summary["success_rate"],
        "matched_fresh_success_rate": fresh_summary["success_rate"],
        "random_success_rate": random_summary["success_rate"],
        "initial_uncertainty_rate": symmetry_summary["initial_uncertainty_rate"],
        "post_causal_uncertainty_rate": symmetry_summary[
            "post_causal_evidence_uncertainty_rate"
        ],
        "retention_success_rate": retention["current_summary"]["success_rate"],
        "report": str(report_path),
    }, indent=2, sort_keys=True))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
