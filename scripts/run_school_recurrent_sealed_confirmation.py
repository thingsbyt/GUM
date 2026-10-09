"""Run a post-freeze sealed confirmation of the cumulative recurrent learner.

This supplemental protocol never mutates the promoted GUM School workspace.
It qualifies (or rejects) an integrity-checked composite candidate, including
matched-fresh/random controls, exact replay, and Lessons 1-2 retention.
"""
from __future__ import annotations

import argparse
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
from gum.school.cumulative import (
    BASE_FILENAME,
    BUNDLE_FILENAME,
    CAUSAL_FILENAME,
    CumulativeSchoolLearner,
)
from gum.school.evaluation import evaluate_promotion_gates
from gum.school.learner import CrossSeedSchoolLearner
from gum.school.recurrent_meta import RecurrentCausalLearner
from gum.school.sealed import (
    CAUSAL_CONTROL_SHIFT,
    SEED_MAXIMUM,
    SEED_MINIMUM,
    _create_sealed_world,
    _load_sealed_world,
    _run_random_episode,
)
from gum.school.training import (
    _paired_advantage_interval,
    _run_episode,
    _summarize,
    _wilson,
)
from gum.school.validation import DEFAULT_CURRICULUM
from gum.school.worlds import FOUNDATIONAL_LOADERS, create_lesson_world
from gum.storage import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHOOL = (
    ROOT / "evidence" / "gum-school" / "sealed" / "object-laboratory-occlusion-v1"
)
DEFAULT_CAUSAL = (
    ROOT / "evidence" / "gum-school" / "research" / "causal-recurrent-meta-v1"
    / "RECURRENT_META_POLICY.pt"
)
DEFAULT_RESEARCH_REPORT = DEFAULT_CAUSAL.with_name("RECURRENT_META_REPORT.json")
REPORT_FILENAME = "SEALED_RECURRENT_CONFIRMATION.json"


class ConfirmationError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    head = result.stdout.strip()
    if len(head) != 40:
        raise ConfirmationError("git did not return a full source revision")
    return head


def _git_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    return not result.stdout.strip()


def _source_paths() -> list[Path]:
    paths = [
        ROOT / "curriculum" / "gum-school-v1.json",
        ROOT / "gum" / "lineage.py",
        ROOT / "gum" / "protocol.py",
        ROOT / "gum" / "storage.py",
        ROOT / "gum" / "school" / "cumulative.py",
        ROOT / "gum" / "school" / "evaluation.py",
        ROOT / "gum" / "school" / "learner.py",
        ROOT / "gum" / "school" / "recurrent_meta.py",
        ROOT / "gum" / "school" / "sealed.py",
        ROOT / "gum" / "school" / "training.py",
        ROOT / "gum" / "school" / "worlds.py",
        Path(__file__).resolve(),
    ]
    paths.extend(sorted((ROOT / "curriculum" / "schemas").glob("gum-school-*.schema.json")))
    return paths


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): f"sha256:{file_sha256(path)}"
        for path in paths
    }


def _candidate_hashes(directory: Path) -> dict[str, str]:
    return {
        filename: f"sha256:{file_sha256(Path(directory) / filename)}"
        for filename in (BASE_FILENAME, CAUSAL_FILENAME, BUNDLE_FILENAME)
    }


def _create_freeze(
    path: Path,
    *,
    candidate: Path,
    research_report: Path,
    trials: int,
) -> dict[str, Any]:
    protocol = {
        "format": "gum-school-recurrent-sealed-protocol-v1",
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
    }
    value = {
        "format": "gum-school-recurrent-source-freeze-v1",
        "created_at_utc": _utc_now(),
        "git_head": _git_head(),
        "git_clean_before_workspace_creation": True,
        "source_hashes": _hashes(_source_paths()),
        "candidate_hashes": _candidate_hashes(candidate),
        "public_training_evidence_sha256": f"sha256:{file_sha256(research_report)}",
        "protocol": protocol,
        "protocol_sha256": f"sha256:{hashlib.sha256(canonical(protocol)).hexdigest()}",
    }
    atomic_write_json(path, value, backup=False, sort_keys=True)
    return value


def _verify_freeze(
    freeze: dict[str, Any], *, candidate: Path, research_report: Path
) -> bool:
    protocol_hash = f"sha256:{hashlib.sha256(canonical(freeze.get('protocol'))).hexdigest()}"
    return (
        freeze.get("format") == "gum-school-recurrent-source-freeze-v1"
        and freeze.get("git_clean_before_workspace_creation") is True
        and freeze.get("source_hashes") == _hashes(_source_paths())
        and freeze.get("candidate_hashes") == _candidate_hashes(candidate)
        and freeze.get("public_training_evidence_sha256")
        == f"sha256:{file_sha256(research_report)}"
        and freeze.get("protocol_sha256") == protocol_hash
        and freeze.get("git_head") == _git_head()
    )


def _all_public_seeds(curriculum: dict[str, Any]) -> set[int]:
    seeds: set[int] = set()
    for lesson in curriculum["lessons"]:
        seeds.update(int(value) for value in lesson["training"]["seeds"])
        seeds.update(int(value) for value in lesson["evaluation"]["development_seeds"])
        seeds.update(int(value) for value in lesson["retention"]["evaluation_seeds"])
    return seeds


def _select_manifest(
    path: Path,
    *,
    freeze: dict[str, Any],
    candidate: Path,
    research_report: Path,
    curriculum: dict[str, Any],
) -> dict[str, Any]:
    if not _verify_freeze(
        freeze, candidate=candidate, research_report=research_report
    ):
        raise ConfirmationError("source or candidate changed before seed selection")
    if path.exists():
        raise FileExistsError(f"sealed seed manifest already exists: {path}")
    public = _all_public_seeds(curriculum)
    trials = int(freeze["protocol"]["trials"])
    rng = secrets.SystemRandom()
    seeds: set[int] = set()
    while len(seeds) < trials:
        seed = rng.randrange(SEED_MINIMUM, SEED_MAXIMUM)
        if seed not in public:
            seeds.add(seed)
    payload = {
        "format": "gum-school-recurrent-sealed-seeds-v1",
        "selected_at_utc": _utc_now(),
        "selected_after_source_and_candidate_freeze": True,
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
    candidate: Path,
    research_report: Path,
    curriculum: dict[str, Any],
) -> bool:
    payload = {key: value for key, value in manifest.items() if key != "commitment"}
    seeds = payload.get("seeds")
    return (
        _verify_freeze(
            freeze, candidate=candidate, research_report=research_report
        )
        and manifest.get("commitment")
        == f"sha256:{hashlib.sha256(canonical(payload)).hexdigest()}"
        and payload.get("format") == "gum-school-recurrent-sealed-seeds-v1"
        and payload.get("selected_after_source_and_candidate_freeze") is True
        and payload.get("source_freeze_sha256")
        == f"sha256:{hashlib.sha256(canonical(freeze)).hexdigest()}"
        and isinstance(seeds, list)
        and len(seeds) == freeze["protocol"]["trials"]
        and len(set(seeds)) == len(seeds)
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
        and all(SEED_MINIMUM <= seed < SEED_MAXIMUM for seed in seeds)
        and not _all_public_seeds(curriculum).intersection(seeds)
    )


def _promoted_state(school: Path) -> Path:
    pointer = json.loads((school / "PROMOTED.json").read_text(encoding="utf-8"))
    return school / "snapshots" / pointer["snapshot_id"] / "state" / BASE_FILENAME


def _retention(
    candidate_directory: Path,
    base_path: Path,
    *,
    current_lesson: dict[str, Any],
    prior_lessons: list[dict[str, Any]],
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evidence_rows = []
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
                baseline, baseline_world, seed=seed, training=False,
                interaction_limit=baseline_world.public_spec().horizon,
            ))
            current_rows.append(_run_episode(
                current, current_world, seed=seed, training=False,
                interaction_limit=current_world.public_spec().horizon,
            ))
        baseline_summary = _summarize(baseline_rows)
        current_summary = _summarize(current_rows)
        baseline_all.extend(baseline_rows)
        current_all.extend(current_rows)
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
    return evidence_rows, baseline_all + current_all, {
        "by_lesson": by_lesson,
        "baseline_summary": _summarize(baseline_all),
        "current_summary": _summarize(current_all),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--school", type=Path, default=DEFAULT_SCHOOL)
    parser.add_argument("--causal-policy", type=Path, default=DEFAULT_CAUSAL)
    parser.add_argument("--research-report", type=Path, default=DEFAULT_RESEARCH_REPORT)
    parser.add_argument("--trials", type=int, default=64)
    args = parser.parse_args(argv)
    if args.trials < 32:
        raise SystemExit("sealed confirmation requires at least 32 trials")
    workspace = args.workspace.resolve()
    if workspace.exists():
        raise FileExistsError(f"sealed workspace already exists: {workspace}")
    if not _git_clean():
        raise ConfirmationError("commit the exact source and candidate before sealed confirmation")
    research = json.loads(args.research_report.read_text(encoding="utf-8"))
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
        item for item in curriculum["lessons"]
        if item["lesson_id"] == "causal-workshop.controls.001"
    )
    prior_lessons = [item for item in curriculum["lessons"] if item["sequence"] < 3]
    freeze = _create_freeze(
        workspace / "SOURCE_FREEZE.json",
        candidate=candidate_directory,
        research_report=args.research_report.resolve(),
        trials=args.trials,
    )
    ledger.append("source-frozen", {
        "git_head": freeze["git_head"],
        "protocol_sha256": freeze["protocol_sha256"],
    })
    manifest = _select_manifest(
        workspace / "SEALED_SEED_MANIFEST.json",
        freeze=freeze,
        candidate=candidate_directory,
        research_report=args.research_report.resolve(),
        curriculum=curriculum,
    )
    ledger.append("sealed-seeds-selected", {"commitment": manifest["commitment"]})
    if not _verify_manifest(
        manifest,
        freeze=freeze,
        candidate=candidate_directory,
        research_report=args.research_report.resolve(),
        curriculum=curriculum,
    ):
        raise ConfirmationError("sealed manifest verification failed")

    worlds_root = workspace / "sealed-worlds"
    worlds_root.mkdir()
    candidate = CumulativeSchoolLearner.load_bundle(candidate_directory)
    fresh = RecurrentCausalLearner(
        candidate.causal.seed, config=candidate.causal.config
    )
    candidate_rows = []
    fresh_rows = []
    random_rows = []
    replay_package = None
    replay_seed = None
    for index, seed in enumerate(manifest["seeds"]):
        package = _create_sealed_world(worlds_root, lesson, seed=seed)
        candidate_world = _load_sealed_world(package, lesson, seed=seed)
        fresh_world = _load_sealed_world(package, lesson, seed=seed)
        random_world = _load_sealed_world(package, lesson, seed=seed)
        candidate_rows.append(_run_episode(
            candidate, candidate_world, seed=seed, training=False, interaction_limit=11
        ))
        fresh_rows.append(_run_episode(
            fresh, fresh_world, seed=seed, training=False, interaction_limit=11
        ))
        random_rows.append(_run_random_episode(
            random_world,
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
        "format": "gum-school-recurrent-sealed-trajectories-v1",
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
        {"format": "gum-school-recurrent-replay-v1", "expected": candidate_rows[0], "replayed": replay},
        backup=False,
        sort_keys=True,
    )
    ledger.append("evaluation-completed", {
        "candidate_successes": candidate_summary["successes"],
        "candidate_trials": candidate_summary["trials"],
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
            "protocol_verified": _verify_manifest(
                manifest,
                freeze=freeze,
                candidate=candidate_directory,
                research_report=args.research_report.resolve(),
                curriculum=curriculum,
            ),
            "trials": candidate_summary["trials"],
            "success_rate": candidate_summary["success_rate"],
            "fresh_success_rate": fresh_summary["success_rate"],
            "uncertainty_rate": candidate_summary["uncertainty_rate"],
            "unnecessary_action_rate": candidate_summary["unnecessary_action_rate"],
            "same_perception": True,
            "same_resource_limits": True,
            "confidence_level": 0.95,
            "success_interval": {
                "lower": success_lower, "upper": success_upper, "method": "wilson"
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
                research_report=args.research_report.resolve(),
            ),
            "protocol_hashes_verified": _verify_manifest(
                manifest,
                freeze=freeze,
                candidate=candidate_directory,
                research_report=args.research_report.resolve(),
                curriculum=curriculum,
            ),
            "trajectory_verified": trajectory_verified,
            "ledger_verified": ledger_valid,
        },
        "resources": {
            "wall_clock_seconds": max(0.001, time.perf_counter() - started),
            "peak_memory_mb": psutil.Process().memory_info().rss / (1024.0 * 1024.0),
            "artifact_storage_mb": artifact_mb,
            # Evaluation is frozen and non-learning; the resource gate counts
            # only interactions that changed the candidate.
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
    gates = evaluate_promotion_gates(
        lesson, evaluation, required_protected_skills=required_skills
    )
    report = {
        "format": "gum-school-recurrent-sealed-confirmation-v1",
        "created_at_utc": _utc_now(),
        "official_curriculum_promotion": False,
        "development_only": False,
        "sealed_confirmation": True,
        "sealed_data_used_for_training": False,
        "promoted_workspace_mutated": False,
        "lesson_id": lesson["lesson_id"],
        "source_freeze": freeze,
        "seed_manifest": manifest,
        "candidate": candidate_summary,
        "matched_fresh": fresh_summary,
        "uniform_random": random_summary,
        "success_interval": evaluation["sealed"]["success_interval"],
        "retention": retention,
        "evaluation": evaluation,
        "gates": gates,
        "training_interactions": int(research["training"]["interactions"]),
        "evaluation_interactions": total_evaluation_interactions,
        "candidate_bundle_hashes": _candidate_hashes(candidate_directory),
        "qualification": "pass" if gates["all_passed"] else "fail",
    }
    report_path = workspace / REPORT_FILENAME
    atomic_write_json(report_path, report, backup=False, sort_keys=True)
    shutil.rmtree(worlds_root)
    shutil.rmtree(workspace / "retention-worlds")
    print(json.dumps({
        "qualification": report["qualification"],
        "all_gates_passed": gates["all_passed"],
        "candidate_success_rate": candidate_summary["success_rate"],
        "matched_fresh_success_rate": fresh_summary["success_rate"],
        "random_success_rate": random_summary["success_rate"],
        "retention_success_rate": retention["current_summary"]["success_rate"],
        "report": str(report_path),
    }, indent=2, sort_keys=True))
    return 0 if gates["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
