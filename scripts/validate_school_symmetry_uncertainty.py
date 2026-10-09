"""Validate the symmetry-aware uncertainty protocol on public development worlds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from gum.lineage import file_sha256
from gum.school.recurrent_meta import RecurrentCausalLearner
from gum.school.symmetry_uncertainty import (
    MINIMUM_INITIAL_UNCERTAINTY_RATE,
    evaluate_symmetry_uncertainty_gate,
    summarize_symmetry_aware_uncertainty,
)
from gum.school.training import _run_episode, _summarize, _trial_seeds
from gum.school.validation import DEFAULT_CURRICULUM
from gum.school.worlds import FOUNDATIONAL_LOADERS, create_lesson_world
from gum.storage import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = (
    ROOT / "evidence" / "gum-school" / "research" / "causal-recurrent-meta-v1"
    / "RECURRENT_META_POLICY.pt"
)
DEFAULT_PRIOR_REPORT = DEFAULT_POLICY.with_name("RECURRENT_META_REPORT.json")
DEFAULT_OUTPUT = (
    ROOT / "evidence" / "gum-school" / "research" / "causal-recurrent-meta-v2"
    / "SYMMETRY_UNCERTAINTY_VALIDATION.json"
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--prior-report", type=Path, default=DEFAULT_PRIOR_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work", type=Path, default=ROOT / ".test-temp" / "symmetry-validation")
    parser.add_argument("--trials", type=int, default=512)
    args = parser.parse_args(argv)
    if args.trials < 32:
        raise SystemExit("public validation requires at least 32 trials")

    curriculum = json.loads(DEFAULT_CURRICULUM.read_text(encoding="utf-8"))
    lesson = next(
        row for row in curriculum["lessons"]
        if row["lesson_id"] == "causal-workshop.controls.001"
    )
    prior = json.loads(args.prior_report.read_text(encoding="utf-8"))
    if prior.get("strict_pass") is not True or prior.get("sealed_data_used") is not False:
        raise RuntimeError("the frozen policy lacks qualifying public-only training evidence")

    work = args.work.resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    learner = RecurrentCausalLearner.load(args.policy.resolve())
    rows = []
    seeds = _trial_seeds(lesson["evaluation"]["development_seeds"], args.trials)
    try:
        for index, seed in enumerate(seeds):
            package = create_lesson_world(work / f"trial-{index:04d}", lesson, seed=seed)
            world = FOUNDATIONAL_LOADERS[lesson["adapter"]](package)
            rows.append(_run_episode(
                learner,
                world,
                seed=seed,
                training=False,
                interaction_limit=11,
            ))
    finally:
        if work.exists():
            shutil.rmtree(work)

    behavioral = _summarize(rows)
    uncertainty = summarize_symmetry_aware_uncertainty(rows)
    gate = evaluate_symmetry_uncertainty_gate(
        uncertainty,
        maximum_post_causal_uncertainty_rate=float(
            lesson["promotion"]["maximum_uncertainty_rate"]
        ),
    )
    evidence_directory = args.output.resolve().parent
    evidence_directory.mkdir(parents=True, exist_ok=True)
    trajectories_path = evidence_directory / "PUBLIC_TRAJECTORIES.json"
    trajectories = {
        "format": "gum-school-public-symmetry-trajectories-v1",
        "public_development_seeds": lesson["evaluation"]["development_seeds"],
        "trials": rows,
    }
    atomic_write_json(trajectories_path, trajectories, backup=False, sort_keys=True)
    report = {
        "format": "gum-school-symmetry-uncertainty-validation-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "sealed_data_used": False,
        "policy_weights_changed": False,
        "lesson_id": lesson["lesson_id"],
        "trials": args.trials,
        "public_development_seeds": lesson["evaluation"]["development_seeds"],
        "behavioral_summary": behavioral,
        "symmetry_aware_uncertainty": uncertainty,
        "symmetry_uncertainty_gate": gate,
        "protocol": {
            "principle": (
                "Require uncertainty before indistinguishable controls have evidence, "
                "then require confidence after an earlier action has produced positive "
                "scalar reward."
            ),
            "positive_causal_evidence": "an earlier action produced scalar reward > 0",
            "current_action_consequence_is_prior_evidence": False,
            "event_labels_used": False,
            "hidden_state_used": False,
            "minimum_initial_uncertainty_rate": MINIMUM_INITIAL_UNCERTAINTY_RATE,
            "maximum_post_causal_uncertainty_rate": lesson["promotion"][
                "maximum_uncertainty_rate"
            ],
        },
        "artifact_hashes": {
            "policy": f"sha256:{file_sha256(args.policy.resolve())}",
            "prior_public_training_report": (
                f"sha256:{file_sha256(args.prior_report.resolve())}"
            ),
            "public_trajectories": f"sha256:{file_sha256(trajectories_path)}",
            "metric_source": (
                f"sha256:{file_sha256(ROOT / 'gum/school/symmetry_uncertainty.py')}"
            ),
            "validation_source": f"sha256:{file_sha256(Path(__file__))}",
        },
        "validation_pass": gate["passed"],
    }
    atomic_write_json(args.output.resolve(), report, backup=False, sort_keys=True)
    print(json.dumps({
        "validation_pass": report["validation_pass"],
        "success_rate": behavioral["success_rate"],
        "initial_uncertainty_rate": uncertainty["initial_uncertainty_rate"],
        "post_causal_uncertainty_rate": uncertainty[
            "post_causal_evidence_uncertainty_rate"
        ],
        "confidence_gain": uncertainty["confidence_gain_after_causal_evidence"],
        "report": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if report["validation_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
