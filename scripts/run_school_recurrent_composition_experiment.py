"""Continue the promoted recurrent policy on public causal-composition worlds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from gum.lineage import file_sha256
from gum.school.cumulative import CAUSAL_FILENAME, CumulativeSchoolLearner
from gum.school.recurrent_meta import RecurrentCausalLearner
from gum.school.training import _wilson
from gum.school.validation import DEFAULT_CURRICULUM
from gum.storage import atomic_write_json
if __package__:
    from scripts.run_school_recurrent_meta_experiment import (
        RandomCausalPolicy,
        _evaluate,
        _train,
    )
else:
    from run_school_recurrent_meta_experiment import (
        RandomCausalPolicy,
        _evaluate,
        _train,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_WORKSPACE = (
    ROOT / "evidence" / "gum-school" / "sealed-recurrent-official-v2"
)
DEFAULT_OUTPUT = (
    ROOT / "evidence" / "gum-school" / "research"
    / "causal-composition-recurrent-v1" / "COMPOSITION_REPORT.json"
)


def _promoted_bundle(workspace: Path) -> Path:
    pointer = json.loads((workspace / "PROMOTED.json").read_text(encoding="utf-8"))
    return workspace / "snapshots" / pointer["snapshot_id"] / "state"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-workspace", type=Path, default=DEFAULT_OFFICIAL_WORKSPACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work", type=Path, default=ROOT / ".test-temp" / "composition-recurrent")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--control-retention-trials", type=int, default=256)
    parser.add_argument("--episode-limit", type=int, default=32)
    parser.add_argument("--self-imitation-epochs", type=int, default=100)
    args = parser.parse_args(argv)
    if min(
        args.trials,
        args.control_retention_trials,
        args.episode_limit,
        args.self_imitation_epochs,
    ) < 1:
        raise SystemExit("trials, limits, and epochs must be positive")

    curriculum = json.loads(DEFAULT_CURRICULUM.read_text(encoding="utf-8"))
    composition = next(
        lesson for lesson in curriculum["lessons"]
        if lesson["lesson_id"] == "causal-workshop.composition.002"
    )
    controls = next(
        lesson for lesson in curriculum["lessons"]
        if lesson["lesson_id"] == "causal-workshop.controls.001"
    )
    budget = int(
        composition["training"]["interaction_budget"]
        if args.budget is None else args.budget
    )
    if budget < 1 or budget > composition["training"]["interaction_budget"]:
        raise SystemExit("budget must be within the public lesson budget")

    promoted_directory = _promoted_bundle(args.official_workspace.resolve())
    CumulativeSchoolLearner.load_bundle(promoted_directory)
    learner = RecurrentCausalLearner.load(promoted_directory / CAUSAL_FILENAME)
    initial = RecurrentCausalLearner.load(promoted_directory / CAUSAL_FILENAME)
    fresh = RecurrentCausalLearner(learner.seed, config=learner.config)
    work = args.work.resolve()
    if work.exists():
        shutil.rmtree(work)
    for name in (
        "train",
        "evaluate-trained",
        "evaluate-promoted-start",
        "evaluate-fresh",
        "evaluate-random",
        "retention-controls",
    ):
        (work / name).mkdir(parents=True, exist_ok=True)
    try:
        training = _train(
            learner,
            composition,
            root=work / "train",
            budget=budget,
            episode_limit=args.episode_limit,
            self_imitation_epochs=args.self_imitation_epochs,
        )
        trained = _evaluate(
            learner,
            composition,
            root=work / "evaluate-trained",
            trials=args.trials,
            episode_limit=args.episode_limit,
        )
        promoted_start = _evaluate(
            initial,
            composition,
            root=work / "evaluate-promoted-start",
            trials=args.trials,
            episode_limit=args.episode_limit,
        )
        untrained = _evaluate(
            fresh,
            composition,
            root=work / "evaluate-fresh",
            trials=args.trials,
            episode_limit=args.episode_limit,
        )
        random = _evaluate(
            RandomCausalPolicy(learner.seed),
            composition,
            root=work / "evaluate-random",
            trials=args.trials,
            episode_limit=args.episode_limit,
        )
        controls_retention = _evaluate(
            learner,
            controls,
            root=work / "retention-controls",
            trials=args.control_retention_trials,
            episode_limit=11,
        )
    finally:
        if work.exists():
            shutil.rmtree(work)

    controls_lower, controls_upper = _wilson(
        controls_retention["successes"], controls_retention["trials"]
    )
    controls_retention["success_interval"] = {
        "lower": controls_lower,
        "upper": controls_upper,
        "method": "wilson",
    }
    minimum_success = float(composition["promotion"]["minimum_success"])
    minimum_lower = float(composition["promotion"]["minimum_success_lower_bound"])
    strict_pass = (
        trained["success_rate"] >= minimum_success
        and trained["success_interval"]["lower"] >= minimum_lower
        and trained["success_rate"] > promoted_start["success_rate"]
        and controls_retention["success_rate"] >= minimum_success
        and controls_lower >= minimum_lower
    )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    policy_path = args.output.resolve().with_name("COMPOSITION_POLICY.pt")
    learner.save(policy_path)
    report = {
        "format": "gum-school-recurrent-composition-experiment-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_curriculum_run": False,
        "development_only": True,
        "sealed_data_used": False,
        "lesson_id": composition["lesson_id"],
        "starting_official_snapshot": json.loads(
            (args.official_workspace.resolve() / "PROMOTED.json").read_text(encoding="utf-8")
        )["snapshot_id"],
        "method": {
            "family": "continued-recurrent-model-free-meta-reinforcement-learning",
            "information_boundary": learner.status()["information_boundary"],
            "forbidden_inputs": [
                "public_info.event",
                "hidden_world_state",
                "advance_action_sequence",
                "tried_action_mask",
                "prescribed_probe_order",
            ],
            "episode_limit": args.episode_limit,
            "policy_weights_updated": True,
            "base_object_laboratory_component_updated": False,
        },
        "public_training_seeds": composition["training"]["seeds"],
        "public_development_seeds": composition["evaluation"]["development_seeds"],
        "training": training,
        "evaluation": {
            "trained_composition_policy": trained,
            "promoted_controls_policy_before_composition_training": promoted_start,
            "same_untrained_network": untrained,
            "uniform_random_policy": random,
            "controls_lesson_retention": controls_retention,
        },
        "strict_thresholds": {
            "composition_success_rate": minimum_success,
            "composition_wilson_lower_bound": minimum_lower,
            "controls_retention_success_rate": minimum_success,
            "controls_retention_wilson_lower_bound": minimum_lower,
            "maximum_actions_per_trial": args.episode_limit,
            "training_interaction_budget": budget,
        },
        "strict_pass": strict_pass,
        "source_hashes": {
            "starting_policy": f"sha256:{file_sha256(promoted_directory / CAUSAL_FILENAME)}",
            "trained_policy": f"sha256:{file_sha256(policy_path)}",
            "recurrent_meta": f"sha256:{file_sha256(ROOT / 'gum/school/recurrent_meta.py')}",
            "worlds": f"sha256:{file_sha256(ROOT / 'gum/school/worlds.py')}",
            "experiment": f"sha256:{file_sha256(Path(__file__))}",
        },
        "interpretation": (
            "The continued recurrent policy met the public composition and controls-retention criteria."
            if strict_pass else
            "The continued recurrent policy did not meet all public composition and retention criteria."
        ),
    }
    atomic_write_json(args.output.resolve(), report, backup=False, sort_keys=True)
    print(json.dumps({
        "strict_pass": strict_pass,
        "training_success_rate": training["success_rate"],
        "composition_success_rate": trained["success_rate"],
        "promoted_start_success_rate": promoted_start["success_rate"],
        "fresh_success_rate": untrained["success_rate"],
        "random_success_rate": random["success_rate"],
        "controls_retention_success_rate": controls_retention["success_rate"],
        "report": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if strict_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
