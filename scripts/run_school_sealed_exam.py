from __future__ import annotations

import argparse
import json
from pathlib import Path

from gum.school.sealed import SealedExamConfig, run_official_sealed_exam
from gum.school.training import TrainingLaneConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the post-freeze sealed exam for the next GUM School lesson."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, default=None)
    parser.add_argument("--learner-seed", type=int, default=8_620_001)
    parser.add_argument("--max-training-interactions", type=int, default=600)
    parser.add_argument("--training-episodes-per-seed", type=int, default=8)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--max-replicas", type=int, default=4)
    parser.add_argument(
        "--continue-existing", action="store_true",
        help="Run the next lesson in an existing promoted school workspace.",
    )
    args = parser.parse_args()
    config = SealedExamConfig(
        training=TrainingLaneConfig(
            max_training_interactions=args.max_training_interactions,
            training_episodes_per_seed=args.training_episodes_per_seed,
            development_trials=32,
            max_replicas=args.max_replicas,
        ),
        trials=args.trials,
    )
    kwargs = {
        "workspace": args.workspace,
        "config": config,
        "learner_seed": args.learner_seed,
        "continue_existing": args.continue_existing,
    }
    if args.curriculum is not None:
        kwargs["curriculum_path"] = args.curriculum
    report = run_official_sealed_exam(**kwargs)
    result = report["result"]
    sequence = int(report["lesson_id"].rsplit(".", 1)[-1])
    report_name = (
        "SEALED_EXAM_REPORT.json"
        if sequence == 1
        else f"SEALED_EXAM_REPORT_{sequence:03d}.json"
    )
    print(json.dumps({
        "outcome": result["outcome"],
        "all_gates_passed": result["all_gates_passed"],
        "candidate_success_rate": result["candidate_success_rate"],
        "matched_fresh_success_rate": result["matched_fresh_success_rate"],
        "random_success_rate": result["random_success_rate"],
        "promoted_snapshot_changed": result["promoted_snapshot_changed"],
        "report": str(args.workspace / report_name),
    }, indent=2, sort_keys=True))
    return 0 if result["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
