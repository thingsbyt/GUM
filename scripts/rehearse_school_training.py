#!/usr/bin/env python3
"""Run the bounded development-only GUM School training rehearsal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gum.lineage import file_sha256
from gum.school.rehearsal import run_nonpromoting_rehearsal
from gum.school.training import TrainingLaneConfig


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=ROOT / "evidence" / "gum-school" / "rehearsal" / "foundational-lane-v1",
    )
    parser.add_argument("--max-training-interactions", type=int, default=300)
    parser.add_argument("--training-episodes-per-seed", type=int, default=4)
    parser.add_argument("--development-trials", type=int, default=32)
    parser.add_argument("--max-replicas", type=int, default=4)
    parser.add_argument("--learner-seed", type=int, default=8_620_001)
    args = parser.parse_args()
    report = run_nonpromoting_rehearsal(
        args.workspace,
        config=TrainingLaneConfig(
            max_training_interactions=args.max_training_interactions,
            training_episodes_per_seed=args.training_episodes_per_seed,
            development_trials=args.development_trials,
            max_replicas=args.max_replicas,
        ),
        learner_seed=args.learner_seed,
    )
    report_path = args.workspace / "REHEARSAL_REPORT.json"
    print(json.dumps({
        "workspace": str(args.workspace),
        "outcome": report["result"]["outcome"],
        "development_learning_observed": report["result"]["development_learning_observed"],
        "candidate_development_success_rate": report["result"][
            "candidate_development_success_rate"
        ],
        "fresh_development_success_rate": report["result"]["fresh_development_success_rate"],
        "replica_count": report["result"]["swarm"]["replica_count"],
        "promoted_snapshot_unchanged": report["result"]["promoted_snapshot_unchanged"],
        "report_sha256": f"sha256:{file_sha256(report_path)}",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
