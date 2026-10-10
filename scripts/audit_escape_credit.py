"""Audit credit from one real successful cooperative-escape episode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gum.school.escape_credit_audit import audit_successful_episode
from gum.storage import atomic_write_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("team_root", type=Path)
    parser.add_argument("episode_id")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluation-seed-start", type=int, default=9_510_000)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--evaluation-horizon", type=int, default=80)
    args = parser.parse_args(argv)
    print("Replaying the successful episode and constructing matched branches...", flush=True)
    report = audit_successful_episode(
        args.team_root,
        args.episode_id,
        args.output_root,
        device=args.device,
        evaluation_seed_start=args.evaluation_seed_start,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_horizon=args.evaluation_horizon,
    )
    target = args.output_root / "CREDIT_AUDIT.json"
    atomic_write_json(target, report, backup=False, sort_keys=True)
    compact_evaluation = {
        branch: {
            key: value for key, value in result.items() if key != "per_seed"
        }
        for branch, result in report["evaluation"]["branches"].items()
    }
    print(json.dumps({
        "report": str(target.resolve()),
        "alignment_checks": report["alignment_checks"],
        "roles": report["observed_cooperation_roles_diagnostic_only"],
        "evaluation": compact_evaluation,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
