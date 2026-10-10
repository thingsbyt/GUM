"""Run a predeclared, matched pilot of the cooperative learning bottleneck."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from gum.school.escape_chamber import DEVELOPMENT_ADAPTER
from gum.school.escape_diagnostics import diagnose_team
from gum.school.escape_team import EscapeTeam
from gum.school.recurrent_meta import RecurrentMetaConfig
from gum.storage import atomic_write_json


FORMAT = "gum-cooperative-escape-development-study-v1"
CONDITIONS = (
    {
        "name": "reference",
        "changes_from_reference": [],
        "exploration_overrides": {},
    },
    {
        "name": "novelty-off",
        "changes_from_reference": ["episodic_novelty_coefficient: 0.01 -> 0.0"],
        "exploration_overrides": {"episodic_novelty_coefficient": 0.0},
    },
    {
        "name": "novelty-off-action-mixer-off",
        "changes_from_novelty_off": ["episodic_action_exploration_mix: 0.50 -> 0.0"],
        "exploration_overrides": {
            "episodic_novelty_coefficient": 0.0,
            "episodic_action_exploration_mix": 0.0,
        },
    },
)


def _episode_summary(capsules: list[dict[str, Any]]) -> dict[str, Any]:
    escaped = [int(row["escaped_count"]) for row in capsules]
    return {
        "episodes": len(capsules),
        "episodes_with_any_escape": sum(value > 0 for value in escaped),
        "episodes_at_legal_maximum": sum(value == 3 for value in escaped),
        "total_escapes": sum(escaped),
        "mean_escapes": float(np.mean(escaped)) if escaped else None,
        "episode_ids": [row["episode_id"] for row in capsules],
    }


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pre_registered_before_first_team_birth": True,
        "environment_adapter": DEVELOPMENT_ADAPTER,
        "hard_room_status": "frozen and not used for development training",
        "geometry_change_only": True,
        "development_geometry": {
            "no_body_starts_on_plate": True,
            "one_body_starts_at_gate_approach": True,
            "plate_gate_manhattan_distance": 2,
        },
        "unchanged": [
            "four independent bodies",
            "anonymous shuffled controls",
            "sparse external reward",
            "simultaneous physics",
            "holder/crosser requirement",
            "no roles, messaging, scripted actions, or learner-visible semantic labels",
            "sharing off",
        ],
        "conditions": list(CONDITIONS),
        "replications_per_condition": args.replications,
        "baseline_evaluation_episodes": args.evaluation_episodes,
        "training_episodes": args.training_episodes,
        "post_evaluation_episodes": args.evaluation_episodes,
        "horizon": args.horizon,
        "seed_base": args.seed_base,
        "matched_design": (
            "Within each replication, all conditions use the same member seeds, "
            "initial random weights, training seeds, and evaluation seeds."
        ),
        "primary_measure": "post-training escapes versus matched pre-training escapes",
        "diagnostic_measures": [
            "plate dwell", "gate-open ticks", "gate proposals while open",
            "gate crossings", "action entropy", "external and intrinsic return",
        ],
        "milestone": (
            "Repeated post-training coordination across replications, improving over "
            "matched fresh-policy baselines; a single escape is not a breakthrough."
        ),
        "claim_boundary": (
            "This is a development pilot. It may identify a bottleneck; it cannot "
            "establish general cooperation or performance in frozen Room A."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--replications", type=int, default=2)
    parser.add_argument("--training-episodes", type=int, default=12)
    parser.add_argument("--evaluation-episodes", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--seed-base", type=int, default=8_410_000)
    parser.add_argument("--print-full", action="store_true")
    args = parser.parse_args(argv)
    if min(
        args.replications, args.training_episodes,
        args.evaluation_episodes, args.horizon,
    ) < 1:
        parser.error("replications, episodes, and horizon must be positive")

    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"study directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    plan = _plan(args)
    atomic_write_json(root / "STUDY_PLAN.json", plan, backup=False, sort_keys=True)

    rows = []
    for replication in range(args.replications):
        member_seed = args.seed_base + replication * 10_000
        member_seeds = tuple(member_seed + offset for offset in (101, 211, 307, 401))
        evaluation_seed = args.seed_base + 500_000 + replication * 1_000
        training_seed = args.seed_base + 700_000 + replication * 1_000
        for condition in CONDITIONS:
            team_root = root / f"replication-{replication + 1}" / condition["name"]
            team = EscapeTeam.create(
                team_root,
                seeds=member_seeds,
                config=RecurrentMetaConfig(batch_episodes=1),
                exploration_overrides=condition["exploration_overrides"],
                device=args.device,
            )
            baseline = [
                team.run_episode(
                    seed=evaluation_seed + index,
                    training=False,
                    horizon=args.horizon,
                    environment_adapter=DEVELOPMENT_ADAPTER,
                )
                for index in range(args.evaluation_episodes)
            ]
            training = [
                team.run_episode(
                    seed=training_seed + index,
                    training=True,
                    horizon=args.horizon,
                    environment_adapter=DEVELOPMENT_ADAPTER,
                )
                for index in range(args.training_episodes)
            ]
            post = [
                team.run_episode(
                    seed=evaluation_seed + index,
                    training=False,
                    horizon=args.horizon,
                    environment_adapter=DEVELOPMENT_ADAPTER,
                )
                for index in range(args.evaluation_episodes)
            ]
            diagnosis = diagnose_team(team_root)
            row = {
                "replication": replication + 1,
                "condition": condition["name"],
                "team_root": str(team_root),
                "member_seeds": list(member_seeds),
                "baseline": _episode_summary(baseline),
                "training": _episode_summary(training),
                "post_training": _episode_summary(post),
                "trajectory_diagnostics": diagnosis["aggregate"],
                "parameter_changes": diagnosis["parameter_changes"],
            }
            rows.append(row)
            atomic_write_json(
                root / "RESULTS_IN_PROGRESS.json",
                {"format": FORMAT, "plan": plan, "completed_rows": rows},
                backup=False,
                sort_keys=True,
            )

    by_condition = {}
    for condition in CONDITIONS:
        name = condition["name"]
        selected = [row for row in rows if row["condition"] == name]
        pre = [row["baseline"]["mean_escapes"] for row in selected]
        post = [row["post_training"]["mean_escapes"] for row in selected]
        by_condition[name] = {
            "independent_teams": len(selected),
            "mean_pre_training_escapes_per_episode": float(np.mean(pre)),
            "mean_post_training_escapes_per_episode": float(np.mean(post)),
            "mean_change": float(np.mean(np.asarray(post) - np.asarray(pre))),
            "replications_improved": sum(after > before for before, after in zip(pre, post)),
            "replications_unchanged": sum(after == before for before, after in zip(pre, post)),
            "replications_regressed": sum(after < before for before, after in zip(pre, post)),
        }
    result = {
        "format": FORMAT,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "results": rows,
        "condition_summary": by_condition,
        "interpretation_required": True,
        "automatic_breakthrough_claim": False,
    }
    atomic_write_json(root / "STUDY_RESULTS.json", result, backup=False, sort_keys=True)
    printed = result if args.print_full else {
        "format": FORMAT,
        "results_file": str(root / "STUDY_RESULTS.json"),
        "condition_summary": by_condition,
    }
    print(json.dumps(printed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
