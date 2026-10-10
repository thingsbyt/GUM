"""Test whether reward-selected replay retains rare cooperative experience."""
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


FORMAT = "gum-cooperative-escape-consolidation-study-v1"
CONDITIONS = (
    {"name": "actor-critic-only", "consolidation_epochs": 0},
    {"name": "reward-selected-consolidation", "consolidation_epochs": 25},
)
EXPLORATION = {
    "episodic_novelty_coefficient": 0.0,
    "episodic_action_exploration_mix": 0.0,
}


def _summary(capsules: list[dict[str, Any]]) -> dict[str, Any]:
    values = [int(row["escaped_count"]) for row in capsules]
    return {
        "episodes": len(values),
        "episodes_with_any_escape": sum(value > 0 for value in values),
        "total_escapes": sum(values),
        "mean_escapes": float(np.mean(values)),
        "episode_ids": [row["episode_id"] for row in capsules],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--replications", type=int, default=2)
    parser.add_argument("--training-episodes", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--seed-base", type=int, default=9_110_000)
    args = parser.parse_args(argv)
    if min(args.replications, args.training_episodes, args.evaluation_episodes, args.horizon) < 1:
        parser.error("replications, episodes, and horizon must be positive")

    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"study directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pre_registered_before_first_team_birth": True,
        "evidence_trigger": (
            "The development pilot produced rewarded trajectories but no held-out "
            "improvement, pointing to retention/credit assignment rather than encounter rarity."
        ),
        "environment_adapter": DEVELOPMENT_ADAPTER,
        "hard_room_status": "frozen and not used for development training",
        "conditions": list(CONDITIONS),
        "exploration_overrides": EXPLORATION,
        "replications_per_condition": args.replications,
        "baseline_evaluation_episodes": args.evaluation_episodes,
        "training_episodes": args.training_episodes,
        "post_evaluation_episodes": args.evaluation_episodes,
        "horizon": args.horizon,
        "seed_base": args.seed_base,
        "matched_design": (
            "Conditions share member seeds, initial weights, environment seeds, and "
            "deterministically reset evaluation sampling within each replication."
        ),
        "consolidation_boundary": (
            "Only a genuinely rewarded training episode may be replayed. Replay uses "
            "member pixels, executed anonymous actions, and scalar rewards only; no "
            "semantic events, roles, routes, teacher actions, or evaluation experience."
        ),
        "primary_measure": "post-training escapes versus matched pre-training escapes",
        "claim_boundary": (
            "Consolidation is engineered learning machinery. Any improvement would show "
            "retention of self-generated rewarded experience, not spontaneous role language "
            "or success in frozen Room A."
        ),
    }
    atomic_write_json(root / "STUDY_PLAN.json", plan, backup=False, sort_keys=True)

    rows = []
    for replication in range(args.replications):
        base = args.seed_base + replication * 10_000
        member_seeds = tuple(base + offset for offset in (101, 211, 307, 401))
        evaluation_seed = args.seed_base + 500_000 + replication * 1_000
        training_seed = args.seed_base + 700_000 + replication * 1_000
        for condition in CONDITIONS:
            team_root = root / f"replication-{replication + 1}" / condition["name"]
            team = EscapeTeam.create(
                team_root,
                seeds=member_seeds,
                config=RecurrentMetaConfig(batch_episodes=1),
                exploration_overrides=EXPLORATION,
                device=args.device,
            )
            baseline = [
                team.run_episode(
                    seed=evaluation_seed + index, training=False, horizon=args.horizon,
                    environment_adapter=DEVELOPMENT_ADAPTER,
                )
                for index in range(args.evaluation_episodes)
            ]
            training = []
            consolidations = []
            for index in range(args.training_episodes):
                capsule = team.run_episode(
                    seed=training_seed + index, training=True, horizon=args.horizon,
                    environment_adapter=DEVELOPMENT_ADAPTER,
                )
                training.append(capsule)
                rewarded = max(float(value) for value in capsule["returns"]) > 0.0
                if rewarded and condition["consolidation_epochs"]:
                    consolidations.append(team.consolidate_rewarded_episode(
                        capsule["episode_id"], epochs=condition["consolidation_epochs"]
                    ))
            post = [
                team.run_episode(
                    seed=evaluation_seed + index, training=False, horizon=args.horizon,
                    environment_adapter=DEVELOPMENT_ADAPTER,
                )
                for index in range(args.evaluation_episodes)
            ]
            diagnosis = diagnose_team(team_root)
            rows.append({
                "replication": replication + 1,
                "condition": condition["name"],
                "team_root": str(team_root),
                "member_seeds": list(member_seeds),
                "baseline": _summary(baseline),
                "training": _summary(training),
                "post_training": _summary(post),
                "consolidations": len(consolidations),
                "consolidation_ids": [row["consolidation_id"] for row in consolidations],
                "trajectory_diagnostics": diagnosis["aggregate"],
                "parameter_changes": diagnosis["parameter_changes"],
            })
            atomic_write_json(
                root / "RESULTS_IN_PROGRESS.json",
                {"format": FORMAT, "plan": plan, "completed_rows": rows},
                backup=False,
                sort_keys=True,
            )

    condition_summary = {}
    for condition in CONDITIONS:
        name = condition["name"]
        selected = [row for row in rows if row["condition"] == name]
        before = np.asarray([row["baseline"]["mean_escapes"] for row in selected])
        after = np.asarray([row["post_training"]["mean_escapes"] for row in selected])
        condition_summary[name] = {
            "independent_teams": len(selected),
            "mean_pre_training_escapes_per_episode": float(before.mean()),
            "mean_post_training_escapes_per_episode": float(after.mean()),
            "mean_change": float((after - before).mean()),
            "replications_improved": int(np.sum(after > before)),
            "replications_unchanged": int(np.sum(after == before)),
            "replications_regressed": int(np.sum(after < before)),
            "total_consolidations": sum(row["consolidations"] for row in selected),
        }
    result = {
        "format": FORMAT,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "results": rows,
        "condition_summary": condition_summary,
        "automatic_breakthrough_claim": False,
    }
    atomic_write_json(root / "STUDY_RESULTS.json", result, backup=False, sort_keys=True)
    print(json.dumps({
        "format": FORMAT,
        "results_file": str(root / "STUDY_RESULTS.json"),
        "condition_summary": condition_summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
