"""Test scalar-reward-event replay on public causal-composition worlds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from gum.lineage import file_sha256
from gum.school.cumulative import CAUSAL_FILENAME, CumulativeSchoolLearner
from gum.school.recurrent_meta import RecurrentCausalLearner
from gum.school.training import _wilson
from gum.school.validation import DEFAULT_CURRICULUM
from gum.storage import atomic_write_json
if __package__:
    from scripts.run_school_recurrent_meta_experiment import (
        RandomCausalPolicy,
        _episode,
        _evaluate,
        _world,
    )
else:
    from run_school_recurrent_meta_experiment import (
        RandomCausalPolicy,
        _episode,
        _evaluate,
        _world,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_WORKSPACE = (
    ROOT / "evidence" / "gum-school" / "sealed-recurrent-official-v2"
)
DEFAULT_OUTPUT = (
    ROOT / "evidence" / "gum-school" / "research"
    / "causal-composition-reward-outcome-v2" / "COMPOSITION_REWARD_OUTCOME_REPORT.json"
)


def _promoted_bundle(workspace: Path) -> Path:
    pointer = json.loads((workspace / "PROMOTED.json").read_text(encoding="utf-8"))
    return workspace / "snapshots" / pointer["snapshot_id"] / "state"


def _train_reward_events(
    learner: RecurrentCausalLearner,
    composition_lesson: dict,
    controls_lesson: dict,
    *,
    root: Path,
    budget: int,
    episode_limit: int,
    replay_epochs: int,
    retention_fraction: float,
) -> dict[str, Any]:
    """Train and replay reward events inside one interleaved curriculum budget."""
    controls_budget = round(budget * retention_fraction)
    composition_budget = budget - controls_budget
    lanes = {
        "composition": {
            "lesson": composition_lesson,
            "budget": composition_budget,
            "interactions": 0,
            "episodes": 0,
            "successes": 0,
            "limit": episode_limit,
        },
        "controls_rehearsal": {
            "lesson": controls_lesson,
            "budget": controls_budget,
            "interactions": 0,
            "episodes": 0,
            "successes": 0,
            "limit": 11,
        },
    }
    recent = []
    reward_event_trajectories = []
    while any(lane["interactions"] < lane["budget"] for lane in lanes.values()):
        available = [
            (name, lane) for name, lane in lanes.items()
            if lane["interactions"] < lane["budget"]
        ]
        lane_name, lane = min(
            available,
            key=lambda item: item[1]["interactions"] / max(1, item[1]["budget"]),
        )
        lesson = lane["lesson"]
        episode_index = lane["episodes"]
        base = lesson["training"]["seeds"][episode_index % len(lesson["training"]["seeds"])]
        seed = int(base) + 10_000 * (episode_index // len(lesson["training"]["seeds"]))
        world = _world(lesson, root / lane_name, seed=seed)
        row = _episode(
            learner,
            world,
            seed=seed,
            training=True,
            limit=min(lane["limit"], lane["budget"] - lane["interactions"]),
            capture_observations=True,
        )
        lane["interactions"] += row["interactions"]
        lane["episodes"] += 1
        lane["successes"] += int(row["success"])
        if any(float(reward) > 0.0 for reward in row["rewards"]):
            reward_event_trajectories.append({
                "observations": row["observations"],
                "actions": row["actions"],
                "rewards": row["rewards"],
                "return": row["return"],
                "interactions": row["interactions"],
                "success": row["success"],
                "lane": lane_name,
            })
        recent.append({**row, "lane": lane_name})
        if len(recent) > 100:
            recent.pop(0)
    learner.flush_training()
    composition_events = [
        item for item in reward_event_trajectories if item["lane"] == "composition"
    ]
    controls_events = [
        item for item in reward_event_trajectories if item["lane"] == "controls_rehearsal"
    ]
    maximum_controls = max(
        1,
        round(len(composition_events) * retention_fraction / (1.0 - retention_fraction)),
    )
    if len(controls_events) > maximum_controls:
        generator = np.random.default_rng(learner.seed + 271)
        selected_indexes = sorted(
            int(index) for index in generator.choice(
                len(controls_events), size=maximum_controls, replace=False
            )
        )
        controls_events = [controls_events[index] for index in selected_indexes]
    replay_trajectories = composition_events + controls_events
    replay = learner.fit_reward_outcome_replay(
        replay_trajectories,
        epochs=replay_epochs,
    )
    replay.update({
        "trajectory_selection": "all trajectories containing positive scalar reward",
        "positive_token_objective": "imitate actions with immediate positive scalar reward",
        "negative_token_objective": "suppress actions with immediate negative scalar reward",
        "successful_selected_trajectories": sum(
            int(item["success"]) for item in replay_trajectories
        ),
        "partial_selected_trajectories": sum(
            not item["success"] for item in replay_trajectories
        ),
        "composition_selected_trajectories": sum(
            item["lane"] == "composition" for item in replay_trajectories
        ),
        "controls_selected_trajectories": sum(
            item["lane"] == "controls_rehearsal" for item in replay_trajectories
        ),
        "replay_lane_weighting": "matches the fixed interaction-budget proportions",
    })
    lane_summary = {}
    for name, lane in lanes.items():
        lane_summary[name] = {
            "budget": lane["budget"],
            "interactions": lane["interactions"],
            "episodes": lane["episodes"],
            "successes": lane["successes"],
            "success_rate": lane["successes"] / lane["episodes"],
        }
    return {
        "budget": budget,
        "interactions": sum(lane["interactions"] for lane in lanes.values()),
        "episodes": sum(lane["episodes"] for lane in lanes.values()),
        "lanes": lane_summary,
        "retention_fraction": retention_fraction,
        "last_100_success_rate": sum(item["success"] for item in recent) / len(recent),
        "last_100_mean_interactions": (
            sum(item["interactions"] for item in recent) / len(recent)
        ),
        "reward_event_replay": replay,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-workspace", type=Path, default=DEFAULT_OFFICIAL_WORKSPACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--work", type=Path,
        default=ROOT / ".test-temp" / "composition-reward-event",
    )
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--control-retention-trials", type=int, default=256)
    parser.add_argument("--episode-limit", type=int, default=32)
    parser.add_argument("--replay-epochs", type=int, default=100)
    parser.add_argument("--retention-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
    if min(
        args.trials,
        args.control_retention_trials,
        args.episode_limit,
        args.replay_epochs,
    ) < 1:
        raise SystemExit("trials, limits, and epochs must be positive")
    if not 0.0 < args.retention_fraction < 1.0:
        raise SystemExit("retention-fraction must be between zero and one")

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

    official_workspace = args.official_workspace.resolve()
    promoted_directory = _promoted_bundle(official_workspace)
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
        training = _train_reward_events(
            learner,
            composition,
            controls,
            root=work / "train",
            budget=budget,
            episode_limit=args.episode_limit,
            replay_epochs=args.replay_epochs,
            retention_fraction=args.retention_fraction,
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
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    policy_path = output.with_name("COMPOSITION_REWARD_OUTCOME_POLICY.pt")
    learner.save(policy_path)
    pointer = json.loads((official_workspace / "PROMOTED.json").read_text(encoding="utf-8"))
    report = {
        "format": "gum-school-recurrent-composition-reward-outcome-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_curriculum_run": False,
        "development_only": True,
        "sealed_data_used": False,
        "lesson_id": composition["lesson_id"],
        "starting_official_snapshot": pointer["snapshot_id"],
        "method": {
            "family": "continued-recurrent-model-free-meta-reinforcement-learning",
            "optimizer": "actor-critic-plus-scalar-reward-outcome-replay",
            "information_boundary": learner.status()["information_boundary"],
            "forbidden_inputs": [
                "public_info.event",
                "hidden_world_state",
                "advance_action_sequence",
                "tried_action_mask",
                "prescribed_probe_order",
                "authored_action_targets",
            ],
            "episode_limit": args.episode_limit,
            "policy_weights_updated": True,
            "base_object_laboratory_component_updated": False,
            "fixed_before_full_run": True,
            "continual_learning_schedule": (
                "lowest-consumed-budget-fraction first across composition and controls"
            ),
        },
        "public_training_seeds": {
            "composition": composition["training"]["seeds"],
            "controls_rehearsal": controls["training"]["seeds"],
        },
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
            "Scalar reward-outcome replay met the public composition and retention criteria."
            if strict_pass else
            "Scalar reward-outcome replay did not meet all public composition and retention criteria."
        ),
    }
    atomic_write_json(output, report, backup=False, sort_keys=True)
    print(json.dumps({
        "strict_pass": strict_pass,
        "training_lanes": training["lanes"],
        "reward_event_replay": training["reward_event_replay"],
        "composition_success_rate": trained["success_rate"],
        "composition_wilson_lower": trained["success_interval"]["lower"],
        "promoted_start_success_rate": promoted_start["success_rate"],
        "fresh_success_rate": untrained["success_rate"],
        "random_success_rate": random["success_rate"],
        "controls_retention_success_rate": controls_retention["success_rate"],
        "controls_retention_wilson_lower": controls_lower,
        "report": str(output),
    }, indent=2, sort_keys=True))
    return 0 if strict_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
