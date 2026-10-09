"""Train and strictly test a reward-grounded recurrent causal policy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from gum.lineage import file_sha256
from gum.school.recurrent_meta import RecurrentCausalLearner, RecurrentMetaConfig
from gum.school.training import _summarize, _trial_seeds, _wilson
from gum.school.validation import DEFAULT_CURRICULUM
from gum.school.worlds import FOUNDATIONAL_LOADERS, create_lesson_world
from gum.storage import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "evidence" / "gum-school" / "research"
    / "causal-recurrent-meta-v1" / "RECURRENT_META_REPORT.json"
)
RESEARCH_BASIS = {
    "rl_squared": "https://arxiv.org/abs/1611.02779",
    "learning_to_reinforcement_learn": "https://arxiv.org/abs/1611.05763",
    "recurrent_pomdp_baseline": "https://proceedings.mlr.press/v162/ni22a.html",
    "self_imitation_learning": "https://proceedings.mlr.press/v80/oh18b.html",
}


class RandomCausalPolicy:
    """Reproducible non-learning control used only as an evaluation baseline."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.action_count = 0

    def begin(self, spec, observation, *, training: bool) -> None:
        self.action_count = spec.action_count

    def act(self, observation, *, training: bool) -> int:
        return int(self.rng.integers(self.action_count))

    def observe(self, action, transition, *, training: bool) -> None:
        return None

    def finish_episode(self, *, training: bool) -> None:
        return None

    def confidence(self) -> float:
        return 0.0


def _episode(
    learner,
    world,
    *,
    seed: int,
    training: bool,
    limit: int,
    capture_observations: bool = False,
) -> dict[str, Any]:
    observation = world.reset(seed)
    learner.begin(world.public_spec(), observation, training=training)
    total = 0.0
    actions = []
    rewards = []
    uncertainty = 0
    unnecessary = 0
    known_ineffective_actions: set[int] = set()
    observations = []
    success = False
    for _ in range(min(limit, world.public_spec().horizon)):
        if capture_observations:
            observations.append(np.asarray(observation).copy())
        action = learner.act(observation, training=training)
        if learner.confidence() < 0.05:
            uncertainty += 1
        transition = world.step(action)
        learner.observe(action, transition, training=training)
        event = transition.public_info.get("event")
        if event == "no-change":
            if action in known_ineffective_actions:
                unnecessary += 1
            known_ineffective_actions.add(action)
        elif event in {"blocked", "rejected"}:
            unnecessary += 1
        actions.append(int(action))
        rewards.append(float(transition.reward))
        total += float(transition.reward)
        observation = transition.observation
        if transition.terminated or transition.truncated:
            success = bool(transition.reward > 0.0)
            break
    loss = learner.finish_episode(training=training)
    result = {
        "seed": int(seed),
        "success": success,
        "interactions": len(actions),
        "return": total,
        "uncertain_actions": uncertainty,
        "unnecessary_actions": unnecessary,
        "actions": actions,
        "rewards": rewards,
        "loss": loss,
    }
    if capture_observations:
        result["observations"] = observations
    return result


def _world(lesson: dict, root: Path, *, seed: int):
    package = create_lesson_world(root, lesson, seed=seed)
    return FOUNDATIONAL_LOADERS[lesson["adapter"]](package)


def _train(
    learner: RecurrentCausalLearner,
    lesson: dict,
    *,
    root: Path,
    budget: int,
    episode_limit: int,
    self_imitation_epochs: int,
) -> dict[str, Any]:
    interactions = 0
    episode_index = 0
    successes = 0
    recent = []
    successful_trajectories = []
    while interactions < budget:
        base = lesson["training"]["seeds"][episode_index % len(lesson["training"]["seeds"])]
        seed = int(base) + 10_000 * (episode_index // len(lesson["training"]["seeds"]))
        world = _world(lesson, root, seed=seed)
        row = _episode(
            learner,
            world,
            seed=seed,
            training=True,
            limit=min(episode_limit, budget - interactions),
            capture_observations=True,
        )
        interactions += row["interactions"]
        episode_index += 1
        successes += int(row["success"])
        if row["success"]:
            successful_trajectories.append({
                "observations": row["observations"],
                "actions": row["actions"],
                "rewards": row["rewards"],
                "return": row["return"],
                "interactions": row["interactions"],
            })
        recent.append(row)
        if len(recent) > 100:
            recent.pop(0)
    learner.flush_training()
    self_imitation = learner.fit_self_imitation(
        successful_trajectories, epochs=self_imitation_epochs
    )
    self_imitation["selection"] = "all trajectories with positive terminal reward"
    return {
        "budget": budget,
        "interactions": interactions,
        "episodes": episode_index,
        "successes": successes,
        "success_rate": successes / episode_index,
        "last_100_success_rate": sum(item["success"] for item in recent) / len(recent),
        "last_100_mean_interactions": sum(item["interactions"] for item in recent) / len(recent),
        "self_imitation": self_imitation,
    }


def _evaluate(
    learner,
    lesson: dict,
    *,
    root: Path,
    trials: int,
    episode_limit: int,
) -> dict[str, Any]:
    rows = []
    for index, seed in enumerate(
        _trial_seeds(lesson["evaluation"]["development_seeds"], trials)
    ):
        world = _world(lesson, root, seed=seed)
        rows.append(_episode(
            learner,
            world,
            seed=seed,
            training=False,
            limit=episode_limit,
        ))
    summary = _summarize(rows)
    lower, upper = _wilson(summary["successes"], summary["trials"])
    summary["success_interval"] = {"lower": lower, "upper": upper, "method": "wilson"}
    summary["episode_limit"] = episode_limit
    summary["mean_return"] = sum(item["return"] for item in rows) / len(rows)
    summary["sample_trajectories"] = rows[:8]
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work", type=Path, default=ROOT / ".test-temp" / "recurrent-meta")
    parser.add_argument("--seed", type=int, default=8_620_101)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--episode-limit", type=int, default=11)
    parser.add_argument("--train-episode-limit", type=int, default=11)
    parser.add_argument("--self-imitation-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--entropy", type=float, default=0.02)
    parser.add_argument("--evaluation-temperature", type=float, default=0.25)
    args = parser.parse_args(argv)
    if (args.trials < 1 or args.episode_limit < 1
            or args.train_episode_limit < 1 or args.self_imitation_epochs < 1):
        raise SystemExit("trials and episode limits must be positive")

    curriculum = json.loads(DEFAULT_CURRICULUM.read_text(encoding="utf-8"))
    lesson = next(
        item for item in curriculum["lessons"]
        if item["lesson_id"] == "causal-workshop.controls.001"
    )
    budget = int(lesson["training"]["interaction_budget"] if args.budget is None else args.budget)
    if budget < 1:
        raise SystemExit("budget must be positive")
    work = args.work.resolve()
    if work.exists():
        shutil.rmtree(work)
    (work / "train").mkdir(parents=True)
    (work / "evaluate-learned").mkdir()
    (work / "evaluate-initial").mkdir()
    (work / "evaluate-random").mkdir()

    config = RecurrentMetaConfig(
        learning_rate=args.learning_rate,
        entropy_coefficient=args.entropy,
        evaluation_temperature=args.evaluation_temperature,
    )
    learner = RecurrentCausalLearner(args.seed, config=config)
    initial = RecurrentCausalLearner(args.seed, config=config)
    training = _train(
        learner,
        lesson,
        root=work / "train",
        budget=budget,
        episode_limit=args.train_episode_limit,
        self_imitation_epochs=args.self_imitation_epochs,
    )
    learned = _evaluate(
        learner,
        lesson,
        root=work / "evaluate-learned",
        trials=args.trials,
        episode_limit=args.episode_limit,
    )
    untrained = _evaluate(
        initial,
        lesson,
        root=work / "evaluate-initial",
        trials=args.trials,
        episode_limit=args.episode_limit,
    )
    random = _evaluate(
        RandomCausalPolicy(args.seed),
        lesson,
        root=work / "evaluate-random",
        trials=args.trials,
        episode_limit=args.episode_limit,
    )
    threshold = float(lesson["promotion"]["minimum_success"])
    lower_threshold = float(lesson["promotion"]["minimum_success_lower_bound"])
    strict_pass = (
        learned["success_rate"] >= threshold
        and learned["success_interval"]["lower"] >= lower_threshold
    )
    report = {
        "format": "gum-school-recurrent-meta-experiment-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_curriculum_run": False,
        "development_only": True,
        "sealed_data_used": False,
        "lesson_id": lesson["lesson_id"],
        "research_basis": RESEARCH_BASIS,
        "method": {
            "family": "recurrent-model-free-meta-reinforcement-learning",
            "optimizer": "batched-advantage-actor-critic-plus-success-replay",
            "information_boundary": learner.status()["information_boundary"],
            "forbidden_inputs": [
                "public_info.event", "hidden_world_state", "tried_action_mask",
                "prescribed_probe_order", "repeat_progress_rule",
            ],
            "config": config.__dict__,
        },
        "public_training_seeds": lesson["training"]["seeds"],
        "public_development_seeds": lesson["evaluation"]["development_seeds"],
        "training": training,
        "evaluation": {
            "trained_recurrent_policy": learned,
            "same_untrained_network": untrained,
            "uniform_random_policy": random,
        },
        "strict_thresholds": {
            "success_rate": threshold,
            "wilson_lower_bound": lower_threshold,
            "maximum_actions_per_trial": args.episode_limit,
            "maximum_training_actions_per_episode": args.train_episode_limit,
        },
        "strict_pass": strict_pass,
        "learner_status": learner.status(),
        "source_hashes": {
            "gum/school/recurrent_meta.py": f"sha256:{file_sha256(ROOT / 'gum/school/recurrent_meta.py')}",
            "gum/school/worlds.py": f"sha256:{file_sha256(ROOT / 'gum/school/worlds.py')}",
            "scripts/run_school_recurrent_meta_experiment.py": f"sha256:{file_sha256(Path(__file__))}",
        },
        "interpretation": (
            "The reward-grounded recurrent policy met the strict development criterion."
            if strict_pass
            else "The reward-grounded recurrent policy did not meet the strict development criterion."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, report, backup=False, sort_keys=True)
    learner.save(args.output.with_name("RECURRENT_META_POLICY.pt"))
    shutil.rmtree(work)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if strict_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
