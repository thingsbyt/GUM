"""Continue the general recurrent policy on unseen, partially observed mazes.

This experiment deliberately does not use the engineered MazeMemoryLearner.
The same recurrent policy promoted on causal tasks receives only pixels, its
previous anonymous action, scalar reward, and termination.  A generic
within-episode pixel-novelty bonus supplies exploration pressure during
training; evaluation uses the environment reward alone.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch

from gum.lineage import file_sha256
from gum.school.recurrent_meta import RecurrentCausalLearner
from gum.school.training import _summarize, _trial_seeds, _wilson
from gum.school.validation import DEFAULT_CURRICULUM
from gum.school.worlds import FOUNDATIONAL_LOADERS, create_lesson_world
from gum.storage import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
PROMOTED = (
    ROOT / "evidence" / "gum-school"
    / "sealed-recurrent-composition-official-v1" / "snapshots"
    / "sha256-059137c51bcf5e977d4e56116590cadf1c64d63e9532b930cde60a2758d866a1"
    / "state" / "CAUSAL_META_POLICY.pt"
)
DEFAULT_OUTPUT = (
    ROOT / "evidence" / "gum-school" / "research"
    / "general-recurrent-maze-v3"
)
DEFAULT_WORK = ROOT / ".test-temp" / "general-recurrent-maze-v3"
RESEARCH_BASIS = {
    "recurrent_meta_rl": "https://arxiv.org/abs/1611.02779",
    "episodic_vs_global_novelty": "https://proceedings.mlr.press/v202/henaff23a.html",
    "elliptical_episodic_bonuses": "https://openreview.net/forum?id=Xg-yZos9qJQ",
    "intrinsic_curiosity_pixels": "https://proceedings.mlr.press/v70/pathak17a.html",
}


class RandomPolicy:
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


def _lesson() -> dict[str, Any]:
    curriculum = json.loads(DEFAULT_CURRICULUM.read_text(encoding="utf-8"))
    return next(
        row for row in curriculum["lessons"]
        if row["lesson_id"] == "changing-maze.memory.001"
    )


def _world(lesson: dict[str, Any], root: Path, seed: int):
    package = create_lesson_world(root, lesson, seed=seed)
    return FOUNDATIONAL_LOADERS[lesson["adapter"]](package)


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
    actions: list[int] = []
    rewards: list[float] = []
    observations = []
    uncertainty = 0
    unnecessary = 0
    success = False
    for _ in range(min(limit, world.public_spec().horizon)):
        if capture_observations:
            observations.append(np.asarray(observation).copy())
        action = int(learner.act(observation, training=training))
        uncertainty += int(learner.confidence() < 0.05)
        transition = world.step(action)
        learner.observe(action, transition, training=training)
        if transition.public_info.get("event") == "blocked" or action >= 4:
            unnecessary += 1
        actions.append(action)
        rewards.append(float(transition.reward))
        observation = transition.observation
        if transition.terminated or transition.truncated:
            success = bool(transition.reward > 0.0)
            break
    loss = learner.finish_episode(training=training)
    result = {
        "seed": int(seed),
        "success": success,
        "interactions": len(actions),
        "return": float(sum(rewards)),
        "uncertain_actions": uncertainty,
        "unnecessary_actions": unnecessary,
        "actions": actions,
        "rewards": rewards,
        "loss": loss,
    }
    if capture_observations:
        result["observations"] = observations
    return result


def _continued_learner(path: Path, *, device: str, novelty: float, seed: int):
    inherited = RecurrentCausalLearner.load(path, device=device)
    config = replace(
        inherited.config,
        learning_rate=3e-4,
        entropy_coefficient=0.03,
        batch_episodes=4,
        evaluation_temperature=0.75,
        training_exploration_mix=0.10,
        episodic_action_exploration_mix=0.80,
        episodic_novelty_coefficient=novelty,
    )
    learner = RecurrentCausalLearner(seed, config=config, device=device)
    learner.policy.load_state_dict(inherited.policy.state_dict())
    return learner


def _train(
    learner: RecurrentCausalLearner,
    lesson: dict[str, Any],
    root: Path,
    *,
    budget: int,
    episode_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    interactions = 0
    rows = []
    started = time.perf_counter()
    while interactions < budget:
        episode = len(rows)
        base = int(lesson["training"]["seeds"][episode % 4])
        seed = base + 1_000_000 * (episode // 4)
        row = _episode(
            learner,
            _world(lesson, root / f"episode-{episode:04d}", seed),
            seed=seed,
            training=True,
            limit=min(episode_limit, budget - interactions),
            capture_observations=True,
        )
        rows.append(row)
        interactions += row["interactions"]
    learner.flush_training()
    successful = [
        {
            "observations": row["observations"],
            "actions": row["actions"],
            "rewards": row["rewards"],
        }
        for row in rows if row["success"]
    ]
    self_imitation = (
        learner.fit_self_imitation(successful, epochs=10, batch_episodes=8)
        if successful else None
    )
    summary = _summarize(rows)
    summary.update({
        "budget": budget,
        "budget_exhausted": interactions == budget,
        "wall_clock_seconds": time.perf_counter() - started,
        "intrinsic_reward_total": learner.intrinsic_reward_total,
        "reward_selected_self_imitation": self_imitation,
    })
    return summary, rows


def _evaluate(
    learner,
    lesson: dict[str, Any],
    root: Path,
    *,
    trials: int,
    limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for index, seed in enumerate(
        _trial_seeds(lesson["evaluation"]["development_seeds"], trials)
    ):
        rows.append(_episode(
            learner,
            _world(lesson, root / f"trial-{index:04d}", seed),
            seed=seed,
            training=False,
            limit=limit,
        ))
    summary = _summarize(rows)
    lower, upper = _wilson(summary["successes"], summary["trials"])
    summary["success_interval"] = {"lower": lower, "upper": upper, "method": "wilson"}
    return summary, rows


def _compact(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value for key, value in row.items()
            if key not in {"actions", "rewards", "loss", "observations"}
        }
        for row in rows
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--policy", type=Path, default=PROMOTED)
    parser.add_argument("--seed", type=int, default=8_630_101)
    parser.add_argument("--budget", type=int, default=15_000)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--episode-limit", type=int, default=240)
    parser.add_argument("--novelty", type=float, default=0.02)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if min(args.budget, args.trials, args.episode_limit) < 1 or args.novelty < 0.0:
        raise SystemExit("budgets and trials must be positive; novelty cannot be negative")
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    output, work = args.output.resolve(), args.work.resolve()
    if output.exists() or work.exists():
        raise SystemExit("refusing to overwrite an existing maze experiment")
    output.mkdir(parents=True)
    work.mkdir(parents=True)
    lesson = _lesson()
    learner = _continued_learner(
        args.policy.resolve(), device=device, novelty=args.novelty, seed=args.seed
    )
    initial = _continued_learner(
        args.policy.resolve(), device=device, novelty=0.0, seed=args.seed
    )
    training, training_rows = _train(
        learner,
        lesson,
        work / "training",
        budget=args.budget,
        episode_limit=args.episode_limit,
    )
    policy_path = output / "GENERAL_RECURRENT_MAZE_POLICY.pt"
    learner.save(policy_path)
    learned, learned_rows = _evaluate(
        learner, lesson, work / "learned", trials=args.trials, limit=args.episode_limit
    )
    inherited, inherited_rows = _evaluate(
        initial, lesson, work / "inherited", trials=args.trials, limit=args.episode_limit
    )
    random, random_rows = _evaluate(
        RandomPolicy(args.seed), lesson, work / "random",
        trials=args.trials, limit=args.episode_limit,
    )
    thresholds = lesson["promotion"]
    gates = {
        "success": learned["success_rate"] >= thresholds["minimum_success"],
        "success_lower_bound": (
            learned["success_interval"]["lower"]
            >= thresholds["minimum_success_lower_bound"]
        ),
        "beats_inherited": learned["success_rate"] > inherited["success_rate"],
        "beats_random": learned["success_rate"] > random["success_rate"],
        "resource_limit": training["interactions"] <= lesson["resource_limits"]["max_interactions"],
    }
    report = {
        "format": "gum-school-general-recurrent-maze-research-v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_curriculum_run": False,
        "development_only": True,
        "sealed_data_used": False,
        "lesson_id": lesson["lesson_id"],
        "research_basis": RESEARCH_BASIS,
        "method": {
            "family": "continued-recurrent-meta-reinforcement-learning",
            "single_general_policy": True,
            "maze_specialist_used": False,
            "generic_exploration": (
                "first-visit pixel novelty plus episodic state-action novelty "
                "weighted by online observed action effects"
            ),
            "config": asdict(learner.config),
            "device": device,
            "learner_inputs": learner.status()["information_boundary"],
            "forbidden_inputs": [
                "map coordinates", "wall coordinates", "goal coordinates",
                "control meanings", "route", "public_info.event", "audit state",
            ],
        },
        "starting_policy": {
            "path": args.policy.resolve().relative_to(ROOT).as_posix(),
            "sha256": f"sha256:{file_sha256(args.policy.resolve())}",
        },
        "training": training,
        "candidate": learned,
        "inherited_before_maze_training": inherited,
        "uniform_random": random,
        "gates": gates,
        "development_pass": all(gates.values()),
        "compact_episode_results": {
            "training": _compact(training_rows),
            "candidate": _compact(learned_rows),
            "inherited": _compact(inherited_rows),
            "random": _compact(random_rows),
        },
        "source_hashes": {
            "gum/school/recurrent_meta.py": f"sha256:{file_sha256(ROOT / 'gum/school/recurrent_meta.py')}",
            "gum/school/worlds.py": f"sha256:{file_sha256(ROOT / 'gum/school/worlds.py')}",
            "scripts/run_school_recurrent_maze_experiment.py": f"sha256:{file_sha256(Path(__file__))}",
        },
        "artifacts": {
            "policy": policy_path.name,
            "policy_sha256": f"sha256:{file_sha256(policy_path)}",
        },
    }
    atomic_write_json(
        output / "GENERAL_RECURRENT_MAZE_REPORT.json",
        report,
        backup=False,
        sort_keys=True,
    )
    shutil.rmtree(work)
    print(json.dumps({
        "development_pass": report["development_pass"],
        "training": training,
        "candidate": learned,
        "inherited": inherited,
        "random": random,
        "gates": gates,
        "device": device,
    }, indent=2))
    return 0 if report["development_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
