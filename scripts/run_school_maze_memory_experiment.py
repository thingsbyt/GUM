"""Run the public, reward-gated spatial-memory experiment for Lesson 5."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from gum.lineage import file_sha256
from gum.protocol import PublicWorldSpec, Transition
from gum.school.maze_memory import MazeMemoryLearner
from gum.school.training import (
    _paired_advantage_interval,
    _run_episode,
    _summarize,
    _trial_seeds,
    _wilson,
)
from gum.school.validation import DEFAULT_CURRICULUM
from gum.school.worlds import create_lesson_world, load_changing_maze
from gum.storage import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "evidence" / "gum-school" / "research"
    / "maze-memory-reward-gated-v1"
)
DEFAULT_WORK = ROOT / ".test-temp" / "maze-memory-reward-gated-v1"


class AmnesicMazeLearner(MazeMemoryLearner):
    """Causal ablation: retain motor grounding but discard the spatial map."""

    def act(self, observation, *, training: bool) -> int:
        self.open_cells.clear()
        self.walls.clear()
        self.goal = None
        return super().act(observation, training=training)


class RandomMazeLearner:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.spec: PublicWorldSpec | None = None

    def begin(self, spec, observation, *, training: bool) -> None:
        self.spec = spec

    def act(self, observation, *, training: bool) -> int:
        assert self.spec is not None
        return int(self.rng.integers(self.spec.action_count))

    def observe(self, action, transition: Transition, *, training: bool) -> None:
        return None

    def finish_episode(self, *, training: bool) -> None:
        return None

    def confidence(self) -> float:
        return 0.0


def _lesson() -> dict:
    curriculum = json.loads(DEFAULT_CURRICULUM.read_text(encoding="utf-8"))
    return next(
        row for row in curriculum["lessons"]
        if row["lesson_id"] == "changing-maze.memory.001"
    )


def _package(root: Path, lesson: dict, seed: int):
    package = create_lesson_world(root, lesson, seed=seed)
    return load_changing_maze(package)


def _train(learner: MazeMemoryLearner, lesson: dict, root: Path) -> tuple[dict, list]:
    budget = int(lesson["training"]["interaction_budget"])
    rows = []
    interactions = 0
    episode = 0
    while interactions < budget:
        base = lesson["training"]["seeds"][episode % len(lesson["training"]["seeds"])]
        seed = int(base) + episode * 1_000_000
        world = _package(root / f"episode-{episode:04d}", lesson, seed)
        row = _run_episode(
            learner,
            world,
            seed=seed,
            training=True,
            interaction_limit=budget - interactions,
        )
        rows.append(row)
        interactions += row["interactions"]
        episode += 1
    summary = _summarize(rows)
    summary["budget"] = budget
    summary["budget_exhausted"] = interactions == budget
    summary["mode_values"] = learner.mode_values.tolist()
    summary["mode_visits"] = learner.mode_visits.tolist()
    summary["selected_mode"] = learner.status()["selected_mode"]
    return summary, rows


def _evaluate(learner, lesson: dict, root: Path, trials: int) -> tuple[dict, list]:
    rows = []
    seeds = _trial_seeds(lesson["evaluation"]["development_seeds"], trials)
    for index, seed in enumerate(seeds):
        world = _package(root / f"trial-{index:04d}", lesson, seed)
        rows.append(_run_episode(
            learner,
            world,
            seed=seed,
            training=False,
            interaction_limit=world.public_spec().horizon,
        ))
    summary = _summarize(rows)
    lower, upper = _wilson(summary["successes"], summary["trials"])
    summary["success_interval"] = {
        "lower": lower,
        "upper": upper,
        "method": "wilson",
    }
    return summary, rows


def _compact(rows: list[dict]) -> list[dict]:
    return [
        {
            key: value for key, value in row.items()
            if key != "trajectory"
        }
        for row in rows
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--trials", type=int, default=128)
    args = parser.parse_args(argv)
    if args.trials < 32:
        raise SystemExit("use at least the curriculum's 32 development trials")
    output = args.output.resolve()
    work = args.work.resolve()
    if output.exists() or work.exists():
        raise SystemExit("refusing to overwrite an existing experiment")
    output.mkdir(parents=True)
    work.mkdir(parents=True)
    started = time.perf_counter()
    lesson = _lesson()

    candidate = MazeMemoryLearner(23)
    before_path = output / "MAZE_MEMORY_BEFORE.json"
    candidate.save(before_path)
    training_summary, training_rows = _train(
        candidate, lesson, work / "training"
    )
    candidate_path = output / "MAZE_MEMORY_AFTER.json"
    candidate.save(candidate_path)

    candidate_summary, candidate_rows = _evaluate(
        candidate, lesson, work / "candidate", args.trials
    )
    fresh_summary, fresh_rows = _evaluate(
        MazeMemoryLearner(23), lesson, work / "fresh", args.trials
    )
    random_summary, random_rows = _evaluate(
        RandomMazeLearner(23), lesson, work / "random", args.trials
    )
    amnesic = AmnesicMazeLearner.load(candidate_path)
    amnesic_summary, amnesic_rows = _evaluate(
        amnesic, lesson, work / "amnesic", args.trials
    )

    replay_seed = int(lesson["evaluation"]["development_seeds"][0]) + 9_000_000
    replay_a = MazeMemoryLearner.load(candidate_path)
    replay_b = MazeMemoryLearner.load(candidate_path)
    replay_row_a = _run_episode(
        replay_a,
        _package(work / "replay-a", lesson, replay_seed),
        seed=replay_seed,
        training=False,
        interaction_limit=240,
    )
    replay_row_b = _run_episode(
        replay_b,
        _package(work / "replay-b", lesson, replay_seed),
        seed=replay_seed,
        training=False,
        interaction_limit=240,
    )
    replay_verified = replay_row_a == replay_row_b

    success_lower = candidate_summary["success_interval"]["lower"]
    advantage_lower, advantage_upper = _paired_advantage_interval(
        [bool(row["success"]) for row in candidate_rows],
        [bool(row["success"]) for row in fresh_rows],
    )
    fresh_rate = fresh_summary["success_rate"]
    advantage = candidate_summary["success_rate"] / max(
        fresh_rate, 1.0 / args.trials
    )
    thresholds = lesson["promotion"]
    gates = {
        "public_success": (
            candidate_summary["success_rate"] >= thresholds["minimum_success"]
            and success_lower >= thresholds["minimum_success_lower_bound"]
        ),
        "fresh_advantage": (
            advantage >= thresholds["minimum_fresh_advantage"]
            and advantage_lower >= thresholds["minimum_fresh_advantage_lower_bound"]
        ),
        "uncertainty": candidate_summary["uncertainty_rate"]
        <= thresholds["maximum_uncertainty_rate"],
        "unnecessary_actions": candidate_summary["unnecessary_action_rate"]
        <= thresholds["maximum_unnecessary_action_rate"],
        "memory_ablation": (
            candidate_summary["success_rate"] - amnesic_summary["success_rate"]
        ) >= 0.50,
        "input_boundary": not candidate.boundary_violations,
        "replay": replay_verified,
        "resource_limit": training_summary["interactions"]
        <= lesson["resource_limits"]["max_interactions"],
    }

    trajectories = {
        "format": "gum-school-maze-memory-trajectories-v1",
        "training": training_rows,
        "candidate": candidate_rows,
        "fresh": fresh_rows,
        "random": random_rows,
        "amnesic": amnesic_rows,
        "replay": replay_row_a,
    }
    trajectory_path = output / "TRAJECTORIES.json"
    atomic_write_json(trajectory_path, trajectories, backup=False, sort_keys=True)
    report = {
        "format": "gum-school-maze-memory-research-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_curriculum_run": False,
        "development_only": True,
        "sealed_data_used": False,
        "lesson_id": lesson["lesson_id"],
        "claim_boundary": {
            "engineered_biology": [
                "pixel-cell segmentation",
                "episodic allocentric map store",
                "shortest-path and frontier search",
                "anonymous-control hypothesis propagation",
            ],
            "learned_from_scalar_reward": (
                "whether to activate the episodic-map strategy"
            ),
            "learned_online_from_observed_effects": [
                "map contents", "goal location", "anonymous control meanings"
            ],
            "not_used": [
                "hidden map", "hidden position", "goal coordinates",
                "control map", "event labels", "successful action sequences",
                "sealed seeds",
            ],
        },
        "training": training_summary,
        "development_trials": args.trials,
        "candidate": candidate_summary,
        "candidate_before_training": fresh_summary,
        "matched_fresh": fresh_summary,
        "random": random_summary,
        "spatial_memory_ablation": amnesic_summary,
        "effect": {
            "candidate_minus_fresh_success": (
                candidate_summary["success_rate"] - fresh_summary["success_rate"]
            ),
            "candidate_minus_amnesic_success": (
                candidate_summary["success_rate"] - amnesic_summary["success_rate"]
            ),
            "fresh_advantage": advantage,
            "fresh_advantage_interval": {
                "lower": advantage_lower,
                "upper": advantage_upper,
                "method": "paired-bootstrap",
            },
        },
        "gates": gates,
        "public_research_pass": all(gates.values()),
        "replay": {
            "verified": replay_verified,
            "seed": replay_seed,
        },
        "resources": {
            "wall_clock_seconds": time.perf_counter() - started,
            "training_interactions": training_summary["interactions"],
        },
        "source_hashes": {
            "gum/school/maze_memory.py": (
                f"sha256:{file_sha256(ROOT / 'gum/school/maze_memory.py')}"
            ),
            "gum/school/worlds.py": (
                f"sha256:{file_sha256(ROOT / 'gum/school/worlds.py')}"
            ),
            "gum/school/training.py": (
                f"sha256:{file_sha256(ROOT / 'gum/school/training.py')}"
            ),
            "scripts/run_school_maze_memory_experiment.py": (
                f"sha256:{file_sha256(Path(__file__))}"
            ),
        },
        "artifacts": {
            "before": before_path.name,
            "after": candidate_path.name,
            "trajectories": trajectory_path.name,
            "trajectory_sha256": f"sha256:{file_sha256(trajectory_path)}",
        },
        "compact_episode_results": {
            "candidate": _compact(candidate_rows),
            "fresh": _compact(fresh_rows),
            "amnesic": _compact(amnesic_rows),
        },
        "next_decision": (
            "Integrate the specialist into a cumulative candidate and run retention plus a frozen sealed exam."
            if all(gates.values())
            else "Do not integrate; diagnose the failed public gate."
        ),
    }
    report_path = output / "MAZE_MEMORY_RESEARCH_REPORT.json"
    atomic_write_json(report_path, report, backup=False, sort_keys=True)
    print(json.dumps({
        "report": str(report_path),
        "public_research_pass": report["public_research_pass"],
        "training": training_summary,
        "candidate": candidate_summary,
        "fresh": fresh_summary,
        "amnesic": amnesic_summary,
        "gates": gates,
    }, indent=2))
    return 0 if report["public_research_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
