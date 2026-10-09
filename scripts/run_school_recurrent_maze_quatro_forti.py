"""Run four disjoint replications of general recurrent maze learning."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import torch

from gum.lineage import file_sha256
from gum.storage import atomic_write_json

try:
    from scripts.run_school_recurrent_maze_experiment import (
        PROMOTED,
        _compact,
        _continued_learner,
        _evaluate,
        _lesson,
        _train,
        RandomPolicy,
    )
except ModuleNotFoundError:
    from run_school_recurrent_maze_experiment import (
        PROMOTED,
        _compact,
        _continued_learner,
        _evaluate,
        _lesson,
        _train,
        RandomPolicy,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "evidence" / "gum-school" / "research"
    / "general-recurrent-maze-quatro-forti-v2"
)
DEFAULT_WORK = ROOT / ".test-temp" / "general-recurrent-maze-quatro-forti-v2"
LEARNER_SEEDS = (8_640_101, 8_640_211, 8_640_307, 8_640_401)
WORLD_OFFSETS = (100_000_000, 200_000_000, 300_000_000, 400_000_000)


def _offset_lesson(offset: int) -> dict:
    lesson = deepcopy(_lesson())
    lesson["training"]["seeds"] = [
        int(seed) + offset for seed in lesson["training"]["seeds"]
    ]
    lesson["evaluation"]["development_seeds"] = [
        int(seed) + offset for seed in lesson["evaluation"]["development_seeds"]
    ]
    return lesson


def _run_replica(
    index: int,
    *,
    output: Path,
    work: Path,
    policy: Path,
    device: str,
    budget: int,
    trials: int,
    novelty: float,
) -> dict:
    learner_seed = LEARNER_SEEDS[index]
    world_offset = WORLD_OFFSETS[index]
    lesson = _offset_lesson(world_offset)
    replica_output = output / f"replica-{index + 1}"
    replica_work = work / f"replica-{index + 1}"
    replica_output.mkdir(parents=True)
    learner = _continued_learner(
        policy, device=device, novelty=novelty, seed=learner_seed
    )
    inherited = _continued_learner(
        policy, device=device, novelty=0.0, seed=learner_seed
    )
    training, training_rows = _train(
        learner,
        lesson,
        replica_work / "training",
        budget=budget,
        episode_limit=240,
    )
    policy_path = replica_output / "GENERAL_RECURRENT_MAZE_POLICY.pt"
    learner.save(policy_path)
    candidate, candidate_rows = _evaluate(
        learner,
        lesson,
        replica_work / "candidate",
        trials=trials,
        limit=240,
    )
    baseline, baseline_rows = _evaluate(
        inherited,
        lesson,
        replica_work / "inherited",
        trials=trials,
        limit=240,
    )
    random, random_rows = _evaluate(
        RandomPolicy(learner_seed),
        lesson,
        replica_work / "random",
        trials=trials,
        limit=240,
    )
    gates = {
        "learned_more_than_inherited": candidate["successes"] > baseline["successes"],
        "learned_more_than_random": candidate["successes"] > random["successes"],
        "fixed_training_budget": training["interactions"] == budget,
        "disjoint_world_partition": True,
    }
    report = {
        "format": "gum-school-general-recurrent-maze-replica-v2",
        "replica": index + 1,
        "learner_seed": learner_seed,
        "world_seed_offset": world_offset,
        "public_seed_partition_reused": False,
        "maze_specialist_used": False,
        "training": training,
        "candidate": candidate,
        "inherited_with_same_exploration_biology": baseline,
        "uniform_random": random,
        "gates": gates,
        "replication_pass": all(gates.values()),
        "compact_episode_results": {
            "training": _compact(training_rows),
            "candidate": _compact(candidate_rows),
            "inherited": _compact(baseline_rows),
            "random": _compact(random_rows),
        },
        "policy_sha256": f"sha256:{file_sha256(policy_path)}",
    }
    atomic_write_json(
        replica_output / "REPLICA_REPORT.json", report, backup=False, sort_keys=True
    )
    shutil.rmtree(replica_work)
    return report


def _aggregate(rows: list[dict], key: str) -> dict:
    summaries = [row[key] for row in rows]
    successes = sum(item["successes"] for item in summaries)
    trials = sum(item["trials"] for item in summaries)
    interactions = sum(item["interactions"] for item in summaries)
    return {
        "successes": successes,
        "trials": trials,
        "success_rate": successes / trials,
        "interactions": interactions,
        "mean_interactions": interactions / trials,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--policy", type=Path, default=PROMOTED)
    parser.add_argument("--budget", type=int, default=15_000)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--novelty", type=float, default=0.02)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.budget < 1 or args.trials < 32 or args.novelty < 0.0:
        raise SystemExit("invalid Quatro Forti budget, trial count, or novelty coefficient")
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    output, work, policy = args.output.resolve(), args.work.resolve(), args.policy.resolve()
    if output.exists() or work.exists():
        raise SystemExit("refusing to overwrite an existing Quatro Forti run")
    output.mkdir(parents=True)
    work.mkdir(parents=True)
    started = time.perf_counter()
    replicas = []
    for index in range(4):
        print(f"Starting Quatro Forti replica {index + 1}/4", flush=True)
        replica = _run_replica(
            index,
            output=output,
            work=work,
            policy=policy,
            device=device,
            budget=args.budget,
            trials=args.trials,
            novelty=args.novelty,
        )
        replicas.append(replica)
        print(json.dumps({
            "replica": index + 1,
            "pass": replica["replication_pass"],
            "candidate": replica["candidate"]["successes"],
            "inherited": replica["inherited_with_same_exploration_biology"]["successes"],
            "random": replica["uniform_random"]["successes"],
        }), flush=True)
    candidate = _aggregate(replicas, "candidate")
    inherited = _aggregate(replicas, "inherited_with_same_exploration_biology")
    random = _aggregate(replicas, "uniform_random")
    report = {
        "format": "gum-school-general-recurrent-maze-quatro-forti-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_curriculum_run": False,
        "development_only": True,
        "sealed_data_used": False,
        "replications": 4,
        "independent_learner_seeds": list(LEARNER_SEEDS),
        "disjoint_world_seed_offsets": list(WORLD_OFFSETS),
        "training_interactions_per_replica": args.budget,
        "candidate_trials_per_replica": args.trials,
        "maze_specialist_used": False,
        "candidate": candidate,
        "inherited_with_same_exploration_biology": inherited,
        "uniform_random": random,
        "replica_passes": sum(row["replication_pass"] for row in replicas),
        "all_four_improved": all(row["replication_pass"] for row in replicas),
        "replica_reports": [f"replica-{index + 1}/REPLICA_REPORT.json" for index in range(4)],
        "wall_clock_seconds": time.perf_counter() - started,
        "device": device,
        "source_hashes": {
            "gum/school/recurrent_meta.py": f"sha256:{file_sha256(ROOT / 'gum/school/recurrent_meta.py')}",
            "scripts/run_school_recurrent_maze_experiment.py": f"sha256:{file_sha256(ROOT / 'scripts/run_school_recurrent_maze_experiment.py')}",
            "scripts/run_school_recurrent_maze_quatro_forti.py": f"sha256:{file_sha256(Path(__file__))}",
        },
    }
    atomic_write_json(
        output / "QUATRO_FORTI_REPORT.json", report, backup=False, sort_keys=True
    )
    shutil.rmtree(work)
    print(json.dumps(report, indent=2))
    return 0 if report["all_four_improved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
