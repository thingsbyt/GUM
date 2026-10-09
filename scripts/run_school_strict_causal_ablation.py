"""Test causal-control learning with the handcrafted intervention scheduler removed."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from gum.lineage import file_sha256
from gum.school.learner import CrossSeedSchoolLearner
from gum.school.training import _run_episode, _summarize, _trial_seeds, _wilson
from gum.school.validation import DEFAULT_CURRICULUM
from gum.school.worlds import FOUNDATIONAL_LOADERS, create_lesson_world
from gum.storage import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHOOL = (
    ROOT / "evidence" / "gum-school" / "sealed" / "object-laboratory-occlusion-v1"
)
DEFAULT_OUTPUT = (
    ROOT / "evidence" / "gum-school" / "strict"
    / "causal-controls-unscaffolded-v1" / "STRICT_CAUSAL_ABLATION.json"
)


class UnscaffoldedLearner(CrossSeedSchoolLearner):
    """Ablation learner with the causal probe/repeat scheduler disabled."""

    def begin(self, spec, observation, *, training: bool) -> None:
        super().begin(spec, observation, training=training)
        self._causal_active = False


def _load_unscaffolded(path: Path) -> UnscaffoldedLearner:
    base = CrossSeedSchoolLearner.load(path)
    learner = UnscaffoldedLearner(
        base.seed,
        alpha=base.alpha,
        gamma=base.gamma,
        max_replicas=base.max_replicas,
        initial_replicas=base.replica_count,
    )
    learner.__dict__.update(base.__dict__)
    return learner


def _snapshot_state(school: Path, snapshot_id: str) -> Path:
    return school / "snapshots" / snapshot_id / "state" / "SCHOOL_LEARNER.json"


def _lesson_two_snapshot(school: Path) -> str:
    report = json.loads((school / "SCHOOL_REPORT.json").read_text(encoding="utf-8"))
    for decision in report["decisions"]:
        if (
            decision["lesson_id"] == "object-laboratory.functional-category.002"
            and decision["outcome"] == "promote"
        ):
            return decision["candidate_snapshot_id"]
    raise RuntimeError("the school has no promoted Lesson 2 snapshot")


def _evaluate(
    learner,
    lesson: dict,
    *,
    root: Path,
    label: str,
    trials: int,
) -> dict:
    worlds = root / label
    rows = []
    for seed in _trial_seeds(lesson["evaluation"]["development_seeds"], trials):
        package = create_lesson_world(worlds, lesson, seed=seed)
        world = FOUNDATIONAL_LOADERS[lesson["adapter"]](package)
        rows.append(_run_episode(
            learner,
            world,
            seed=seed,
            training=False,
            interaction_limit=world.public_spec().horizon,
        ))
    summary = _summarize(rows)
    lower, upper = _wilson(summary["successes"], summary["trials"])
    summary["success_interval"] = {"lower": lower, "upper": upper, "method": "wilson"}
    return summary


def _train(learner, lesson: dict, *, root: Path, budget: int) -> dict:
    worlds = root / "training-worlds"
    interactions = 0
    episodes = 0
    while interactions < budget:
        seed = lesson["training"]["seeds"][episodes % len(lesson["training"]["seeds"])]
        package = create_lesson_world(worlds, lesson, seed=seed + episodes * 10_000)
        world = FOUNDATIONAL_LOADERS[lesson["adapter"]](package)
        row = _run_episode(
            learner,
            world,
            seed=seed + episodes * 1_000_000,
            training=True,
            interaction_limit=budget - interactions,
        )
        interactions += row["interactions"]
        episodes += 1
    return {"interactions": interactions, "episodes": episodes}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--school", type=Path, default=DEFAULT_SCHOOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--work", type=Path, default=ROOT / ".test-temp" / "strict-causal")
    args = parser.parse_args(argv)
    if args.trials < 1:
        raise SystemExit("trials must be positive")
    school = args.school.resolve()
    work = args.work.resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    curriculum = json.loads(DEFAULT_CURRICULUM.read_text(encoding="utf-8"))
    lesson = curriculum["lessons"][2]
    promoted = json.loads((school / "PROMOTED.json").read_text(encoding="utf-8"))
    promoted_path = _snapshot_state(school, promoted["snapshot_id"])
    lesson_two_path = _snapshot_state(school, _lesson_two_snapshot(school))
    scaffolded = CrossSeedSchoolLearner.load(promoted_path)
    promoted_unscaffolded = _load_unscaffolded(promoted_path)
    trained_unscaffolded = _load_unscaffolded(lesson_two_path)
    budget = int(lesson["training"]["interaction_budget"])
    training = _train(trained_unscaffolded, lesson, root=work, budget=budget)
    scaffolded_summary = _evaluate(
        scaffolded, lesson, root=work, label="scaffolded", trials=args.trials
    )
    promoted_ablation = _evaluate(
        promoted_unscaffolded,
        lesson,
        root=work,
        label="promoted-unscaffolded",
        trials=args.trials,
    )
    trained_ablation = _evaluate(
        trained_unscaffolded,
        lesson,
        root=work,
        label="trained-unscaffolded",
        trials=args.trials,
    )
    threshold = float(lesson["promotion"]["minimum_success"])
    strict_pass = (
        trained_ablation["success_rate"] >= threshold
        and trained_ablation["success_interval"]["lower"]
        >= lesson["promotion"]["minimum_success_lower_bound"]
    )
    report = {
        "format": "gum-school-strict-causal-ablation-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_curriculum_run": False,
        "development_only": True,
        "sealed_data_used": False,
        "lesson_id": lesson["lesson_id"],
        "removed_scaffold": "probe-each-control-once-then-repeat-progress-control",
        "public_training_seeds": lesson["training"]["seeds"],
        "public_development_seeds": lesson["evaluation"]["development_seeds"],
        "development_trials": args.trials,
        "full_curriculum_training_budget": budget,
        "source_hashes": {
            "gum/school/learner.py": f"sha256:{file_sha256(ROOT / 'gum/school/learner.py')}",
            "gum/school/training.py": f"sha256:{file_sha256(ROOT / 'gum/school/training.py')}",
            "gum/school/worlds.py": f"sha256:{file_sha256(ROOT / 'gum/school/worlds.py')}",
            "scripts/run_school_strict_causal_ablation.py": (
                f"sha256:{file_sha256(Path(__file__))}"
            ),
        },
        "scaffolded_promoted_control": scaffolded_summary,
        "promoted_snapshot_without_scaffold": promoted_ablation,
        "full_budget_training_without_scaffold": {
            "training": training,
            "development": trained_ablation,
        },
        "strict_threshold": threshold,
        "strict_pass": strict_pass,
        "interpretation": (
            "The unscaffolded learner met the strict criterion."
            if strict_pass
            else "The unscaffolded learner did not discover a general causal intervention policy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, report, backup=False, sort_keys=True)
    shutil.rmtree(work)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
