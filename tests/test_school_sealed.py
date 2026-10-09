from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gum.lineage import HashLedger, canonical
from gum.school.learner import CrossSeedSchoolLearner
from gum.school.sealed import (
    SealedExamConfig,
    SealedExamError,
    _verify_seed_manifest,
    run_official_sealed_exam,
    select_seed_manifest,
)
from gum.school.training import TrainingLaneConfig
from gum.school.validation import DEFAULT_CURRICULUM
from gum.school.worlds import (
    FOUNDATIONAL_LOADERS,
    OBJECT_LABORATORY_ADAPTER,
    create_foundational_world,
)


class _DeterministicSystemRandom:
    next_seed = 1_100_000_003

    def randrange(self, start: int, stop: int) -> int:
        value = self.next_seed
        self.__class__.next_seed += 7_919
        assert start <= value < stop
        return value


def test_shifted_occlusion_varies_structure_but_remains_trackable(tmp_path: Path):
    learner = CrossSeedSchoolLearner(seed=22)
    signatures = set()
    for index, seed in enumerate(range(22001, 22033)):
        package = create_foundational_world(
            tmp_path / str(index), OBJECT_LABORATORY_ADAPTER,
            seed=seed, mechanism="occlusion-shifted",
        )
        world = FOUNDATIONAL_LOADERS[OBJECT_LABORATORY_ADAPTER](package)
        observation = world.reset(seed)
        audit = world.audit_state()
        signatures.add((
            tuple(audit["episode_slots"]),
            audit["episode_final_phase"],
            tuple(audit["episode_barrier"]),
        ))
        expected = audit["final_order"].index(audit["target_identity"])
        learner.begin(world.public_spec(), observation, training=False)
        action = learner.act(observation, training=False)
        transition = world.step(action)
        learner.observe(action, transition, training=False)
        learner.act(transition.observation, training=False)
        assert learner._predicted_slot == expected
    assert len(signatures) > 24


def test_seed_manifest_rejects_a_changed_freeze(tmp_path: Path):
    lesson = json.loads(DEFAULT_CURRICULUM.read_text(encoding="utf-8"))["lessons"][0]
    freeze = {
        "format": "gum-school-source-freeze-v1",
        "created_at_utc": "2026-10-08T00:00:00+00:00",
        "git_head": None,
        "curriculum_id": "gum-school-v1",
        "lesson_id": lesson["lesson_id"],
        "examination_id": lesson["evaluation"]["examination_ids"][0],
        "source_hashes": {},
        "protocol": {},
        "protocol_sha256": "sha256:" + "0" * 64,
    }
    try:
        select_seed_manifest(tmp_path / "manifest.json", freeze=freeze, lesson=lesson)
    except SealedExamError as error:
        assert "source freeze changed" in str(error)
    else:
        raise AssertionError("changed source freeze unexpectedly selected sealed seeds")


def test_official_sealed_exam_selects_after_training_and_promotes(
    tmp_path: Path, monkeypatch,
):
    _DeterministicSystemRandom.next_seed = 1_100_000_003
    monkeypatch.setattr("gum.school.sealed.secrets.SystemRandom", _DeterministicSystemRandom)
    monkeypatch.setattr("gum.school.sealed.secrets.token_hex", lambda count: "ab" * count)
    workspace = tmp_path / "sealed"
    report = run_official_sealed_exam(
        workspace,
        config=SealedExamConfig(
            training=TrainingLaneConfig(
                max_training_interactions=600,
                training_episodes_per_seed=8,
                development_trials=32,
                max_replicas=4,
            ),
            trials=32,
        ),
    )
    assert report["official_curriculum_run"] is True
    assert report["development_only"] is False
    assert report["sealed_data_used_for_training"] is False
    assert report["result"]["outcome"] == "promote"
    assert report["result"]["all_gates_passed"] is True
    assert report["result"]["candidate_success_rate"] >= 0.8
    assert report["result"]["matched_fresh_success_rate"] == 0.0
    assert report["result"]["promoted_snapshot_changed"] is True
    assert all(row["passed"] for row in report["gates"]["gates"].values())
    assert HashLedger(workspace / "SCHOOL_LEDGER.jsonl").verify()["valid"]
    assert not list(workspace.rglob("genome.private.json"))
    assert not list(workspace.rglob("world.json"))

    run = workspace / "runs" / report["run_id"]
    freeze = json.loads((run / "SOURCE_FREEZE.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run / "SEALED_SEED_MANIFEST.json").read_text(encoding="utf-8")
    )
    lesson = json.loads(DEFAULT_CURRICULUM.read_text(encoding="utf-8"))["lessons"][0]
    assert _verify_seed_manifest(manifest, freeze=freeze, lesson=lesson)
    payload = {
        key: value for key, value in manifest.items()
        if key != "seed_manifest_commitment"
    }
    assert manifest["seed_manifest_commitment"] == (
        f"sha256:{hashlib.sha256(canonical(payload)).hexdigest()}"
    )
    assert manifest["selected_after_training"] is True
    assert len(manifest["seeds"]) == len(set(manifest["seeds"])) == 32
    training_mtime = (run / "TRAINING.json").stat().st_mtime_ns
    manifest_mtime = (run / "SEALED_SEED_MANIFEST.json").stat().st_mtime_ns
    assert manifest_mtime >= training_mtime

    for relative, expected in report["artifact_hashes"].items():
        actual = hashlib.sha256((workspace / relative).read_bytes()).hexdigest()
        assert expected == f"sha256:{actual}"


def test_second_official_lesson_keeps_first_lesson_and_promotes(
    tmp_path: Path, monkeypatch,
):
    _DeterministicSystemRandom.next_seed = 1_200_000_003
    monkeypatch.setattr("gum.school.sealed.secrets.SystemRandom", _DeterministicSystemRandom)
    monkeypatch.setattr("gum.school.sealed.secrets.token_hex", lambda count: "cd" * count)
    workspace = tmp_path / "continued-school"
    run_official_sealed_exam(
        workspace,
        config=SealedExamConfig(
            training=TrainingLaneConfig(
                max_training_interactions=600,
                training_episodes_per_seed=8,
                development_trials=32,
                max_replicas=4,
            ),
            trials=32,
        ),
    )
    _DeterministicSystemRandom.next_seed = 1_300_000_003
    report = run_official_sealed_exam(
        workspace,
        continue_existing=True,
        config=SealedExamConfig(
            training=TrainingLaneConfig(
                max_training_interactions=4_000,
                training_episodes_per_seed=64,
                development_trials=32,
                max_replicas=4,
            ),
            trials=32,
        ),
    )
    assert report["lesson_id"] == "object-laboratory.functional-category.002"
    assert report["result"]["outcome"] == "promote"
    assert report["result"]["candidate_success_rate"] >= 0.8
    assert report["result"]["candidate_before_training_success_rate"] < 0.8
    assert report["result"]["retention_baseline_success_rate"] == 1.0
    assert report["result"]["retention_current_success_rate"] == 1.0
    assert report["result"]["promoted_snapshot_changed"] is True
    assert all(row["passed"] for row in report["gates"]["gates"].values())
    school_report = json.loads((workspace / "SCHOOL_REPORT.json").read_text(encoding="utf-8"))
    assert school_report["promoted_lessons"][:2] == [
        "object-laboratory.occlusion.001",
        "object-laboratory.functional-category.002",
    ]
    assert (workspace / "SEALED_EXAM_REPORT_002.json").is_file()
    assert not list(workspace.rglob("genome.private.json"))
    assert not list(workspace.rglob("world.json"))

    _DeterministicSystemRandom.next_seed = 1_400_000_003
    causal_report = run_official_sealed_exam(
        workspace,
        continue_existing=True,
        config=SealedExamConfig(
            training=TrainingLaneConfig(
                max_training_interactions=2_000,
                training_episodes_per_seed=32,
                development_trials=32,
                max_replicas=4,
            ),
            trials=32,
        ),
    )
    assert causal_report["lesson_id"] == "causal-workshop.controls.001"
    assert causal_report["result"]["outcome"] == "promote"
    assert causal_report["result"]["candidate_success_rate"] == 1.0
    assert causal_report["result"]["candidate_before_training_success_rate"] < 0.8
    assert causal_report["result"]["matched_fresh_success_rate"] < 0.8
    assert causal_report["result"]["retention_baseline_success_rate"] == 1.0
    assert causal_report["result"]["retention_current_success_rate"] == 1.0
    assert all(row["passed"] for row in causal_report["gates"]["gates"].values())
    school_report = json.loads((workspace / "SCHOOL_REPORT.json").read_text(encoding="utf-8"))
    assert school_report["promoted_lessons"][:3] == [
        "object-laboratory.occlusion.001",
        "object-laboratory.functional-category.002",
        "causal-workshop.controls.001",
    ]
    assert (workspace / "SEALED_EXAM_REPORT_003.json").is_file()
