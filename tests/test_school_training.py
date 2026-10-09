from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from gum.lineage import HashLedger, canonical
from gum.school.learner import CrossSeedSchoolLearner, SchoolLearnerError
from gum.school.rehearsal import run_nonpromoting_rehearsal
from gum.school.training import TrainingLaneConfig
from gum.school.validation import DEFAULT_CURRICULUM
from gum.school.worlds import (
    FOUNDATIONAL_LOADERS,
    OBJECT_LABORATORY_ADAPTER,
    create_foundational_world,
)
from gum.storage import atomic_write_json


def _object_world(root: Path, *, package_seed: int, episode_seed: int):
    package = create_foundational_world(
        root, OBJECT_LABORATORY_ADAPTER, seed=package_seed, mechanism="occlusion"
    )
    world = FOUNDATIONAL_LOADERS[OBJECT_LABORATORY_ADAPTER](package)
    return world, world.reset(episode_seed)


def test_learner_shares_parameters_across_world_ids_and_round_trips(tmp_path: Path):
    learner = CrossSeedSchoolLearner(seed=123)
    world_ids = []
    for index, seed in enumerate((11001, 11002)):
        world, observation = _object_world(
            tmp_path / f"world-{index}", package_seed=seed, episode_seed=seed + 100
        )
        world_ids.append(world.public_spec().world_id)
        learner.begin(world.public_spec(), observation, training=True)
        action = learner.act(observation, training=True)
        learner.observe(action, world.step(action), training=True)
        learner.finish_episode(training=True)

    assert set(learner.weights) == {OBJECT_LABORATORY_ADAPTER}
    assert learner.status()["seen_world_count"] == 2
    assert learner.adapter_steps[OBJECT_LABORATORY_ADAPTER] == 2
    assert set(world_ids) == learner.seen_worlds

    path = tmp_path / "learner.json"
    learner.save(path)
    restored = CrossSeedSchoolLearner.load(path)
    assert canonical(restored.to_json()) == canonical(learner.to_json())
    assert np.array_equal(
        restored.weights[OBJECT_LABORATORY_ADAPTER],
        learner.weights[OBJECT_LABORATORY_ADAPTER],
    )


def test_learner_evaluation_is_deterministic_and_does_not_change_saved_state(tmp_path: Path):
    learner = CrossSeedSchoolLearner(seed=456)
    state_path = tmp_path / "learner.json"
    learner.save(state_path)
    before = state_path.read_bytes()
    actions = []
    for index in range(2):
        world, observation = _object_world(
            tmp_path / f"evaluation-{index}", package_seed=7, episode_seed=77
        )
        learner = CrossSeedSchoolLearner.load(state_path)
        learner.begin(world.public_spec(), observation, training=False)
        episode_actions = []
        for _ in range(8):
            action = learner.act(observation, training=False)
            episode_actions.append(action)
            transition = world.step(action)
            learner.observe(action, transition, training=False)
            observation = transition.observation
        actions.append(episode_actions)
    assert actions[0] == actions[1]
    assert state_path.read_bytes() == before


def test_pixel_memory_tracks_unseen_object_motion_without_audit_input(tmp_path: Path):
    learner = CrossSeedSchoolLearner(seed=654)
    for index, seed in enumerate(range(11101, 11133)):
        world, observation = _object_world(
            tmp_path / f"tracking-{index}", package_seed=seed, episode_seed=seed
        )
        expected = world.audit_state()["final_order"].index(
            world.audit_state()["target_identity"]
        )
        learner.begin(world.public_spec(), observation, training=False)
        action = learner.act(observation, training=False)
        transition = world.step(action)
        learner.observe(action, transition, training=False)
        learner.act(transition.observation, training=False)
        assert learner._predicted_slot == expected


def test_sustained_uncertainty_spawns_bounded_helpers_that_communicate(tmp_path: Path):
    learner = CrossSeedSchoolLearner(seed=777, max_replicas=4)
    for episode in range(12):
        seed = 11001 + episode
        world, observation = _object_world(
            tmp_path / f"swarm-{episode}", package_seed=seed, episode_seed=seed
        )
        learner.begin(world.public_spec(), observation, training=True)
        for _ in range(world.public_spec().horizon):
            action = learner.act(observation, training=True)
            transition = world.step(action)
            learner.observe(action, transition, training=True)
            observation = transition.observation
            if transition.terminated or transition.truncated:
                break
        learner.finish_episode(training=True)
    assert learner.replica_count == learner.max_replicas == 4
    assert len(learner.spawn_events) == 3
    assert all(row["reason"] == "sustained-low-action-confidence"
               for row in learner.spawn_events)
    assert sum(learner.replica_episodes) == 12
    assert all(count > 0 for count in learner.replica_episodes)
    assert learner.communication_rounds > 0


def test_learner_rejects_unknown_fields_and_bad_parameter_shapes(tmp_path: Path):
    path = tmp_path / "learner.json"
    learner = CrossSeedSchoolLearner(seed=789)
    learner.save(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["surprise"] = True
    atomic_write_json(path, value, backup=False, sort_keys=True)
    with pytest.raises(SchoolLearnerError, match="fields differ"):
        CrossSeedSchoolLearner.load(path)

    world, observation = _object_world(tmp_path / "shape", package_seed=8, episode_seed=88)
    learner.begin(world.public_spec(), observation, training=True)
    learner.save(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["weights"][OBJECT_LABORATORY_ADAPTER][0].pop()
    atomic_write_json(path, value, backup=False, sort_keys=True)
    with pytest.raises(SchoolLearnerError, match="invalid parameter shape"):
        CrossSeedSchoolLearner.load(path)


def test_development_rehearsal_trains_candidate_but_cannot_promote(tmp_path: Path):
    workspace = tmp_path / "school"
    report = run_nonpromoting_rehearsal(
        workspace,
        curriculum_path=DEFAULT_CURRICULUM,
        config=TrainingLaneConfig(
            max_training_interactions=48,
            training_episodes_per_seed=1,
            development_trials=2,
        ),
        learner_seed=9001,
    )
    result = report["result"]
    assert report["official_curriculum_run"] is False
    assert report["candidate_rehearsal_training_performed"] is True
    assert report["development_only"] is True and report["sealed_data_used"] is False
    assert result["outcome"] == "quarantine"
    assert 1 <= result["training_interactions"] <= 48
    assert result["promoted_snapshot_unchanged"] is True
    assert result["promoted_lessons"] == []
    assert {"sealed-performance", "evidence-integrity"}.issubset(result["failed_gates"])
    assert result["replay_verified"] and result["input_boundary_verified"]
    assert 1 <= result["swarm"]["replica_count"] <= 4
    assert result["swarm"]["max_replicas"] == 4
    assert HashLedger(workspace / "SCHOOL_LEDGER.jsonl").verify()["valid"]

    run_id = report["run_id"]
    run_state = json.loads(
        (workspace / "runs" / run_id / "RUN.json").read_text(encoding="utf-8")
    )
    assert run_state["status"] == "quarantined"
    assert (workspace / "quarantine" / f"{run_id}.json").is_file()
    assert (workspace / "runs" / run_id / "candidate" / "SCHOOL_LEARNER.json").is_file()
    assert not list(workspace.rglob("genome.private.json"))
    assert not list(workspace.rglob("world.json"))


def test_rehearsal_refuses_to_reuse_nonempty_workspace(tmp_path: Path):
    workspace = tmp_path / "school"
    workspace.mkdir()
    marker = workspace / "do-not-overwrite.txt"
    marker.touch()
    with pytest.raises(FileExistsError, match="not empty"):
        run_nonpromoting_rehearsal(
            workspace,
            config=TrainingLaneConfig(max_training_interactions=1, development_trials=1),
        )
    assert marker.is_file()


def test_four_replica_swarm_learns_before_single_replica_control(tmp_path: Path):
    shared = dict(
        max_training_interactions=300,
        training_episodes_per_seed=4,
        development_trials=32,
    )
    swarm = run_nonpromoting_rehearsal(
        tmp_path / "swarm-control",
        config=TrainingLaneConfig(**shared, max_replicas=4),
    )
    single = run_nonpromoting_rehearsal(
        tmp_path / "single-control",
        config=TrainingLaneConfig(**shared, max_replicas=1),
    )
    assert swarm["result"]["development_learning_observed"] is True
    assert single["result"]["development_learning_observed"] is False
    assert swarm["result"]["candidate_development_success_rate"] == 1.0
    assert single["result"]["candidate_development_success_rate"] < 0.8
    assert swarm["result"]["training_interactions"] <= 300
    assert single["result"]["training_interactions"] <= 300


def test_saved_rehearsal_evidence_matches_sources_and_artifacts():
    root = DEFAULT_CURRICULUM.parents[1]
    workspace = root / "evidence" / "gum-school" / "rehearsal" / "foundational-lane-v1"
    report = json.loads((workspace / "REHEARSAL_REPORT.json").read_text(encoding="utf-8"))
    assert report["result"]["outcome"] == "quarantine"
    assert report["result"]["promoted_snapshot_unchanged"] is True
    assert report["result"]["development_learning_observed"] is True
    assert report["result"]["candidate_development_success_rate"] == 1.0
    assert report["result"]["fresh_development_success_rate"] == 0.0
    assert report["result"]["swarm"]["replica_count"] == 4
    assert not list(workspace.rglob("genome.private.json"))
    # Rehearsal evidence is an immutable historical run. Its recorded source
    # hashes identify the code used then; later lessons may legitimately evolve
    # the learner without rewriting that earlier evidence.
    for relative, recorded in report["source_hashes"].items():
        assert (root / relative).is_file()
        assert recorded.startswith("sha256:") and len(recorded) == 71
    for relative, expected in report["artifact_hashes"].items():
        actual = hashlib.sha256((workspace / relative).read_bytes()).hexdigest()
        assert expected == f"sha256:{actual}"
    assert HashLedger(workspace / "SCHOOL_LEDGER.jsonl").verify()["valid"]
