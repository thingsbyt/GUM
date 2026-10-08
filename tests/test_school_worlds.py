from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gum.harness import GUMHarness
from gum.school.admission import run_foundational_admission
from gum.school.validation import DEFAULT_CURRICULUM, load_json
from gum.school.worlds import (
    CAUSAL_WORKSHOP_ADAPTER,
    CHANGING_MAZE_ADAPTER,
    FOUNDATIONAL_LOADERS,
    OBJECT_LABORATORY_ADAPTER,
    FoundationalWorldError,
    create_foundational_world,
    create_lesson_world,
)


ADAPTERS = (
    OBJECT_LABORATORY_ADAPTER,
    CAUSAL_WORKSHOP_ADAPTER,
    CHANGING_MAZE_ADAPTER,
)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_foundational_packages_are_data_only_and_match_public_contract(tmp_path: Path, adapter: str):
    folder = create_foundational_world(tmp_path, adapter, seed=101)
    assert sorted(path.name for path in folder.iterdir()) == ["genome.private.json", "world.json"]
    public = json.loads((folder / "world.json").read_text())
    private = json.loads((folder / "genome.private.json").read_text())
    assert "seed" not in public and "seed" in private
    assert "action_map" not in public and "goal" not in public
    world = FOUNDATIONAL_LOADERS[adapter](folder)
    observation = world.reset(202)
    assert observation.dtype == np.uint8
    assert observation.shape == tuple(public["observation_shape"])
    assert world.public_spec().adapter == adapter


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_foundational_reset_and_replay_are_deterministic(tmp_path: Path, adapter: str):
    folder = create_foundational_world(tmp_path, adapter, seed=303)
    left = FOUNDATIONAL_LOADERS[adapter](folder)
    right = FOUNDATIONAL_LOADERS[adapter](folder)
    assert np.array_equal(left.reset(404), right.reset(404))
    safe_action = {OBJECT_LABORATORY_ADAPTER: 0, CAUSAL_WORKSHOP_ADAPTER: 0,
                   CHANGING_MAZE_ADAPTER: 5}[adapter]
    for _ in range(12):
        one = left.step(safe_action)
        two = right.step(safe_action)
        assert one.reward == two.reward
        assert one.terminated == two.terminated and one.truncated == two.truncated
        assert one.public_info == two.public_info
        assert np.array_equal(one.observation, two.observation)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_foundational_worlds_enforce_actions_horizon_and_terminal_state(tmp_path: Path, adapter: str):
    folder = create_foundational_world(tmp_path, adapter, seed=505)
    world = FOUNDATIONAL_LOADERS[adapter](folder)
    world.reset(606)
    with pytest.raises(FoundationalWorldError, match="integer"):
        world.step(True)
    with pytest.raises(FoundationalWorldError, match="action must be"):
        world.step(world.public_spec().action_count)
    world.reset(606)
    safe_action = {OBJECT_LABORATORY_ADAPTER: 0, CAUSAL_WORKSHOP_ADAPTER: 0,
                   CHANGING_MAZE_ADAPTER: 5}[adapter]
    for _ in range(world.public_spec().horizon):
        final = world.step(safe_action)
        if final.terminated or final.truncated:
            break
    assert final.truncated and world.steps == world.public_spec().horizon
    with pytest.raises(FoundationalWorldError, match="episode is over"):
        world.step(safe_action)


def test_foundational_mechanics_change_identity_causality_and_navigation(tmp_path: Path):
    object_folder = create_foundational_world(
        tmp_path, OBJECT_LABORATORY_ADAPTER, seed=1, mechanism="occlusion"
    )
    object_world = FOUNDATIONAL_LOADERS[OBJECT_LABORATORY_ADAPTER](object_folder)
    before = object_world.reset(11002)
    initial = object_world.audit_state()
    for _ in range(6):
        after = object_world.step(0).observation
    final = object_world.audit_state()
    assert initial["phase"] == 0 and final["phase"] == 6
    assert not np.array_equal(before, after)

    functional_folder = create_foundational_world(
        tmp_path, OBJECT_LABORATORY_ADAPTER, seed=4, mechanism="functional-category"
    )
    functional = FOUNDATIONAL_LOADERS[OBJECT_LABORATORY_ADAPTER](functional_folder)
    functional.reset(11004)
    effect = functional.step(2)
    assert effect.public_info["event"] == "effect-observed"
    assert functional.audit_state()["probed"]
    shifted = functional.step(0)
    assert shifted.public_info["event"] == "appearance-shifted"
    assert functional.audit_state()["active_mechanism"] == "functional-category"

    causal_folder = create_foundational_world(tmp_path, CAUSAL_WORKSHOP_ADAPTER, seed=2)
    causal_world = FOUNDATIONAL_LOADERS[CAUSAL_WORKSHOP_ADAPTER](causal_folder)
    causal_world.reset(21001)
    first_controls = causal_world.audit_state()["advance_actions"]
    causal_world.reset(21002)
    assert causal_world.audit_state()["advance_actions"] != first_controls

    maze_folder = create_foundational_world(tmp_path, CHANGING_MAZE_ADAPTER, seed=3,
                                             mechanism="revision")
    maze_world = FOUNDATIONAL_LOADERS[CHANGING_MAZE_ADAPTER](maze_folder)
    maze_world.reset(31002)
    before_map = maze_world.audit_state()["action_map"]
    for _ in range(8):
        change = maze_world.step(5)
    state = maze_world.audit_state()
    assert change.public_info["event"] == "world-changed"
    assert state["changed"] and state["action_map"] != before_map
    assert state["change_cell"] in state["walls"]


def test_strict_loader_rejects_executable_or_unknown_package_fields(tmp_path: Path):
    folder = create_foundational_world(tmp_path, OBJECT_LABORATORY_ADAPTER, seed=707)
    path = folder / "genome.private.json"
    value = json.loads(path.read_text())
    value["module"] = "untrusted.py"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(FoundationalWorldError, match="unknown=.*module"):
        FOUNDATIONAL_LOADERS[OBJECT_LABORATORY_ADAPTER](folder)


def test_harness_registers_and_loads_all_foundational_adapters(tmp_path: Path):
    harness = GUMHarness(tmp_path / "workspace")
    assert set(ADAPTERS).issubset(harness.status()["registered_adapters"])
    folder = harness.create_school_world(OBJECT_LABORATORY_ADAPTER, seed=808)
    assert harness.status()["active_world"]["world_id"] == folder.name


def test_all_six_lesson_generators_resolve_to_the_declared_adapter(tmp_path: Path):
    curriculum = load_json(DEFAULT_CURRICULUM)
    for index, lesson in enumerate(curriculum["lessons"]):
        folder = create_lesson_world(tmp_path, lesson, seed=900_000 + index)
        world = FOUNDATIONAL_LOADERS[lesson["adapter"]](folder)
        assert world.public_spec().adapter == lesson["adapter"]
        assert world.mechanism != "mixed"


def test_foundational_admission_suite_passes_without_training(tmp_path: Path):
    report = run_foundational_admission(working_directory=tmp_path, trials=8)
    assert report["passed"]
    assert report["training_performed"] is False
    assert report["sealed_worlds_generated"] is False
    assert len(report["adapters"]) == 3
    for row in report["adapters"]:
        assert row["passed"] and all(row["checks"].values())
        assert row["audit_boundary"]["policy_received_audit_state"] is False
        scripted = row["baselines"]["scripted-public-observation"]
        random = row["baselines"]["random"]
        assert scripted["success_rate"] >= random["success_rate"]
        assert (scripted["success_rate"] > random["success_rate"]
                or scripted["mean_steps"] < random["mean_steps"])


def test_saved_admission_evidence_matches_curriculum_hashes():
    root = DEFAULT_CURRICULUM.parents[1]
    evidence = root / "evidence" / "gum-school" / "admissions" / "foundational-v1.json"
    curriculum = load_json(DEFAULT_CURRICULUM)
    report = load_json(evidence)
    expected = f"sha256:{hashlib.sha256(evidence.read_bytes()).hexdigest()}"
    assert all(row["status"] == "admitted" for row in curriculum["world_admissions"])
    assert all(row["evidence_hashes"] == [expected] for row in curriculum["world_admissions"])
    for name, digest in report["source_hashes"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert digest == f"sha256:{actual}"
