"""Contracts for the persistent four-member mission swarm."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from gum.protocol import PublicWorldSpec, Transition
from gum.school import mission_swarm as mission_swarm_module
from gum.school.mission_swarm import MissionSwarm, MissionSwarmError
from gum.school.recurrent_meta import RecurrentCausalLearner, RecurrentMetaConfig


pytestmark = pytest.mark.neural


def _source(path: Path) -> Path:
    RecurrentCausalLearner(
        70,
        config=RecurrentMetaConfig(
            hidden_size=16,
            action_memory_size=8,
            visual_width=8,
            batch_episodes=1,
        ),
    ).save(path)
    return path


def _spec() -> PublicWorldSpec:
    return PublicWorldSpec(
        "public-mission-world",
        "changing-maze",
        "gum-changing-maze-v1",
        1,
        "rgb",
        (88, 88, 3),
        "discrete-anonymous",
        6,
        240,
        (-1.0, 1.0),
    )


def test_four_members_keep_identities_share_success_and_round_trip(tmp_path: Path):
    root = tmp_path / "swarm"
    swarm = MissionSwarm.create(
        root,
        source_policy=_source(tmp_path / "source.pt"),
        share_epochs=1,
    )
    observation = np.zeros((88, 88, 3), dtype=np.uint8)
    changed = observation.copy()
    changed[8:16, 8:16] = 255
    started = swarm.begin_mission(_spec(), observation, training=True)
    action = swarm.act(observation)
    swarm.observe(
        action,
        Transition(changed, 1.0, True, False, {}),
    )
    capsule = swarm.finish_mission(success=True)

    assert len(swarm.members) == 4
    assert len({member.identity.member_id for member in swarm.members}) == 4
    assert len({member.identity.fingerprint for member in swarm.members}) == 4
    assert capsule["mission_id"] == started["mission_id"]
    assert len(capsule["shared_imports"]) == 3
    assert swarm.communication_rounds == 1
    assert all(member.missions == 1 for member in swarm.members)
    assert sum(member.shared_imports for member in swarm.members) == 3
    assert (root / "snapshots" / capsule["mission_id"]).is_dir()
    experience_path = root / capsule["experience"]["path"]
    assert experience_path.is_file()
    assert capsule["experience"]["format"] == "gum-mission-experience-v2"
    with np.load(experience_path, allow_pickle=False) as experience:
        assert set(experience.files) == {
            "observations", "next_observations", "actions", "rewards",
            "terminated", "truncated", "proposals", "confidences", "selected",
        }
        assert np.array_equal(experience["next_observations"], changed[None])
        assert experience["terminated"].tolist() == [True]
        assert experience["truncated"].tolist() == [False]
    assert swarm.status()["experience_archive"] == {
        "valid": True,
        "records": 1,
        "errors": [],
    }

    restored = MissionSwarm.load(root)
    assert restored.status()["team_size"] == 4
    assert restored.status()["completed_missions"] == 1
    assert restored.status()["mission_ledger"]["valid"] is True
    assert restored.status()["knowledge_ledger"]["valid"] is True
    assert restored.status()["experience_archive"]["valid"] is True


def test_failure_is_kept_and_changes_captain_before_retry(tmp_path: Path):
    root = tmp_path / "swarm"
    swarm = MissionSwarm.create(root, source_policy=_source(tmp_path / "source.pt"))
    observation = np.zeros((88, 88, 3), dtype=np.uint8)
    first = swarm.begin_mission(_spec(), observation, training=True)
    action = swarm.act(observation)
    swarm.observe(action, Transition(observation, -1.0, True, False, {}))
    failed = swarm.finish_mission(success=False)
    second = swarm.begin_mission(_spec(), observation, training=True)

    assert failed["strategy_changed_before_exact_retry"] is True
    assert first["captain_id"] != second["captain_id"]
    assert swarm.strategy_revisions == 1
    assert (root / "snapshots" / failed["mission_id"]).is_dir()
    assert (root / failed["experience"]["path"]).is_file()


def test_distinct_missions_balance_captaincy(tmp_path: Path):
    swarm = MissionSwarm.create(
        tmp_path / "swarm",
        source_policy=_source(tmp_path / "source.pt"),
    )
    observation = np.zeros((88, 88, 3), dtype=np.uint8)
    captain_ids = []
    for index in range(4):
        spec = replace(_spec(), world_id=f"public-mission-world-{index}")
        started = swarm.begin_mission(spec, observation, training=True)
        captain_ids.append(started["captain_id"])
        action = swarm.act(observation)
        swarm.observe(action, Transition(observation, -1.0, True, False, {}))
        swarm.finish_mission(success=False)

    assert len(set(captain_ids)) == 4
    assert [member.captain_missions for member in swarm.members] == [1, 1, 1, 1]


def test_reload_recovers_checkpoint_when_final_pointer_write_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "swarm"
    swarm = MissionSwarm.create(root, source_policy=_source(tmp_path / "source.pt"))
    observation = np.zeros((88, 88, 3), dtype=np.uint8)
    started = swarm.begin_mission(_spec(), observation, training=True)
    action = swarm.act(observation)
    swarm.observe(action, Transition(observation, 1.0, True, False, {}))

    real_atomic_write_json = mission_swarm_module.atomic_write_json
    interrupted = False

    def interrupt_pointer_once(path, value, **kwargs):
        nonlocal interrupted
        if Path(path) == root / "MISSION_SWARM.json" and not interrupted:
            interrupted = True
            raise OSError("simulated loss before final pointer replacement")
        return real_atomic_write_json(path, value, **kwargs)

    monkeypatch.setattr(
        mission_swarm_module, "atomic_write_json", interrupt_pointer_once
    )
    with pytest.raises(OSError, match="simulated loss"):
        swarm.finish_mission(success=True)
    assert interrupted is True

    restored = MissionSwarm.load(root)
    assert restored.completed_missions == 1
    assert all(member.missions == 1 for member in restored.members)
    pointer = json.loads((root / "MISSION_SWARM.json").read_text(encoding="utf-8"))
    assert pointer["checkpoint"] == started["mission_id"]
    assert pointer["recovered_after_incomplete_pointer_update"] is True
    assert restored.status()["mission_ledger"]["records"] == 3
    assert restored.status()["knowledge_ledger"]["records"] == 1


def test_reload_rejects_ledger_state_without_a_matching_checkpoint(tmp_path: Path):
    root = tmp_path / "swarm"
    swarm = MissionSwarm.create(root, source_policy=_source(tmp_path / "source.pt"))
    observation = np.zeros((88, 88, 3), dtype=np.uint8)
    swarm.begin_mission(_spec(), observation, training=True)

    with pytest.raises(MissionSwarmError, match="checkpoint/ledger inconsistency"):
        MissionSwarm.load(root)
