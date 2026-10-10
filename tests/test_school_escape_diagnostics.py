from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from gum.school.escape_chamber import TEAM_SIZE, WAIT, EscapeChamberRoomA, RewardTable
from gum.school.escape_diagnostics import (
    DIAGNOSTIC_FORMAT,
    _replay_metrics,
    diagnose_team,
    probe_visual_mechanism,
)
from gum.school.escape_team import EscapeTeam
from gum.school.recurrent_meta import RecurrentMetaConfig


pytestmark = pytest.mark.neural


def _tiny_team(root: Path) -> EscapeTeam:
    return EscapeTeam.create(
        root,
        seeds=(81, 82, 83, 84),
        config=RecurrentMetaConfig(
            hidden_size=8, action_memory_size=4, visual_width=4, pooled_size=4,
            batch_episodes=1,
        ),
    )


def test_replay_metrics_count_plate_entry_gate_open_and_waiting():
    world = EscapeChamberRoomA(seed=441, horizon=3)
    observations = world.reset()
    rows = {name: [] for name in (
        "observations", "next_observations", "actions", "rewards", "terminated",
        "truncated", "active_before", "active_after", "state_hashes",
    )}
    meanings = ["west", "south", "south"]
    for meaning in meanings:
        actions = [world.action_slot_for_test(meaning)] + [
            world.action_slot_for_test(WAIT)
        ] * 3
        step = world.step(actions)
        rows["observations"].append(np.stack(observations))
        rows["next_observations"].append(np.stack(step.observations))
        rows["actions"].append(actions)
        rows["rewards"].append(step.rewards)
        rows["terminated"].append(step.terminated)
        rows["truncated"].append(step.truncated)
        rows["active_before"].append(step.active_before)
        rows["active_after"].append(step.active_after)
        rows["state_hashes"].append(
            world.audit_state()["state_sha256"].removeprefix("sha256:").encode()
        )
        observations = step.observations
    arrays = {name: np.asarray(values) for name, values in rows.items()}
    capsule = {
        "episode_id": "diagnostic-fixture", "seed": 441, "training": False,
        "escaped_count": 0, "updates": [None] * TEAM_SIZE,
        "archive": {"environment_seed": 441},
    }

    result = _replay_metrics(capsule, arrays, RewardTable())

    assert result["replay_verified"] is True
    assert result["gate_open_events"] == 1
    assert result["gate_open_ticks"] == 1
    assert result["members"][0]["plate_entries"] == 1
    assert result["members"][0]["longest_plate_dwell"] == 1
    assert result["members"][1]["wait_actions"] == 3
    assert result["members"][1]["longest_wait_streak"] == 3


def test_team_diagnosis_is_read_only_and_reports_parameter_change(tmp_path: Path):
    team = _tiny_team(tmp_path / "team")
    team.run_episode(seed=991, training=True, horizon=4)
    before = (team.root / "ESCAPE_TEAM.json").read_bytes()

    report = diagnose_team(team.root)

    assert report["format"] == DIAGNOSTIC_FORMAT
    assert report["measurement_only"] is True
    assert report["aggregate"]["episodes"] == 1
    assert report["aggregate"]["total_joint_ticks"] == 4
    assert report["episodes"][0]["replay_verified"] is True
    assert all(row["brain_file_hash_changed"] for row in report["parameter_changes"])
    assert (team.root / "ESCAPE_TEAM.json").read_bytes() == before


def test_frozen_encoder_probe_keeps_labels_outside_policy_training(tmp_path: Path):
    team = _tiny_team(tmp_path / "team")
    report = probe_visual_mechanism(team.root, configurations=8)
    assert report["measurement_only"] is True
    assert report["labels_used_for_policy_training"] is False
    assert report["frozen_encoder"] is True
    assert report["matched_birth_encoder_control"] is True
    assert len(report["members"]) == TEAM_SIZE
    for member in report["members"]:
        assert 0.0 <= member["current_gate_visual_only"]["held_out_accuracy"] <= 1.0
        assert member["current_gate_visual_only"]["mean_paired_embedding_l2"] > 0.0
        assert 0.0 <= member["birth_gate_visual_only"]["held_out_accuracy"] <= 1.0
