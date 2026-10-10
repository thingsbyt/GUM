"""Contracts for the bounded multi-team gradient-correction study."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from gum.school.escape_chamber import DEVELOPMENT_ADAPTER, RewardTable
from gum.school.escape_gradient_study import _evaluate_branch
from gum.school.recurrent_meta import RecurrentCausalLearner, RecurrentMetaConfig


pytestmark = pytest.mark.neural


def test_frozen_evaluation_preserves_complete_transitions(tmp_path: Path):
    config = RecurrentMetaConfig(hidden_size=12, visual_width=10, action_memory_size=6)
    learners = [RecurrentCausalLearner(800 + index, config=config) for index in range(4)]
    result = _evaluate_branch(
        learners,
        seeds=[771001],
        horizon=5,
        adapter=DEVELOPMENT_ADAPTER,
        reward_table=RewardTable(),
        output=tmp_path,
        team_name="team-test",
        branch="control",
    )

    assert result["episodes"] == 1
    episode = result["per_seed"][0]
    archive = tmp_path / episode["transition_archive"]["path"]
    with np.load(archive, allow_pickle=False) as saved:
        assert saved["observations"].shape[0] == episode["joint_ticks"]
        assert saved["next_observations"].shape == saved["observations"].shape
        assert saved["actions"].shape == (episode["joint_ticks"], 4)
        assert saved["rewards"].shape == (episode["joint_ticks"], 4)
        assert saved["terminated"].shape == (episode["joint_ticks"],)
        assert saved["truncated"].shape == (episode["joint_ticks"],)
