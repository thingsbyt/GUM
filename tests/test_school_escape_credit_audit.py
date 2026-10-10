"""Contracts for the bounded cooperative-escape credit audit."""

import numpy as np

from gum.school.escape_credit_audit import _member_trajectories


def test_member_trajectories_preserve_delayed_team_reward():
    arrays = {
        "observations": np.zeros((3, 4, 4, 4, 3), dtype=np.uint8),
        "actions": np.asarray([
            [1, 2, 3, 4],
            [5, 1, 2, 3],
            [-1, 4, 5, 1],
        ]),
        "rewards": np.asarray([
            [-0.001, -0.001, -0.001, -0.001],
            [0.999, 0.999, 0.999, 0.999],
            [1.0, -0.001, -0.001, -0.001],
        ]),
        "active_before": np.asarray([
            [True, True, True, True],
            [True, True, True, True],
            [False, True, True, True],
        ]),
    }

    trajectories = _member_trajectories(arrays)

    assert trajectories[0]["ticks"] == [1, 2]
    assert trajectories[0]["actions"] == [1, 5]
    assert trajectories[0]["rewards"] == [-0.001, 1.999]
    assert trajectories[0]["delayed_rewards"] == [{
        "source_tick": 3,
        "credited_to_action_tick": 2,
        "reward": 1.0,
    }]
    assert trajectories[1]["ticks"] == [1, 2, 3]
    assert trajectories[1]["delayed_rewards"] == []
