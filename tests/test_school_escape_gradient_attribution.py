"""Contracts for frozen cooperative gradient attribution."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gum.protocol import PublicWorldSpec, Transition
from gum.school.escape_gradient_attribution import (
    _apply_isolated_update,
    _component_gradients,
    _gradients,
    _gradient_alignment,
    _parameter_deltas,
    _recorded_graph,
    _state_copy,
)
from gum.school.recurrent_meta import RecurrentCausalLearner, RecurrentMetaConfig


pytestmark = pytest.mark.neural


class _ForbiddenInfo:
    def __bool__(self):
        raise AssertionError("hidden info must remain unused")


def _spec() -> PublicWorldSpec:
    return PublicWorldSpec(
        "attribution-test",
        "cooperative-test",
        "attribution-test-v1",
        1,
        "rgb",
        (24, 24, 3),
        "discrete-anonymous",
        5,
        8,
        (-1.0, 1.0),
    )


def _trajectory_and_checkpoint(tmp_path: Path):
    config = RecurrentMetaConfig(
        hidden_size=12,
        visual_width=10,
        action_memory_size=6,
        batch_episodes=1,
        training_exploration_mix=0.2,
    )
    learner = RecurrentCausalLearner(707, config=config)
    checkpoint = tmp_path / "before.pt"
    learner.save(checkpoint)
    observations = [
        np.full((24, 24, 3), value, dtype=np.uint8)
        for value in (10, 50, 100, 180)
    ]
    rewards = [-0.001, -0.001, 0.999, -0.001]
    actions = []
    learner.begin(_spec(), observations[0], training=True)
    for index, (observation, reward) in enumerate(
        zip(observations, rewards, strict=True)
    ):
        action = learner.act(observation, training=True)
        actions.append(action)
        learner.observe(
            action,
            Transition(
                observations[min(index + 1, len(observations) - 1)],
                reward,
                False,
                index == len(observations) - 1,
                _ForbiddenInfo(),
            ),
            training=True,
        )
    learner.finish_episode(training=True)
    trajectory = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "ticks": [1, 2, 3, 4],
        "action_count": 5,
    }
    return checkpoint, learner, trajectory


def test_component_losses_sum_to_original_and_gradients_sum(tmp_path: Path):
    checkpoint, _, trajectory = _trajectory_and_checkpoint(tmp_path)
    learner = RecurrentCausalLearner.load(checkpoint)
    graph = _recorded_graph(learner, trajectory)
    torch.testing.assert_close(
        graph["losses"]["full"],
        graph["losses"]["actor"]
        + graph["losses"]["value"]
        + graph["losses"]["entropy"],
    )
    vectors = _component_gradients(checkpoint, trajectory, device="cpu")
    alignment = _gradient_alignment(vectors)
    assert alignment["full_minus_component_sum_norm"]["all"] < 1e-6


def test_full_replay_matches_and_value_update_leaves_policy_head_untouched(
    tmp_path: Path,
):
    checkpoint, online, trajectory = _trajectory_and_checkpoint(tmp_path)
    full = RecurrentCausalLearner.load(checkpoint)
    full.fit_recorded_actor_critic(trajectory)
    for expected, actual in zip(
        online.policy.parameters(), full.policy.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual, rtol=0.0, atol=1e-7)

    value_only = RecurrentCausalLearner.load(checkpoint)
    before = _state_copy(value_only)
    metrics, _ = _apply_isolated_update(value_only, trajectory, "value")
    deltas = _parameter_deltas(before, value_only)
    assert metrics["component"] == "value"
    assert float(torch.linalg.vector_norm(deltas["policy_head"])) == 0.0
    assert float(torch.linalg.vector_norm(deltas["visual_encoder"])) > 0.0


def test_zero_value_shared_gradient_scale_trains_only_value_head(tmp_path: Path):
    checkpoint, _, trajectory = _trajectory_and_checkpoint(tmp_path)
    learner = RecurrentCausalLearner.load(checkpoint)
    learner.config = RecurrentMetaConfig(
        **{
            **learner.config.__dict__,
            "value_shared_gradient_scale": 0.0,
        }
    )
    learner.policy.config = learner.config
    graph = _recorded_graph(learner, trajectory)
    vectors = _gradients(learner, graph["losses"]["value"])

    assert float(torch.linalg.vector_norm(vectors["visual_encoder"])) == 0.0
    assert float(torch.linalg.vector_norm(vectors["recurrent_layers"])) == 0.0
    assert float(torch.linalg.vector_norm(vectors["policy_head"])) == 0.0
    assert float(torch.linalg.vector_norm(vectors["value_head"])) > 0.0
