"""Contracts for the experimental reward-grounded recurrent meta-policy."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gum.protocol import PublicWorldSpec, Transition
from gum.school.recurrent_meta import RecurrentCausalLearner, RecurrentMetaConfig


pytestmark = pytest.mark.neural


def _spec() -> PublicWorldSpec:
    return PublicWorldSpec(
        "public-test-world",
        "causal-workshop",
        "gum-causal-workshop-v1",
        1,
        "rgb",
        (72, 72, 3),
        "discrete-anonymous",
        8,
        200,
        (-1.0, 1.0),
    )


def _maze_spec() -> PublicWorldSpec:
    return PublicWorldSpec(
        "public-maze-world",
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


class _ForbiddenInfo(dict):
    def get(self, *args, **kwargs):
        raise AssertionError("the learner read public_info")

    def __iter__(self):
        raise AssertionError("the learner iterated public_info")

    def items(self):
        raise AssertionError("the learner read public_info items")


def test_policy_does_not_read_event_labels():
    learner = RecurrentCausalLearner(7)
    observation = np.zeros((72, 72, 3), dtype=np.uint8)
    learner.begin(_spec(), observation, training=False)
    action = learner.act(observation, training=False)
    learner.observe(
        action,
        Transition(observation, 0.2, False, False, _ForbiddenInfo(event="progress")),
        training=False,
    )


def test_anonymous_controls_are_symmetric_until_their_outcomes_differ():
    learner = RecurrentCausalLearner(8)
    observation = np.zeros((72, 72, 3), dtype=np.uint8)
    learner.begin(_spec(), observation, training=False)
    logits, _, hidden, memory = learner._policy_step(observation)
    torch.testing.assert_close(logits, logits[:, :1].expand_as(logits))

    _, _, _, changed = learner.policy(
        torch.from_numpy(observation),
        torch.tensor([3]),
        torch.tensor([0.2]),
        torch.tensor([0.0]),
        hidden,
        memory,
    )
    assert torch.count_nonzero(changed[:, 3]).item() > 0
    assert torch.count_nonzero(changed[:, :3]).item() == 0
    assert torch.count_nonzero(changed[:, 4:]).item() == 0


def test_one_general_policy_masks_action_slots_absent_from_a_world():
    learner = RecurrentCausalLearner(12)
    observation = np.zeros((88, 88, 3), dtype=np.uint8)
    learner.begin(_maze_spec(), observation, training=False)
    for _ in range(20):
        assert 0 <= learner.act(observation, training=False) < 6
        assert 0.0 <= learner.confidence() <= 1.0


def test_episodic_novelty_rewards_new_pixels_without_public_labels():
    learner = RecurrentCausalLearner(
        13,
        config=RecurrentMetaConfig(
            batch_episodes=1,
            episodic_novelty_coefficient=0.2,
        ),
    )
    observation = np.zeros((88, 88, 3), dtype=np.uint8)
    novel = observation.copy()
    novel[0, 0, 0] = 1
    learner.begin(_maze_spec(), observation, training=True)
    action = learner.act(observation, training=True)
    learner.observe(
        action,
        Transition(novel, 0.0, False, False, _ForbiddenInfo()),
        training=True,
    )
    assert learner._rewards == pytest.approx([0.2])
    assert learner.intrinsic_reward_total == pytest.approx(0.2)


def test_generic_action_explorer_learns_effects_from_pixel_change():
    learner = RecurrentCausalLearner(
        14,
        config=RecurrentMetaConfig(episodic_action_exploration_mix=1.0),
    )
    observation = np.zeros((88, 88, 3), dtype=np.uint8)
    changed = observation.copy()
    changed[1, 1, 1] = 1
    learner.begin(_maze_spec(), observation, training=False)
    action = learner.act(observation, training=False)
    learner.observe(
        action,
        Transition(changed, 0.0, False, False, _ForbiddenInfo()),
        training=False,
    )
    assert learner._action_effect_trials[action] == 1.0
    assert learner._action_effect_changes[action] == 1.0
    assert sum(learner._episodic_action_counts.values())[action] == 1


def test_each_learner_owns_a_reproducible_sampling_stream():
    observation = np.zeros((72, 72, 3), dtype=np.uint8)
    left = RecurrentCausalLearner(15)
    right = RecurrentCausalLearner(15)
    left.begin(_spec(), observation, training=True)
    right.begin(_spec(), observation, training=True)
    assert [left.act(observation, training=True) for _ in range(8)] == [
        right.act(observation, training=True) for _ in range(8)
    ]


def test_training_sampling_stream_survives_checkpoint(tmp_path: Path):
    observation = np.zeros((72, 72, 3), dtype=np.uint8)
    learner = RecurrentCausalLearner(115)
    learner.begin(_spec(), observation, training=True)
    for _ in range(8):
        learner.act(observation, training=True)
    state_before_next_episode = learner._training_generator.get_state().clone()
    learner.begin(_spec(), observation, training=True)
    assert torch.equal(
        state_before_next_episode,
        learner._training_generator.get_state(),
    )
    path = tmp_path / "sampling-stream.pt"
    learner.save(path)
    restored = RecurrentCausalLearner.load(path)
    assert torch.equal(
        learner._training_generator.get_state(),
        restored._training_generator.get_state(),
    )
    learner.begin(_spec(), observation, training=True)
    restored.begin(_spec(), observation, training=True)
    assert [learner.act(observation, training=True) for _ in range(8)] == [
        restored.act(observation, training=True) for _ in range(8)
    ]


def test_delayed_team_reward_credits_last_action_without_new_interaction():
    observation = np.zeros((72, 72, 3), dtype=np.uint8)
    learner = RecurrentCausalLearner(16)
    learner.begin(_spec(), observation, training=True)
    action = learner.act(observation, training=True)
    learner.observe(
        action,
        Transition(observation, 1.0, False, False, _ForbiddenInfo()),
        training=True,
    )
    interactions = learner.training_interactions
    learner.credit_delayed_reward(0.5, training=True)
    assert learner._rewards == pytest.approx([1.5])
    assert learner.training_interactions == interactions
    assert learner.delayed_reward_events == 1


def test_checkpoint_round_trip(tmp_path: Path):
    learner = RecurrentCausalLearner(9, config=RecurrentMetaConfig(hidden_size=16))
    path = tmp_path / "policy.pt"
    learner.save(path)
    restored = RecurrentCausalLearner.load(path)
    assert restored.status() == learner.status()
    for expected, actual in zip(
        learner.policy.parameters(), restored.policy.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual)


def test_reward_event_imitation_selects_only_positive_reward_tokens():
    learner = RecurrentCausalLearner(
        10, config=RecurrentMetaConfig(hidden_size=16, action_memory_size=8)
    )
    observation = np.zeros((72, 72, 3), dtype=np.uint8)
    result = learner.fit_reward_event_imitation(
        [{
            "observations": [observation, observation, observation],
            "actions": [1, 2, 3],
            "rewards": [-0.01, 0.2, 1.0],
        }],
        epochs=1,
        batch_episodes=1,
    )
    assert result["selected_trajectories"] == 1
    assert result["positive_reward_tokens"] == 2
    assert result["updates"] == 1


def test_reward_outcome_replay_counts_scalar_supervision_tokens():
    learner = RecurrentCausalLearner(
        11, config=RecurrentMetaConfig(hidden_size=16, action_memory_size=8)
    )
    observation = np.zeros((72, 72, 3), dtype=np.uint8)
    result = learner.fit_reward_outcome_replay(
        [{
            "observations": [observation, observation, observation],
            "actions": [1, 2, 3],
            "rewards": [-0.01, 0.2, 1.0],
        }],
        epochs=1,
        batch_episodes=1,
    )
    assert result["selected_trajectories"] == 1
    assert result["positive_reward_tokens"] == 2
    assert result["negative_reward_tokens"] == 1
    assert result["updates"] == 1


def test_saved_research_result_is_a_development_pass_only():
    report_path = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "gum-school" / "research"
        / "causal-recurrent-meta-v1" / "RECURRENT_META_REPORT.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    learned = report["evaluation"]["trained_recurrent_policy"]
    random = report["evaluation"]["uniform_random_policy"]
    untrained = report["evaluation"]["same_untrained_network"]
    assert report["development_only"] is True
    assert report["sealed_data_used"] is False
    assert report["official_curriculum_run"] is False
    assert report["strict_pass"] is True
    assert learned["successes"] == 439
    assert learned["trials"] == 512
    assert learned["success_rate"] > random["success_rate"]
    assert learned["success_rate"] > untrained["success_rate"]
