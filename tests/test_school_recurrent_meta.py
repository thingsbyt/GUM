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
