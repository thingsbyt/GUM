"""Persistence and learner-boundary tests for the four-body escape team."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from gum.school import escape_team as escape_team_module
from gum.school.escape_chamber import RewardTable
from gum.school.escape_team import EscapeTeam, EscapeTeamError
from gum.school.recurrent_meta import RecurrentMetaConfig


pytestmark = pytest.mark.neural


def _team(root: Path) -> EscapeTeam:
    return EscapeTeam.create(
        root,
        seeds=(101, 211, 307, 401),
        config=RecurrentMetaConfig(
            hidden_size=16,
            action_memory_size=8,
            visual_width=8,
            pooled_size=4,
            batch_episodes=1,
        ),
        reward_table=RewardTable(treatment="team"),
    )


def test_four_bodies_execute_separate_actions_and_archive_complete_transitions(tmp_path: Path):
    root = tmp_path / "team"
    team = _team(root)
    seen = []
    capsule = team.run_episode(
        seed=501,
        training=True,
        horizon=6,
        on_step=lambda world, row: seen.append(row),
    )
    assert len(team.members) == 4
    assert len({member.identity.member_id for member in team.members}) == 4
    assert capsule["steps"] == 6 and len(seen) == 6
    assert capsule["sharing_mode"] == "off"
    assert capsule["scripted_control_used"] is False
    path = root / capsule["archive"]["path"]
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "observations", "next_observations", "actions", "rewards",
            "terminated", "truncated", "active_before", "active_after",
            "state_hashes",
        }
        assert archive["observations"].shape == (6, 4, 96, 160, 3)
        assert archive["next_observations"].shape == archive["observations"].shape
        assert archive["actions"].shape == (6, 4)
        assert archive["rewards"].shape == (6, 4)
        assert archive["truncated"][-1]
    assert team.verify_archives() == {"valid": True, "records": 1, "errors": []}
    assert [member.learner.training_interactions for member in team.members] == [6] * 4

    restored = EscapeTeam.load(root)
    assert restored.completed_episodes == 1
    assert restored.sharing_mode == "off"
    assert all(member.episodes == 1 for member in restored.members)
    assert restored.verify_archives()["valid"]


def test_pointer_interruption_recovers_matching_episode_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "team"
    team = _team(root)
    real_write = escape_team_module.atomic_write_json
    interrupted = False

    def interrupt_once(path, value, **kwargs):
        nonlocal interrupted
        if Path(path) == root / "ESCAPE_TEAM.json" and not interrupted:
            interrupted = True
            raise OSError("simulated final pointer interruption")
        return real_write(path, value, **kwargs)

    monkeypatch.setattr(escape_team_module, "atomic_write_json", interrupt_once)
    with pytest.raises(OSError, match="simulated final pointer"):
        team.run_episode(seed=601, training=True, horizon=4)
    restored = EscapeTeam.load(root)
    assert restored.completed_episodes == 1
    pointer = json.loads((root / "ESCAPE_TEAM.json").read_text(encoding="utf-8"))
    assert pointer["recovered_after_incomplete_pointer_update"] is True
    assert restored.status()["episode_ledger"]["records"] == 2


def test_reload_rejects_ledger_advance_without_coherent_brains(tmp_path: Path):
    root = tmp_path / "team"
    team = _team(root)
    team.ledger.append("episode-completed", {"incomplete": True}, run_id="fault")
    with pytest.raises(EscapeTeamError, match="checkpoint/ledger inconsistency"):
        EscapeTeam.load(root)


def test_inherited_weights_are_declared_but_stored_as_separate_learners(tmp_path: Path):
    source_team = _team(tmp_path / "source-team")
    source = tmp_path / "source.pt"
    source_team.members[0].learner.save(source)
    team = EscapeTeam.create(
        tmp_path / "inherited",
        source_policy=source,
        seeds=(701, 702, 703, 704),
    )
    assert team.initialization["kind"] == "identical-inherited-policy-weights"
    assert team.initialization["optimizer_state_inherited"] is False
    assert team.initialization["separate_copies_are_prior_independent_learning"] is False
    first = list(team.members[0].learner.policy.parameters())
    for member in team.members[1:]:
        for expected, actual in zip(first, member.learner.policy.parameters(), strict=True):
            assert np.array_equal(expected.detach().cpu().numpy(), actual.detach().cpu().numpy())
