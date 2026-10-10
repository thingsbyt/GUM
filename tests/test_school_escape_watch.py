"""Watch interface state and security contracts."""
from pathlib import Path

import pytest

pytest.importorskip("torch")

from gum.school.escape_team import EscapeTeam
from gum.school.escape_watch import HTML, EscapeWatchSession, build_escape_watch_server
from gum.school.recurrent_meta import RecurrentMetaConfig


pytestmark = pytest.mark.neural


def _team(root: Path) -> EscapeTeam:
    return EscapeTeam.create(
        root,
        seeds=(11, 22, 33, 44),
        config=RecurrentMetaConfig(
            hidden_size=8, action_memory_size=4, visual_width=4, pooled_size=4,
            batch_episodes=1,
        ),
    )


def test_watch_is_truth_labeled_and_contains_required_live_controls():
    for text in (
        "Authoritative room", "Exact learner rasters", "Start learning",
        "Pause safely", "Single step", "Stop + checkpoint", "Sharing", "OFF",
        "FAILURES INCLUDED", "No LLM, scripted route, captain, or browser timing",
    ):
        assert text in HTML


def test_preview_is_explicitly_not_training_and_has_real_frames(tmp_path: Path):
    session = EscapeWatchSession(_team(tmp_path / "team"))
    state = session.state()
    assert state["mode"] == "preview · not training"
    assert state["weights_updating"] is False
    assert state["sharing_mode"] == "off"
    assert len(state["members"]) == 4
    assert session.frame_png().startswith(b"\x89PNG")
    assert session.frame_png(0).startswith(b"\x89PNG")


def test_watch_server_rejects_non_loopback_binding(tmp_path: Path):
    with pytest.raises(ValueError, match="loopback"):
        build_escape_watch_server(_team(tmp_path / "team"), host="0.0.0.0", port=0)
