import json

from PIL import Image
import pytest

pytest.importorskip("torch")

from gum.school.escape_recording import RECORDING_FORMAT, record_episode_gif
from gum.school.escape_team import EscapeTeam
from gum.school.escape_study_viewer import StudyViewer
from gum.school.recurrent_meta import RecurrentMetaConfig


pytestmark = pytest.mark.neural


def test_recording_replays_every_archived_tick_and_writes_sidecar(tmp_path):
    config = RecurrentMetaConfig(
        hidden_size=8, action_memory_size=4, visual_width=4, pooled_size=4,
        batch_episodes=1,
    )
    team = EscapeTeam.create(tmp_path / "team", config=config, device="cpu")
    capsule = team.run_episode(seed=611, training=False, horizon=3)
    output = tmp_path / "episode.gif"

    result = record_episode_gif(team.root, output, fps=10, width=320)

    assert result["format"] == RECORDING_FORMAT
    assert result["episode_id"] == capsule["episode_id"]
    assert result["joint_ticks"] == 3
    assert result["recorded_frames"] == 4
    assert result["every_joint_tick_included"] is True
    assert result["archive_replay_verified"] is True
    with Image.open(output) as image:
        assert image.n_frames == 4
    sidecar = json.loads(output.with_suffix(".gif.json").read_text(encoding="utf-8"))
    assert sidecar["recording_sha256"].startswith("sha256:")
    assert sidecar["learning_method"] == "GUM"
    assert sidecar["display_phase"] == "replay of frozen evaluation"
    viewer = StudyViewer(team.root)
    entry = viewer.episodes()[0]
    viewer.replay(key=entry["key"])
    for tick in range(3):
        state = viewer.replay(tick=tick)
        assert state["phase"] == "replay of frozen evaluation"
        assert state["cumulative"]["evaluation_joint_ticks"] == tick+1
        assert state["cumulative"]["current_episode_individual_actions"] == 4*(tick+1)
