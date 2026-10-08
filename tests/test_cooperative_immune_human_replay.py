import numpy as np

from jepa_asteroids.cooperative_immune_human_replay import CANVAS, _panel
from jepa_asteroids.cooperative_immune_savior import CooperativeImmuneSaviorGame, CooperativeImmuneTeam


def test_human_panel_does_not_change_private_agent_observations():
    game = CooperativeImmuneSaviorGame(2_410_003); frames = game.reset(); team = CooperativeImmuneTeam(); team.begin(frames)
    before = [frame.copy() for frame in frames]
    panel = _panel(frames, game, 0, team.status(), 0.0, "entering tissue", False, False)
    assert panel.size == CANVAS
    assert all(np.array_equal(left, right) for left, right in zip(before, frames))
