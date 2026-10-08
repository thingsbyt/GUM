import numpy as np
from unittest.mock import patch

from gum.resilient_generalist import ResilientGeneralist


def test_negative_terminal_creates_an_isolated_context_branch():
    class Stub:
        def observe(self, *args): return {"learned": True}
    mind = ResilientGeneralist(); mind.context = "world-x"; mind.active = Stub(); mind.shared = mind.active
    result = mind.observe(np.zeros((1,)), 0, np.zeros((1,)), -1.0, True)
    assert "world-x" in mind.branches
    assert result["context_failures"] == 1


def test_full_horizon_stagnation_creates_a_branch_on_next_begin():
    class Stub:
        def begin(self, frame): return None
    mind = ResilientGeneralist(); frame = np.zeros((3, 132, 112), dtype=np.uint8)
    mind.context = "stalled"; mind.active = mind.shared; mind.episode_open = True
    mind.episode_steps = 300; mind.episode_reward = 0.0
    mind._context = lambda _: "stalled"; mind.shared.begin = lambda _: None
    with patch("gum.resilient_generalist.AdaptiveUniversalAgent", return_value=Stub()): mind.begin(frame)
    assert "stalled" in mind.branches
