"""Development contracts for visual layout learning on pre-existing worlds."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from jepa_asteroids.adaptive_visual_agent import AdaptiveUniversalAgent, LearnedLayoutAdapter, dashboard_embed
from jepa_asteroids.immune_savior import ImmuneSaviorGame


def run_raw(agent, game, budget=2400):
    used = 0; success = False; final = {}
    while used < budget and not success:
        source = game.reset(); frame = dashboard_embed(source, seed=game.seed)
        agent.begin(frame)
        for _ in range(min(300, budget - used)):
            action, _ = agent.act(frame); nxt, reward, done, info = game.step(action)
            raw_next = dashboard_embed(nxt, seed=game.seed)
            agent.observe(frame, action, raw_next, reward, done); frame = raw_next; used += 1; final = info
            if done: success = bool(info["success"]); break
    return success, used, final


class AdaptiveVisualAgentTests(unittest.TestCase):
    def test_adapter_discovers_unprovided_layout(self):
        game = ImmuneSaviorGame(602); raw = dashboard_embed(game.reset(), top=37, left=151, scale=2, seed=9)
        adapter = LearnedLayoutAdapter(); canonical = adapter.adapt(raw)
        self.assertEqual(canonical.shape, (3, 126, 112)); self.assertEqual(adapter.geometry, (2, 37, 151))

    def test_adaptive_agent_solves_transformed_old_world_and_round_trips(self):
        agent = AdaptiveUniversalAgent(); success, _, info = run_raw(agent, ImmuneSaviorGame(602))
        self.assertTrue(success); self.assertEqual(info["healthy_damaged"], 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mind.json"; agent.save(path); restored = AdaptiveUniversalAgent.load(path)
        success, used, info = run_raw(restored, ImmuneSaviorGame(602), 700)
        self.assertTrue(success); self.assertLess(used, 200); self.assertEqual(info["healthy_damaged"], 0)


if __name__ == "__main__": unittest.main()
