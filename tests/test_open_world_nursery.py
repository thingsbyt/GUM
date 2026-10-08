"""Contracts for open-world visual causal discovery."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from jepa_asteroids.open_world_nursery import (
    OpenWorldMind, OpenWorldNursery, _run_until_success, perceive,
    run_open_world_benchmark,
)


class OpenWorldNurseryTests(unittest.TestCase):
    def test_pixels_hide_private_object_names(self):
        world = OpenWorldNursery(101); frame = world.reset(202); view = perceive(frame)
        self.assertEqual(frame.shape, (3, 112, 176))
        self.assertEqual(len(view["objects"]), 6)
        self.assertNotIn("v0", repr(view))

    def test_self_generated_experiments_discover_and_plan(self):
        mind = OpenWorldMind(); world = OpenWorldNursery(303)
        result = _run_until_success(mind, world, 420, 404)
        self.assertTrue(result["success"])
        self.assertGreaterEqual(mind.status()["causal_rules_discovered"], 5)
        self.assertGreater(mind.status()["self_generated_experiments"], 0)

    def test_memory_round_trip_retains_rules(self):
        mind = OpenWorldMind(); _run_until_success(mind, OpenWorldNursery(505), 420, 606)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mind.json"; mind.save(path); restored = OpenWorldMind.load(path)
        result = _run_until_success(restored, OpenWorldNursery(505, goal="branch"), 30, 707)
        self.assertTrue(result["success"])
        self.assertLessEqual(result["interactions"], 3)

    def test_small_frozen_audit_beats_random_and_revisits(self):
        with tempfile.TemporaryDirectory() as folder:
            report = run_open_world_benchmark(Path(folder) / "audit.json", worlds=3, budget=420, seed=808)
        self.assertEqual(report["aggregate"]["open_world_mind"]["success_rate"], 1.0)
        self.assertEqual(report["aggregate"]["later_revisit_new_goals"]["success_rate"], 1.0)
        self.assertGreater(report["growth"]["causal_rules_discovered"], 0)


if __name__ == "__main__": unittest.main()
