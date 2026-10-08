"""Contracts for the persistent partially-observable workshop."""
from __future__ import annotations
import tempfile
from pathlib import Path
import unittest

from jepa_asteroids.persistent_workshop import (
    PersistentWorkshop, WorkshopMind, _run, perceive_workshop, run_workshop_benchmark,
)


class PersistentWorkshopTests(unittest.TestCase):
    def test_pixels_hide_world_state_and_objects_leave_view(self):
        world = PersistentWorkshop(11); frame = world.reset(); view = perceive_workshop(frame)
        self.assertEqual(frame.shape, (3, 126, 112)); self.assertEqual(len(view["floor"] | view["walls"]), 25)
        self.assertNotIn("raw0", repr(view))

    def test_mind_grounds_controls_remembers_and_solves(self):
        mind = WorkshopMind(); result = _run(mind, PersistentWorkshop(22), 900)
        self.assertTrue(result["success"]); self.assertTrue(result["controls_grounded"])
        self.assertGreater(result["remembered_objects"], 4); self.assertGreater(mind.status()["self_generated_subgoals"], 0)

    def test_persistent_brain_recomposes_changed_goal(self):
        mind = WorkshopMind(); _run(mind, PersistentWorkshop(33), 900)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "brain.json"; mind.save(path); restored = WorkshopMind.load(path)
        result = _run(restored, PersistentWorkshop(33, goal="branch"), 180)
        self.assertTrue(result["success"])

    def test_small_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            report = run_workshop_benchmark(Path(folder) / "audit.json", worlds=2, budget=900, seed=44)
        self.assertEqual(report["aggregate"]["growing_workshop_mind"]["success_rate"], 1.0)
        self.assertEqual(report["aggregate"]["old_worlds_changed_goal"]["success_rate"], 1.0)


if __name__ == "__main__": unittest.main()
