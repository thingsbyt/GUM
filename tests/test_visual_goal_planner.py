from __future__ import annotations

import unittest

import numpy as np

from jepa_asteroids.universal_explorer import _objects_exact
from jepa_asteroids.visual_goal_planner import VisualGoalPlanner


def block(grid, color, x, y, size=3):
    grid[y:y + size, x:x + size] = color


class VisualGoalPlannerTests(unittest.TestCase):
    def test_selects_correct_body_when_one_action_moves_bodies_oppositely(self):
        grid = np.full((32, 32), 9, dtype=np.uint8)
        block(grid, 4, 5, 10); block(grid, 5, 20, 10); block(grid, 11, 27, 10)
        objects = _objects_exact(grid)
        signatures = {obj["color"]: obj["sig"] for obj in objects}
        motion = {
            signatures[4]: {1: [(3, 0)] * 2, 2: [(-3, 0)] * 2,
                            3: [(0, -3)] * 2, 4: [(0, 3)] * 2},
            signatures[5]: {1: [(-3, 0)] * 2, 2: [(3, 0)] * 2,
                            3: [(0, -3)] * 2, 4: [(0, 3)] * 2},
        }
        planner = VisualGoalPlanner()
        recommendation = planner.recommend(grid, objects, [1, 2, 3, 4], motion, {9})
        self.assertEqual(recommendation, (2, "infer-multi-body-visual-goal"))
        self.assertEqual(planner.status()["multi_body_plan_steps"], 1)

    def test_approaching_visual_region_triggers_safe_gap_prediction(self):
        grid = np.full((32, 32), 9, dtype=np.uint8)
        block(grid, 4, 13, 13)
        grid[20:23, :] = 15; grid[20:23, 15:27] = 9
        objects = _objects_exact(grid)
        signature = next(obj["sig"] for obj in objects if obj["color"] == 4)
        danger_signature = next(obj["sig"] for obj in objects if obj["color"] == 15)
        motion = {signature: {1: [(-3, 0)] * 2, 2: [(3, 0)] * 2},
                  danger_signature: {1: [(0, -2)] * 2, 2: [(0, -2)] * 2}}
        planner = VisualGoalPlanner(); planner.color_approach[15] = [-2.0, -2.0, -2.0]
        recommendation = planner.recommend(grid, objects, [1, 2], motion, {9})
        self.assertEqual(recommendation, (2, "predict-visual-collision"))
        self.assertGreater(planner.status()["dynamic_avoidance_steps"], 0)

    def test_learned_hazards_and_goal_models_survive_round_trip(self):
        planner = VisualGoalPlanner(); planner.hazard_colors[15] = 1.5
        planner.relation_attempts["a->b"] = 3; planner.relation_success["a->b"] = 1
        planner.action_feature_deltas[2].append([.1] * 8)
        planner.progress_feature_direction = np.asarray([.2] * 8); planner.progress_feature_examples = 2
        restored = VisualGoalPlanner(); restored.restore(planner.export())
        self.assertEqual(restored.hazard_colors[15], 1.5)
        self.assertEqual(restored.relation_attempts["a->b"], 3)
        self.assertEqual(restored.progress_feature_examples, 2)
        self.assertTrue(np.allclose(restored.progress_feature_direction, [.2] * 8))

    def test_beam_search_rejects_positions_that_round_outside_image(self):
        grid = np.full((16, 16), 9, dtype=np.uint8)
        block(grid, 4, 1, 5); block(grid, 5, 12, 5); block(grid, 11, 12, 10)
        objects = _objects_exact(grid)
        signatures = {obj["color"]: obj["sig"] for obj in objects}
        motion = {signatures[4]: {1: [(-3, 0)] * 2, 2: [(3, 0)] * 2,
                                  3: [(0, -3)] * 2, 4: [(0, 3)] * 2},
                  signatures[5]: {1: [(1.7, 0)] * 2, 2: [(-3, 0)] * 2,
                                  3: [(0, -3)] * 2, 4: [(0, 3)] * 2}}
        planner = VisualGoalPlanner()
        recommendation = planner.recommend(grid, objects, [1, 2, 3, 4], motion, {9})
        self.assertIsNotNone(recommendation)

    def test_large_sweeping_panel_is_not_treated_as_a_controlled_body(self):
        planner = VisualGoalPlanner()
        compact = (4, 3, 3, 3); sweeping = (15, 8, 15, 15)
        motion = {compact: {1: [(-3, 0)] * 2, 2: [(3, 0)] * 2},
                  sweeping: {1: [(0, -3)] * 2, 2: [(0, -6)] * 2}}
        controllable = planner._controllable(motion, [1, 2])
        self.assertIn(compact, controllable)
        self.assertNotIn(sweeping, controllable)

    def test_terminal_occlusion_keeps_effector_for_hazard_credit(self):
        before = np.full((24, 24), 9, dtype=np.uint8); block(before, 4, 10, 10)
        after = before.copy(); after[10:13, :] = 15; after[10:13, 10:13] = 15
        objects = _objects_exact(before); signature = next(obj["sig"] for obj in objects if obj["color"] == 4)
        motion = {signature: {1: [(-2, 0)] * 2, 2: [(2, 0)] * 2}}
        planner = VisualGoalPlanner()
        planner.observe(before, 1, before, False, False, objects, objects, motion)
        planner.observe(before, 1, after, False, True, objects, _objects_exact(after), motion)
        self.assertGreater(planner.hazard_colors[15], 0)

    def test_large_divider_is_route_structure_not_a_contact_goal(self):
        grid = np.full((40, 40), 9, dtype=np.uint8)
        block(grid, 4, 5, 5); block(grid, 5, 25, 5); block(grid, 11, 30, 25)
        grid[:, 19:22] = 10
        objects = _objects_exact(grid)
        signatures = {obj["color"]: obj["sig"] for obj in objects if obj["color"] in (4, 5)}
        motion = {signatures[4]: {1: [(0, -3)] * 2, 2: [(0, 3)] * 2,
                                  3: [(-3, 0)] * 2, 4: [(3, 0)] * 2},
                  signatures[5]: {1: [(0, -3)] * 2, 2: [(0, 3)] * 2,
                                  3: [(3, 0)] * 2, 4: [(-3, 0)] * 2}}
        planner = VisualGoalPlanner(); planner.recommend(grid, objects, [1, 2, 3, 4], motion, {9})
        self.assertNotIn("->10,", planner.current_goal["key"])

    def test_multi_body_goal_waits_for_two_control_axes(self):
        grid = np.full((24, 24), 9, dtype=np.uint8)
        block(grid, 4, 3, 3); block(grid, 5, 12, 3); block(grid, 11, 18, 15)
        objects = _objects_exact(grid); signatures = {o["color"]: o["sig"] for o in objects}
        motion = {signatures[4]: {1: [(-3, 0)] * 2, 2: [(3, 0)] * 2},
                  signatures[5]: {1: [(3, 0)] * 2, 2: [(-3, 0)] * 2}}
        planner = VisualGoalPlanner()
        self.assertIsNone(planner.recommend(grid, objects, [1, 2], motion, {9}))

    def test_composite_avatar_parts_do_not_trigger_multi_body_planning(self):
        grid = np.full((24, 24), 9, dtype=np.uint8)
        block(grid, 4, 3, 3); block(grid, 5, 12, 3); block(grid, 11, 18, 15)
        objects = _objects_exact(grid); signatures = {o["color"]: o["sig"] for o in objects}
        shared = {1: [(0, -3)] * 2, 2: [(0, 3)] * 2,
                  3: [(-3, 0)] * 2, 4: [(3, 0)] * 2}
        motion = {signatures[4]: dict(shared), signatures[5]: dict(shared)}
        planner = VisualGoalPlanner()
        self.assertIsNone(planner.recommend(grid, objects, [1, 2, 3, 4], motion, {9}))
        self.assertEqual(planner.status()["multi_body_plan_steps"], 0)

    def test_joint_planner_moves_opposed_bodies_toward_paired_targets(self):
        grid = np.full((32, 32), 9, dtype=np.uint8)
        block(grid, 4, 3, 10); block(grid, 5, 25, 10)
        block(grid, 11, 10, 10); block(grid, 11, 18, 10)
        objects = _objects_exact(grid)
        signatures = {obj["color"]: obj["sig"] for obj in objects if obj["color"] in (4, 5)}
        motion = {signatures[4]: {1: [(3, 0)] * 2, 2: [(-3, 0)] * 2,
                                  3: [(0, -3)] * 2, 4: [(0, 3)] * 2},
                  signatures[5]: {1: [(-3, 0)] * 2, 2: [(3, 0)] * 2,
                                  3: [(0, -3)] * 2, 4: [(0, 3)] * 2}}
        planner = VisualGoalPlanner()
        recommendation = planner._joint_multi_action(
            grid, objects, [1, 2, 3, 4], planner._controllable(motion, [1, 2, 3, 4]), {9})
        self.assertEqual(recommendation, (1, "infer-joint-visual-goal"))


if __name__ == "__main__":
    unittest.main()
