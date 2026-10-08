"""Contracts for the task-independent online explorer."""
from __future__ import annotations

import unittest
import numpy as np
from pathlib import Path
import tempfile

from jepa_asteroids.concept_memory import object_concept
from jepa_asteroids.semantic_composer import visual_symbol_sequence
from jepa_asteroids.universal_explorer import (
    UniversalExplorer, _objects_exact, state_key, visual_context_key)


def frame(position: int, jitter: int = 0):
    value = np.zeros((1, 16, 16), dtype=np.uint8)
    value[0, 7:9, position:position + 2] = 5
    if jitter: value[0, 0, 0] = jitter
    return value


class UniversalExplorerTests(unittest.TestCase):
    def test_state_identity_ignores_single_pixel_noise(self):
        self.assertEqual(state_key(frame(4)), state_key(frame(4, 3)))

    def test_visual_context_tracks_world_structure_but_not_object_position(self):
        self.assertEqual(visual_context_key(frame(4)), visual_context_key(frame(9)))
        different = np.zeros((1, 16, 16), dtype=np.uint8)
        different[0, 5:9, 7] = 5
        self.assertNotEqual(visual_context_key(frame(4)), visual_context_key(different))

    def test_initial_interventions_cover_controls(self):
        agent = UniversalExplorer(3); obs = frame(4); actions = [1, 2, 3, 4]; selected = []
        for _ in range(15):
            choice, _ = agent.act(obs, actions, 0); selected.append(choice)
            agent.observe(obs, choice, obs, 0.0, False, 0, 0)
        self.assertEqual(set(selected), set(actions))

    def test_grounded_visual_goal_can_control_the_shared_agent(self):
        agent = UniversalExplorer(4); obs = frame(4); agent.reset_episode(0, obs)
        goal = np.zeros((1, 8, 8), dtype=np.uint8)
        goal[0, 2:5, 2] = 7; goal[0, 2, 2:5] = 7
        token = visual_symbol_sequence(goal)[0]
        agent.semantic_composer.token_evidence[token]["entity-appeared"] = 3
        agent.semantic_composer.context_effects[agent.current_context][2]["entity-appeared"] = 3
        agent.set_visual_goal(goal)
        action, evidence = agent.act(obs, [1, 2], 0)
        self.assertEqual(action, 2)
        self.assertEqual(evidence["reason"], "execute-composed-semantic-goal")

    def test_complex_action_explores_visual_coordinates(self):
        agent = UniversalExplorer(5); obs = frame(4); points = []
        for _ in range(12):
            action, evidence = agent.act(obs, [6], 0)
            self.assertEqual(action, 6); point = evidence["action_data"]
            self.assertTrue(0 <= point["x"] <= 63 and 0 <= point["y"] <= 63)
            points.append((point["x"], point["y"]))
            agent.observe(obs, action, obs, 0.0, False, 0, 0)
        self.assertGreaterEqual(len(set(points)), 8)

    def test_terminal_click_assigns_state_specific_point_hazard(self):
        agent = UniversalExplorer(6); obs = frame(4); agent.reset_episode(0, obs)
        action, _ = agent.act(obs, [6], 0)
        agent.observe(obs, action, obs, -1.0, True, 0, 0)
        self.assertGreater(agent.status()["point_interaction"]["hazardous_points"], 0)

    def test_effectful_object_concept_is_selected_again(self):
        obs = np.zeros((1, 64, 64), dtype=np.uint8)
        obs[0, 8:13, 8:13] = 14; obs[0, 8:13, 28:33] = 14
        obs[0, 40:43, 45:48] = 3
        blue = next(obj for obj in _objects_exact(obs[0]) if obj["color"] == 14)
        concept = object_concept(blue); agent = UniversalExplorer(23)
        agent.click_effect_sum[concept] = 26.0 / obs[0].size
        agent.click_effect_count[concept] = 1
        point = agent._point_action(obs)
        self.assertTrue((8 <= point["x"] <= 12) or (28 <= point["x"] <= 32))
        self.assertTrue(8 <= point["y"] <= 12)

    def test_click_created_object_receives_causal_credit(self):
        before = np.zeros((1, 32, 32), dtype=np.uint8); before[0, 4:9, 4:9] = 14
        after = np.zeros((1, 32, 32), dtype=np.uint8); after[0, 20:23, 20:23] = 7
        spawned = object_concept(_objects_exact(after[0])[0]); agent = UniversalExplorer(29)
        agent.pending_click = {"x": 6, "y": 6, "concept": "gate", "state": "s",
                               "point_key": "s:6:6"}
        agent.observe(before, 6, after, 0.0, False, 0, 0)
        self.assertGreater(agent.click_spawn_credit[spawned], 0)

    def test_effectful_sweep_visits_each_visible_instance_once(self):
        obs = np.zeros((1, 64, 64), dtype=np.uint8)
        obs[0, 8:13, 8:13] = 14; obs[0, 8:13, 28:33] = 14
        objects = _objects_exact(obs[0]); concept = object_concept(objects[0])
        agent = UniversalExplorer(31); agent.click_effect_sum[concept] = 26.0 / 4096
        agent.click_effect_count[concept] = 1
        self.assertTrue(agent._effectful_click_available(objects, 4096))
        for obj in objects:
            agent.episode_effectful_instances.add(
                agent._click_instance_key(concept, obj["cx"], obj["cy"]))
        self.assertFalse(agent._effectful_click_available(objects, 4096))
        self.assertFalse(agent._effectful_click_available([], 4096))
        self.assertTrue(agent._effectful_click_available(objects, 4096))

    def test_unvalidated_dominant_action_is_overridden(self):
        agent = UniversalExplorer(37); obs = frame(4); agent.reset_episode(0, obs)
        agent.episode_action_counts[1] = 13
        agent.macro_action = 1; agent.macro_remaining = 1
        agent.hypothesis_engine.recommend = lambda *args: ("graph-frontier", None)
        agent.causal_discovery.replay = lambda *args: None
        agent.visual_goal_planner.recommend = lambda *args, **kwargs: None
        choice, evidence = agent.act(obs, [1, 2], 0)
        self.assertEqual(choice, 2)
        self.assertEqual(evidence["reason"], "restore-control-diversity")

    def test_failed_episodes_trigger_broad_control_search_until_progress(self):
        agent = UniversalExplorer(11); obs = frame(4)
        agent.reset_episode(0); self.assertEqual(agent.exploration_regime, "model-based")
        agent.reset_episode(0); self.assertEqual(agent.exploration_regime, "broad-random")
        _, evidence = agent.act(obs, [1, 6], 0)
        self.assertIn(evidence["reason"].split(":", 1)[0],
                      ("causal-information-experiment", "replay-novelty-frontier"))
        agent.context_progress["default"] = 1; agent.reset_episode(0)
        self.assertEqual(agent.exploration_regime, "model-based")

    def test_unrelated_visual_contexts_isolate_and_restore_skill_modules(self):
        agent = UniversalExplorer(13); world_a = frame(4)
        world_b = np.zeros((1, 16, 16), dtype=np.uint8); world_b[0, 5:9, 7] = 5
        agent.reset_episode(0, world_a)
        action, _ = agent.act(world_a, [1, 2], 0)
        agent.observe(world_a, action, world_a, 1.0, False, 0, 1)
        learned_actions = dict(agent.global_actions)
        agent.reset_episode(0, world_b)
        self.assertEqual(dict(agent.global_actions), {})
        agent.reset_episode(0, world_a)
        self.assertEqual(dict(agent.global_actions), learned_actions)
        self.assertGreater(agent.status()["remembered_progress_states"], 0)

    def test_all_skill_modules_survive_snapshot_round_trip(self):
        agent = UniversalExplorer(17); world_a = frame(4)
        world_b = np.zeros((1, 16, 16), dtype=np.uint8); world_b[0, 5:9, 7] = 5
        agent.reset_episode(0, world_a)
        action, _ = agent.act(world_a, [1, 2], 0)
        agent.observe(world_a, action, world_a, 1.0, False, 0, 1)
        remembered_a = dict(agent.global_actions)
        agent.reset_episode(0, world_b)
        action, _ = agent.act(world_b, [3, 4], 0)
        agent.observe(world_b, action, world_b, 1.0, False, 0, 1)
        remembered_b = dict(agent.global_actions)
        agent.click_spawn_credit["created:test"] = 9.0
        agent.semantic_world_model.templates[("entity-appeared", "entity-disappeared")] = 3
        agent.semantic_composer.token_evidence["glyph:test"]["entity-appeared"] = 3
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "brain.json"; agent.save(path)
            restored = UniversalExplorer.load(path)
        self.assertEqual(dict(restored.global_actions), remembered_b)
        self.assertEqual(restored.click_spawn_credit["created:test"], 9.0)
        self.assertEqual(restored.status()["semantic_world_model"]["best_template"],
                         ["entity-appeared", "entity-disappeared"])
        self.assertEqual(restored.status()["semantic_composer"]["grounded_symbol_meanings"], 1)
        restored.reset_episode(0, world_a)
        self.assertEqual(dict(restored.global_actions), remembered_a)
        self.assertEqual(restored.status()["stored_skill_modules"], 2)

    def test_progress_path_is_remembered_and_terminal_path_is_revised(self):
        agent = UniversalExplorer(7); a, _ = agent.act(frame(2), [1, 2], 0)
        agent.observe(frame(2), a, frame(6), 1.0, False, 0, 1)
        self.assertGreater(agent.status()["remembered_progress_states"], 0)
        remembered, evidence = agent.act(frame(2), [1, 2], 0)
        self.assertEqual(remembered, a); self.assertEqual(evidence["reason"], "replay-validated-procedure")
        agent.observe(frame(2), remembered, frame(2), -1.0, True, 0, 0)
        self.assertEqual(agent.status()["remembered_progress_states"], 0)


if __name__ == "__main__": unittest.main()
