"""Contracts for visual grounding and never-seen semantic compositions."""
from __future__ import annotations

import unittest
import numpy as np

from jepa_asteroids.semantic_composer import SemanticSkillComposer, visual_symbol_sequence


GLYPHS = (
    ((1, 1, 1), (0, 1, 0), (0, 1, 0)),
    ((1, 0, 0), (1, 1, 1), (0, 0, 1)),
    ((1, 1, 0), (0, 1, 0), (0, 1, 1)),
    ((1, 0, 1), (1, 1, 1), (0, 1, 0)),
    ((1, 1, 1), (1, 0, 0), (1, 1, 1)),
)


def goal(tokens, color=4, offset=1):
    frame = np.zeros((1, 9, 4 + len(tokens) * 6), dtype=np.uint8)
    for position, token in enumerate(tokens):
        mask = np.asarray(GLYPHS[token], dtype=bool); x = offset + position * 6
        frame[0, 2:5, x:x + 3][mask] = color
    return frame


def event(kind, color=3):
    identity = f"object:c{color}:a2:w2:h2"
    return {
        "entity-appeared": f"event:appeared:{identity}",
        "entity-disappeared": f"event:disappeared:{identity}",
        "entity-moved-horizontal": f"event:moved:{identity}:dx1:dy0",
        "relation-created": f"event:relation-created:{identity}:near:other",
    }[kind]


class SemanticComposerTests(unittest.TestCase):
    def test_visual_tokens_ignore_color_and_position(self):
        self.assertEqual(visual_symbol_sequence(goal([0, 3], 4, 1)),
                         visual_symbol_sequence(goal([0, 3], 12, 2)))
        self.assertNotEqual(visual_symbol_sequence(goal([0, 3])),
                            visual_symbol_sequence(goal([3, 0])))

    def test_successes_ground_words_then_compose_unseen_order(self):
        model = SemanticSkillComposer()
        meanings = ["entity-appeared", "entity-moved-horizontal",
                    "entity-disappeared", "relation-created"]
        for repeat in range(2):
            for left in range(4):
                right = (left + 1 + repeat) % 4
                model.reset_episode("teaching", goal([left, right]))
                model.observe(left + 1, [event(meanings[left], 2 + repeat)], False, False)
                model.observe(right + 1, [event(meanings[right], 7 + repeat)], True, True)
        model.reset_episode("novel", goal([3, 1, 0, 2]))
        self.assertEqual(model.goal_semantics,
                         [meanings[3], meanings[1], meanings[0], meanings[2]])

    def test_composed_goal_grounds_new_controls_by_intervention(self):
        model = SemanticSkillComposer(); meanings = ["entity-appeared",
            "entity-moved-horizontal", "entity-disappeared", "relation-created"]
        tokens = visual_symbol_sequence(goal([0, 1, 2, 3]))
        for token, meaning in zip(tokens, meanings): model.token_evidence[token][meaning] = 3
        model.reset_episode("remapped", goal([3, 1, 0, 2]))
        desired = [meanings[3], meanings[1], meanings[0], meanings[2]]
        mapping = {1: meanings[2], 2: meanings[0], 3: meanings[3], 4: meanings[1]}
        stage = 0
        for _ in range(50):
            action, _ = model.recommend([1, 2, 3, 4]); outcome = mapping[action]
            stage = stage + 1 if outcome == desired[stage] else (1 if outcome == desired[0] else 0)
            solved = stage == len(desired)
            model.observe(action, [event(outcome)], solved, solved)
            if solved: break
        self.assertTrue(solved)
        self.assertGreater(model.status()["counterfactual_queries"], 0)

    def test_curriculum_prefers_unknown_symbols_and_memory_round_trips(self):
        model = SemanticSkillComposer(); known = visual_symbol_sequence(goal([0]))[0]
        model.token_evidence[known]["entity-appeared"] = 3
        self.assertEqual(model.choose_curriculum([goal([0]), goal([4])]), 1)
        payload = model.export(); restored = SemanticSkillComposer(); restored.restore(payload)
        self.assertEqual(restored.status()["grounded_symbol_meanings"], 1)
        self.assertEqual(restored.choose_curriculum([goal([0]), goal([4])]), 1)


if __name__ == "__main__":
    unittest.main()
