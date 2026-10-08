"""Contracts for identity-free causal transfer across visual contexts."""
from __future__ import annotations

import unittest

from jepa_asteroids.semantic_world_model import SharedSemanticWorldModel, semantic_event


def raw(kind: str, identity: str = "object:c7:a3:w2:h2") -> str:
    return {
        "entity-appeared": f"event:appeared:{identity}",
        "entity-disappeared": f"event:disappeared:{identity}",
        "entity-moved-horizontal": f"event:moved:{identity}:dx1:dy0",
        "entity-moved-vertical": f"event:moved:{identity}:dx0:dy-1",
        "relation-created": f"event:relation-created:{identity}:near:other",
        "relation-destroyed": f"event:relation-destroyed:{identity}:near:other",
    }[kind]


class SemanticWorldModelTests(unittest.TestCase):
    def test_semantics_discard_visual_identity(self):
        self.assertEqual(semantic_event(raw("entity-appeared", "object:c2:a1:w1:h1")),
                         semantic_event(raw("entity-appeared", "object:c15:a5:w4:h2")))
        self.assertEqual(semantic_event(raw("entity-moved-horizontal")),
                         "entity-moved-horizontal")

    def test_repeated_success_isolates_stable_suffix_from_incidental_history(self):
        model = SharedSemanticWorldModel()
        grammar = ["entity-appeared", "entity-moved-horizontal", "entity-disappeared"]
        for episode, distractor in enumerate(("relation-created", "entity-moved-vertical")):
            model.reset_episode("source")
            model.observe("source", 9, [raw(distractor)], False, False)
            for index, kind in enumerate(grammar):
                model.observe("source", index + 1, [raw(kind, f"object:c{episode + index + 2}:a2:w2:h2")],
                              index == len(grammar) - 1, index == len(grammar) - 1)
        self.assertEqual(model.status()["best_template"], grammar)
        self.assertEqual(model.status()["best_template_support"], 2)

    def test_transferred_grammar_is_grounded_to_remapped_controls(self):
        model = SharedSemanticWorldModel()
        grammar = ["entity-appeared", "entity-moved-horizontal", "entity-disappeared"]
        model.templates[tuple(grammar)] = 3
        mapping = {1: "entity-moved-vertical", 2: "entity-moved-horizontal",
                   3: "entity-appeared", 4: "entity-disappeared"}
        model.reset_episode("novel")
        stage = 0; solved = False
        for _ in range(120):
            plan = model.recommend("novel", [1, 2, 3, 4])
            action = plan[0] if plan else 1 + (_ % 4)
            outcome = mapping[action]
            if outcome == grammar[stage]:
                stage += 1
            else:
                stage = 1 if outcome == grammar[0] else 0
            solved = stage == len(grammar)
            model.observe("novel", action, [raw(outcome)], solved, solved)
            if solved:
                break
        self.assertTrue(solved)
        self.assertGreater(model.status()["counterfactual_queries"], 0)
        self.assertGreater(model.context_matches["novel"], 0)

    def test_snapshot_round_trip_preserves_shared_theory(self):
        model = SharedSemanticWorldModel(); model.templates[("entity-appeared", "entity-disappeared")] = 4
        model.context_effects["world"]["0"][3]["entity-appeared"] = 2
        payload = model.export(); restored = SharedSemanticWorldModel(); restored.restore(payload)
        self.assertEqual(restored.status()["best_template"],
                         ["entity-appeared", "entity-disappeared"])
        self.assertEqual(restored.context_effects["world"]["0"][3]["entity-appeared"], 2)


if __name__ == "__main__":
    unittest.main()
