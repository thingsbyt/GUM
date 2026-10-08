"""Contracts for position-independent visual concepts and causal plans."""
from __future__ import annotations

import unittest

from jepa_asteroids.concept_memory import ConceptMemory, concept_state_key, object_concept


def obj(color, area, x, y, w=2, h=2):
    return {"color": color, "area": area, "cx": float(x), "cy": float(y),
            "w": w, "h": h, "sig": (color, area, w, h)}


class ConceptMemoryTests(unittest.TestCase):
    def test_object_and_scene_concepts_ignore_absolute_position(self):
        first = [obj(2, 4, 5, 5), obj(3, 4, 15, 5)]
        shifted = [obj(2, 4, 25, 25), obj(3, 4, 35, 25)]
        self.assertEqual(object_concept(first[0]), object_concept(shifted[0]))
        self.assertEqual(concept_state_key(first), concept_state_key(shifted))

    def test_delayed_events_become_abstract_dependency_plan(self):
        memory = ConceptMemory(); key = obj(1, 1, 5, 5, 1, 1); exit_obj = obj(9, 8, 15, 5, 4, 2)
        memory.observe([key, exit_obj], 3, [exit_obj], False)
        memory.observe([exit_obj], 4, [], True)
        status = memory.status()
        self.assertGreaterEqual(status["learned_dependencies"], 1)
        self.assertEqual(status["successful_concept_plans"], 1)
        memory.reset_episode()
        key_score, _ = memory.target_priority(key)
        unrelated_score, _ = memory.target_priority(obj(7, 1, 30, 30, 1, 1))
        self.assertGreater(key_score, unrelated_score)

    def test_grown_ontology_round_trips(self):
        memory = ConceptMemory(); item = obj(6, 4, 4, 4)
        memory.observe([item], 1, [], True)
        restored = ConceptMemory(); restored.restore(memory.export())
        self.assertEqual(restored.status()["object_and_relation_concepts"],
                         memory.status()["object_and_relation_concepts"])
        self.assertEqual(restored.status()["successful_concept_plans"], 1)


if __name__ == "__main__": unittest.main()
