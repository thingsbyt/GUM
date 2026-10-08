import tempfile
import unittest
from pathlib import Path

from gum.concept_skill_factory import ConceptSkillFactory, MechanismWorld


class ConceptSkillFactoryTests(unittest.TestCase):
    def test_invents_stable_opaque_concepts_and_reuses_a_skill(self):
        mind = ConceptSkillFactory()
        target = (4, 1, 3)
        first = MechanismWorld(101, target, 0)
        acquired = mind.acquire(first)

        self.assertTrue(acquired["success"])
        self.assertEqual(len(mind.concepts), 5)
        self.assertTrue(all(row.concept_id.startswith("concept-") for row in mind.concepts.values()))
        self.assertEqual({row.signature for row in mind.concepts.values()},
                         {(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)})

        revisit = mind.revisit(first)
        self.assertTrue(revisit["success"])
        self.assertEqual(revisit["interactions"], 3)
        self.assertLess(revisit["interactions"], acquired["interactions"])

        variant = mind.solve_with_skills(MechanismWorld(202, target, 2))
        self.assertTrue(variant["success"])
        self.assertEqual(len(mind.concepts), 5)
        self.assertEqual(mind.mapping_branches, 0)

    def test_persistence_and_portable_sharing_exclude_control_maps(self):
        mind = ConceptSkillFactory()
        target = (0, 2, 4)
        self.assertTrue(mind.acquire(MechanismWorld(303, target, 1))["success"])

        scratch_root = Path(__file__).parent / ".scratch"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as directory:
            path = Path(directory) / "mind.json"
            mind.save(path)
            loaded = ConceptSkillFactory.load(path)
            self.assertTrue(loaded.revisit(MechanismWorld(303, target, 1))["success"])

        portable = mind.share_portable_knowledge()
        self.assertNotIn("context_memory", portable)
        self.assertNotIn("action_to_concept", str(portable))
        recipient = ConceptSkillFactory()
        recipient.import_portable_knowledge(portable)
        result = recipient.solve_with_skills(MechanismWorld(404, target, 0))
        self.assertTrue(result["success"])
        self.assertEqual(len(recipient.context_memory), 1)

    def test_composes_two_prior_skills_for_a_longer_world(self):
        mind = ConceptSkillFactory()
        left = (0, 3, 1)
        right = (4, 2, 0)
        self.assertTrue(mind.acquire(MechanismWorld(501, left, 0))["success"])
        self.assertTrue(mind.acquire(MechanismWorld(502, right, 1))["success"])

        result = mind.solve_composed(MechanismWorld(503, left + right, 2))
        self.assertTrue(result["success"])
        self.assertEqual(len(mind.skills), 3)
        self.assertEqual(len(mind.skills[result["skill_id"]].sequence), 6)
        self.assertLessEqual(result["compositions_tried"], 4)


if __name__ == "__main__":
    unittest.main()
