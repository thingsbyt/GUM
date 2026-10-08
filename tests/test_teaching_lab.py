from pathlib import Path
import tempfile
import unittest

from gum.teaching_lab import TeachingLab


class TeachingLabTests(unittest.TestCase):
    def folder(self):
        return tempfile.TemporaryDirectory(dir=Path.cwd())

    def test_starter_lessons_persist_and_resolve_ambiguity(self):
        with self.folder() as folder:
            root = Path(folder); lab = TeachingLab(root)
            learned = lab.starter_curriculum()
            self.assertEqual(learned["learned_word_count"], 6)

            lab.new_scene("ambiguity")
            question = lab.begin("approach the red object")
            self.assertEqual(question["last_turn"]["status"], "clarify")
            self.assertIn("square", question["last_turn"]["response"])

            resolved = lab.answer("the square")
            self.assertEqual(resolved["last_turn"]["status"], "execute")
            self.assertEqual(resolved["last_turn"]["target_id"], "object-1")

            reopened = TeachingLab(root)
            self.assertEqual(reopened.state()["learned_word_count"], 6)

    def test_pointing_example_is_saved_and_reset_is_explicit(self):
        with self.folder() as folder:
            lab = TeachingLab(Path(folder))
            # The lesson scene always places one object at this display coordinate.
            state = lab.demonstrate("please choose this thing", 125, 150)
            self.assertEqual(state["examples"], 1)
            self.assertTrue(lab.memory_path.exists())

            reset = lab.reset()
            self.assertEqual(reset["examples"], 0)
            self.assertEqual(reset["learned_word_count"], 0)

    def test_reference_audit_runs_a_fresh_learner(self):
        with self.folder() as folder:
            report = TeachingLab(Path(folder)).audit()
            self.assertTrue(report["passed"])
            self.assertEqual(report["results"]["overall_successes"], 45)
            self.assertEqual(report["results"]["overall_trials"], 45)


if __name__ == "__main__":
    unittest.main()
