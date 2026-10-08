from pathlib import Path
import tempfile
import unittest

from gum.clarification_challenge import (
    ClarifyingAgent, ObjectSpec, VisualWordGrounder, render_scene,
    run_challenge, teach_visual_words,
)


def learned_agent() -> ClarifyingAgent:
    lexicon = VisualWordGrounder(); teach_visual_words(lexicon)
    return ClarifyingAgent(lexicon)


class ClarificationChallengeTests(unittest.TestCase):
    def test_words_are_induced_from_pixels_and_pointing_demonstrations(self):
        agent = learned_agent(); learned = agent.lexicon.word_to_feature
        self.assertLessEqual(set(("red", "blue", "green", "circle", "square", "triangle")),
                             set(learned))
        self.assertNotIn("the", learned); self.assertNotIn("approach", learned)

    def test_ambiguous_reference_asks_and_uses_answer(self):
        scene = render_scene([
            ObjectSpec("red", "circle", (25, 30)), ObjectSpec("red", "square", (70, 30)),
            ObjectSpec("blue", "triangle", (48, 70)),
        ])
        agent = learned_agent(); first = agent.begin(scene, "approach the red object")
        self.assertEqual(first.status, "clarify")
        self.assertIn("circle", first.response); self.assertIn("square", first.response)
        second = agent.answer("the square")
        self.assertEqual(second.status, "execute"); self.assertEqual(second.target_id, "object-1")

    def test_clear_unknown_and_no_pending_cases_do_not_guess(self):
        scene = render_scene([ObjectSpec("green", "triangle", (48, 48))])
        agent = learned_agent()
        self.assertEqual(agent.begin(scene, "approach the green triangle").status, "execute")
        unknown = agent.begin(scene, "approach the crimson object")
        self.assertEqual(unknown.status, "clarify"); self.assertIn("crimson", unknown.unknown_words)
        self.assertEqual(agent.answer("the triangle").status, "clarify")

    def test_challenge_passes_and_memory_round_trips(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder); report = run_challenge(root, ambiguous_trials=12, clear_trials=6)
            self.assertTrue(report["passed"])
            self.assertEqual(report["results"]["ambiguous_resolved"], 12)
            loaded = VisualWordGrounder.load(root / "CLARIFICATION_LANGUAGE_MEMORY.json")
            self.assertEqual(loaded.word_to_feature, report["learned_lexicon"])
