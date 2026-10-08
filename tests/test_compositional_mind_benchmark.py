"""Small deterministic contract for the compositional growth audit."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from jepa_asteroids.compositional_mind_benchmark import run_benchmark


class CompositionalMindBenchmarkTests(unittest.TestCase):
    def test_learns_dictionary_composes_and_revisits(self):
        with tempfile.TemporaryDirectory() as folder:
            report = run_benchmark(Path(folder) / "audit.json", tasks=8, horizon=80)
        self.assertEqual(report["curriculum"]["dictionary_before"], 0)
        self.assertEqual(report["curriculum"]["dictionary_after"], 5)
        self.assertEqual(report["composition_novelty"]["exact_heldout_commands_seen_during_training"], 0)
        self.assertGreater(report["aggregate"]["semantic_composer"]["success_rate"],
                           report["aggregate"]["no_cross_task_memory"]["success_rate"])
        self.assertLessEqual(report["aggregate"]["revisit_after_all_tasks"]["mean_steps"], 4.5)
        self.assertTrue(report["growth"]["dictionary_unchanged_after_heldout_and_revisit"])


if __name__ == "__main__":
    unittest.main()
