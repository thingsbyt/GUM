"""Deterministic smoke contract for the semantic-transfer audit."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from jepa_asteroids.semantic_transfer_benchmark import run_benchmark


class SemanticTransferBenchmarkTests(unittest.TestCase):
    def test_report_is_reproducible_and_contains_ablations(self):
        with tempfile.TemporaryDirectory() as folder:
            first = run_benchmark(Path(folder) / "a.json", tasks=3, horizon=80)
            second = run_benchmark(Path(folder) / "b.json", tasks=3, horizon=80)
        self.assertEqual(first["aggregate"], second["aggregate"])
        self.assertEqual(set(first["aggregate"]),
                         {"semantic_transfer", "no_cross_task_memory", "numeric_action_replay"})
        self.assertEqual(first["learned_theory"]["best_template"],
                         ["entity-appeared", "entity-moved-horizontal",
                          "entity-disappeared", "relation-created"])


if __name__ == "__main__":
    unittest.main()
