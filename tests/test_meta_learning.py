import tempfile
import unittest
from pathlib import Path

import numpy as np

from gum.meta_learning import FAMILIES, MetaLearningMind


class MetaLearningTests(unittest.TestCase):
    def test_strategy_evolves_without_retaining_solutions(self):
        mind = MetaLearningMind(77, 14)
        for index in range(16):
            mind.learn_task(510_000 + index * 1009, FAMILIES[index % 4], 1200)
        evaluation = mind.evaluate([(810_000 + index * 2017, FAMILIES[index % 4]) for index in range(12)], 1200)
        evolved = [row["evolved"]["interactions"] if row["evolved"]["success"] else 1800
                   for row in evaluation["rows"]]
        initial = [row["initial"]["interactions"] if row["initial"]["success"] else 1800
                   for row in evaluation["rows"]]
        self.assertLess(np.mean(evolved), np.mean(initial))
        self.assertEqual(mind.solutions_retained, 0)
        self.assertNotEqual(evaluation["champion"]["weights"], [0.0, 0.0, 0.0])

        scratch = Path(__file__).parent / ".scratch"; scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / "meta-mind.json"; mind.save(path)
            restored = MetaLearningMind.load(path)
        self.assertEqual(restored.champion().genome_id, mind.champion().genome_id)
        self.assertEqual(restored.solutions_retained, 0)


if __name__ == "__main__":
    unittest.main()
