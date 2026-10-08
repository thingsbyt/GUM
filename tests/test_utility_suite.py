"""End-to-end contracts for ten practical learned workflows."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from jepa_asteroids.utility_suite import run_suite


class UtilitySuiteTests(unittest.TestCase):
    def test_all_ten_tasks_verify_record_and_improve(self):
        with tempfile.TemporaryDirectory() as folder:
            report = run_suite(Path(folder))
        self.assertEqual(report["tasks"], 10)
        self.assertTrue(report["all_first_runs_verified"])
        self.assertTrue(report["all_revisits_verified"])
        self.assertEqual(report["optimal_revisits"], 10)
        self.assertEqual(report["videos_created"], 20)
        self.assertLess(report["revisit_mean_actions"], report["first_mean_actions"])


if __name__ == "__main__":
    unittest.main()
