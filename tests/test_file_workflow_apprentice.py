"""Practical and safety contracts for the learned file workflow."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from jepa_asteroids.file_workflow_apprentice import create_demo_inbox, run_workflow


class FileWorkflowApprenticeTests(unittest.TestCase):
    def test_creates_verified_archive_without_changing_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = create_demo_inbox(root / "inbox")
            before = {path.name: path.read_bytes() for path in source.iterdir()}
            report = run_workflow(source, root / "output")
            revisit = run_workflow(source, root / "output")
            after = {path.name: path.read_bytes() for path in source.iterdir()}
            self.assertEqual(before, after)
            self.assertTrue(report["verification"]["verified"])
            self.assertEqual(report["verification"]["files"], len(before))
            self.assertLessEqual(report["actions_to_complete"], 25)
            self.assertEqual(revisit["actions_to_complete"], 5)
            self.assertTrue(revisit["resumed_prior_experience"])
            self.assertFalse(report["safety"]["source_files_modified"])


if __name__ == "__main__":
    unittest.main()
