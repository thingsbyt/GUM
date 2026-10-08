"""Independent verifier rejects a falsified trace."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jepa_asteroids.compositional_mind_benchmark import run_benchmark
from jepa_asteroids.verify_problem_solving_audit import verify


class ProblemSolvingVerifierTests(unittest.TestCase):
    def test_verifies_real_audit_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audit.json"; run_benchmark(path, tasks=3)
            self.assertTrue(verify(path)["verified"])
            value = json.loads(path.read_text())
            value["aggregate"]["semantic_composer"]["rows"][0]["success"] = False
            path.write_text(json.dumps(value))
            with self.assertRaises(AssertionError): verify(path)


if __name__ == "__main__":
    unittest.main()
