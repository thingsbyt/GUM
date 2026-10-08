"""Contracts for causal-role abstraction and structural transfer."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from jepa_asteroids.role_schema import CausalRoleSchema, structural_features
from jepa_asteroids.structural_transfer import run


class RoleSchemaTests(unittest.TestCase):
    def test_role_features_discard_color(self):
        first = "object:c1:a0:w1:h1"; remapped = "object:c14:a0:w1:h1"
        self.assertTrue((structural_features(first) == structural_features(remapped)).all())

    def test_learned_chain_selects_remapped_roles_in_order(self):
        schema = CausalRoleSchema(); schema.note_contact("object:c1:a0:w1:h1")
        schema.note_contact("object:c9:a3:w5:h2"); self.assertTrue(schema.learn_from_progress())
        self.assertGreater(schema.priority("object:c14:a0:w1:h1"),
                           schema.priority("object:c12:a4:w7:h7"))
        schema.note_contact("object:c14:a0:w1:h1")
        self.assertGreater(schema.priority("object:c8:a3:w5:h2"),
                           schema.priority("object:c12:a4:w7:h7"))

    def test_a_b_a_transfer_beats_scratch_and_retains_a(self):
        with tempfile.TemporaryDirectory() as folder:
            report = run(Path(folder) / "proof.json", variants=32)
        self.assertEqual(report["b_transfer"]["successes"], 32)
        self.assertGreater(report["positive_transfer"]["relative_step_reduction"], .5)
        self.assertTrue(report["retention"]["successful"])
        self.assertLessEqual(report["retention"]["steps_after"], report["retention"]["steps_before"])


if __name__ == "__main__": unittest.main()
