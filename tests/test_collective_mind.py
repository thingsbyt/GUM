"""Contracts for provenance-aware multi-agent learning."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from jepa_asteroids.collective_learning_benchmark import run_collective_audit
from jepa_asteroids.collective_mind import ExperiencePacket, SharedExperienceLibrary


class CollectiveMindTests(unittest.TestCase):
    def test_independent_consensus_quarantines_single_source_conflict(self):
        library = SharedExperienceLibrary()
        good_a = ExperiencePacket("a", "glyph-x", "move", 2, ("task-a",))
        good_b = ExperiencePacket("b", "glyph-x", "move", 2, ("task-b",))
        bad = ExperiencePacket("noisy", "glyph-x", "erase", 99, ("claim",))
        for packet in (good_a, good_b, bad): library.ingest(packet)
        decision = library.decision("glyph-x")
        self.assertEqual(decision["assertion"], "move")
        self.assertIn(bad.packet_id, decision["quarantined_packet_ids"])

    def test_library_round_trip_preserves_provenance(self):
        library = SharedExperienceLibrary(); packet = ExperiencePacket(
            "a", "glyph-x", "move", 3, ("task-a",), confidence=1.0)
        library.ingest(packet); library.validate([packet.packet_id], "b", True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "library.json"; library.save(path)
            restored = SharedExperienceLibrary.load(path)
        self.assertEqual(restored.decision("glyph-x")["assertion"], "move")
        self.assertEqual(restored.decision("glyph-x")["validated_by"], ["b"])

    def test_tampered_packet_hash_is_rejected(self):
        packet = ExperiencePacket("a", "glyph-x", "move", 3, ("task-a",))
        row = packet.to_dict(); row["assertion"] = "erase"
        tampered = ExperiencePacket.from_dict(row)
        with self.assertRaisesRegex(ValueError, "provenance hash mismatch"):
            SharedExperienceLibrary().ingest(tampered)

    def test_disjoint_agents_combine_for_unseen_full_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            report = run_collective_audit(Path(folder) / "audit.json", tasks=12, horizon=42)
        self.assertTrue(report["precommitted_pass"])
        self.assertGreaterEqual(report["aggregate"]["verified-collective"]["success_rate"], .9)
        self.assertLessEqual(report["aggregate"]["isolated"]["success_rate"], .2)
        self.assertGreaterEqual(report["exchange"]["status_after_target"]["quarantined_packets"], 1)


if __name__ == "__main__": unittest.main()
