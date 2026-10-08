import tempfile
import unittest
from pathlib import Path

from gum.concept_genesis import ConceptGenesisMind, GenesisWorld


class ConceptGenesisTests(unittest.TestCase):
    def _mind(self):
        mind = ConceptGenesisMind()
        worlds = [GenesisWorld(10_000 + index * 101, (0,), index) for index in range(8)]
        report = mind.discover(worlds)
        self.assertEqual(report["selected_event_concepts"], 5)
        return mind

    def test_discovers_five_concepts_without_a_requested_count(self):
        mind = self._mind()
        self.assertEqual(mind.status()["concepts_created"], 5)
        self.assertEqual(len(mind.status()["concept_evidence"]), 5)
        self.assertTrue(all(value == 8 for value in mind.status()["concept_evidence"].values()))

        for index in range(4):
            _, mapping, _ = mind.ground(GenesisWorld(20_000 + index, (0, 1, 2), index))
            self.assertEqual(len(set(mapping.values())), 5)

    def test_reuses_composes_and_persists_discovered_concepts(self):
        mind = self._mind()
        left, right = (0, 4, 1), (2, 3, 0)
        first = mind.acquire(GenesisWorld(30_001, left, 0))
        second = mind.acquire(GenesisWorld(30_002, right, 1))
        self.assertTrue(first["success"] and second["success"])

        transfer = mind.solve(GenesisWorld(30_003, left, 5))
        self.assertTrue(transfer["success"])
        composed = mind.compose(GenesisWorld(30_004, left + right, 7))
        self.assertTrue(composed["success"])
        self.assertEqual(len(set(composed["parents"])), 2)

        scratch = Path(__file__).parent / ".scratch"; scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / "genesis.json"; mind.save(path)
            restored = ConceptGenesisMind.load(path)
            retained = restored.revisit(GenesisWorld(30_001, left, 0))
        self.assertTrue(retained["success"])
        self.assertEqual(retained["interactions"], 3)


if __name__ == "__main__":
    unittest.main()
