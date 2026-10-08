import hashlib
import tempfile
import unittest
from pathlib import Path

from gum.concept_genesis import ConceptGenesisMind
from gum.data_rescue_world import JsonDataRescueWorld, create_messy_jsonl


class DataRescueWorldTests(unittest.TestCase):
    def test_real_file_is_repaired_without_changing_source(self):
        scratch = Path(__file__).parent / ".scratch"; scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory); source = create_messy_jsonl(root / "messy.jsonl", 801)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            world = JsonDataRescueWorld(source, root / "work", 901)
            inverse = {operation: action for action, operation in enumerate(world.control_to_operation)}
            # One of the genuinely valid workflows; verification, not this list, controls success.
            actions = [inverse[value] for value in (1, 0, 3, 4, 2)]
            result = None
            for action in actions: _, _, _, result = world.step(action)
            self.assertTrue(result["success"])
            self.assertTrue(world.verify())
            self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertNotEqual(source.read_bytes(), world.active.read_bytes())

    def test_same_genesis_algorithm_selects_file_event_concepts(self):
        scratch = Path(__file__).parent / ".scratch"; scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory); worlds = []
            for index in range(6):
                source = create_messy_jsonl(root / f"messy-{index}.jsonl", 1000 + index)
                worlds.append(JsonDataRescueWorld(source, root / f"work-{index}", 2000 + index))
            mind = ConceptGenesisMind(); report = mind.discover(worlds)
            self.assertEqual(report["selected_event_concepts"], 5)
            self.assertTrue(all(value == 6 for value in report["concept_support"].values()))


if __name__ == "__main__":
    unittest.main()
