import unittest

import numpy as np

from jepa_asteroids.concept_memory import concept_state_key
from jepa_asteroids.event_option_benchmark import ShiftingEventChainWorld
from jepa_asteroids.universal_explorer import _objects_exact


class EventOptionBenchmarkTests(unittest.TestCase):
    def test_only_complete_chain_is_rewarded(self):
        world = ShiftingEventChainWorld(41); world.reset(500)
        for action in world.chain[:-1]:
            _, reward, done, _ = world.step(action)
            self.assertEqual(reward, 0.0); self.assertFalse(done)
        _, reward, done, info = world.step(world.chain[-1])
        self.assertEqual(reward, 1.0); self.assertTrue(done); self.assertTrue(info["success"])

    def test_wrong_action_resets_visible_prefix(self):
        world = ShiftingEventChainWorld(43); root = world.reset(600)
        world.step(world.chain[0])
        wrong = next(action for action in world.actions if action != world.chain[1])
        observation, reward, done, info = world.step(wrong)
        self.assertEqual(reward, 0.0); self.assertFalse(done)
        self.assertEqual(info["terminal_stage"], 0)
        self.assertTrue(np.array_equal(observation, root))

    def test_concept_state_transfers_across_layout_translation(self):
        world = ShiftingEventChainWorld(47)
        first = world.reset(700); first, _, _, _ = world.step(world.chain[0])
        second = world.reset(701); second, _, _, _ = world.step(world.chain[0])
        key_a = concept_state_key(_objects_exact(first[0]))
        key_b = concept_state_key(_objects_exact(second[0]))
        self.assertEqual(key_a, key_b)


if __name__ == "__main__":
    unittest.main()
