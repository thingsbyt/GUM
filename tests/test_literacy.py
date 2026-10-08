"""Contracts for self-discovered visual literacy and grounded language."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from jepa_asteroids.literacy import (
    ARTIFICIAL,
    GLYPHS,
    GroundedInstructionWorld,
    GroundedLanguageModel,
    LiteracyTerminal,
    VisualAlphabet,
    _copy_episode,
    _decode_instruction,
    _object_features,
)


class LiteracyTests(unittest.TestCase):
    def test_terminal_exposes_pixels_and_rewards_only_exact_submission(self):
        env = LiteracyTerminal(mapping_seed=811)
        observation = env.reset(7, target=[0, 1, 2])
        self.assertEqual(observation.shape, (1, 48, 96))
        self.assertEqual(observation.dtype, np.uint8)
        _, reward, done, _ = env.step(env.submit)
        self.assertTrue(done)
        self.assertEqual(reward, -1.0)

        observation = env.reset(8, target=[0, 1, 2])
        inverse = {glyph: action for action, glyph in enumerate(env.mapping)}
        for glyph in (0, 1, 2):
            observation, reward, done, _ = env.step(inverse[glyph])
            self.assertEqual(reward, 0.0)
            self.assertFalse(done)
        _, reward, done, info = env.step(env.submit)
        self.assertTrue(done)
        self.assertTrue(info["success"])
        self.assertEqual(reward, 1.0)

    def test_visual_alphabet_is_discovered_and_supports_memory_and_repair(self):
        factory = lambda: LiteracyTerminal(mapping_seed=811)
        alphabet = VisualAlphabet.discover(factory, samples=2, seed_base=310_000)
        self.assertTrue(_copy_episode(factory, alphabet, 311_000)["success"])
        delayed = lambda: LiteracyTerminal(mapping_seed=811, delayed=True)
        self.assertTrue(_copy_episode(delayed, alphabet, 312_000)["success"])
        fault = lambda: LiteracyTerminal(mapping_seed=811, fault=True)
        repaired = _copy_episode(fault, alphabet, 313_000)
        self.assertTrue(repaired["success"])
        self.assertEqual(repaired["corrections"], 1)

    def test_same_discovery_algorithm_transfers_to_new_keyboard_and_glyphs(self):
        first = VisualAlphabet.discover(lambda: LiteracyTerminal(811, bank=ARTIFICIAL), samples=2)
        second_factory = lambda: LiteracyTerminal(1777, bank=GLYPHS)
        second = VisualAlphabet.discover(second_factory, samples=2, seed_base=320_000)
        self.assertEqual(len(first.prototypes), len(second.prototypes))
        for seed in range(4):
            self.assertTrue(_copy_episode(second_factory, second, 321_000 + seed)["success"])

    def test_grounded_world_and_model_use_visual_compositional_inputs(self):
        mapping = LiteracyTerminal(811).mapping
        alphabet = VisualAlphabet.discover(lambda: LiteracyTerminal(811), samples=2, seed_base=330_000)
        train_world = GroundedInstructionWorld(mapping, held_out=False)
        held_world = GroundedInstructionWorld(mapping, held_out=True)
        self.assertTrue(set(train_world.allowed).isdisjoint(held_world.allowed))
        frame = held_world.reset(331_000)
        tokens = _decode_instruction(alphabet, frame)
        colors, shapes = _object_features(frame)
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tuple(colors.shape), (4, 3))
        self.assertEqual(tuple(shapes.shape), (4, 399))
        logits = GroundedLanguageModel()(torch.tensor([tokens]), colors[None], shapes[None])
        self.assertEqual(tuple(logits.shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
