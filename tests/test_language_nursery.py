"""Contracts for reward-grounded compositional language learning."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jepa_asteroids.language_nursery import (
    LanguageNurseryWorld,
    NurseryStrategy,
    VersionSpaceLanguage,
    VisualConcepts,
    _evaluate,
    _instruction_tokens,
    _position_sets,
    _train_language,
)
from jepa_asteroids.literacy import ARTIFICIAL, LiteracyTerminal, VisualAlphabet


def _components():
    factory = lambda: LanguageNurseryWorld(4101, bank=ARTIFICIAL)
    alphabet = VisualAlphabet.discover(lambda: LiteracyTerminal(811, bank=ARTIFICIAL),
                                       samples=2, seed_base=71_000_000)
    concepts = VisualConcepts.discover(factory, observations=24, seed_base=71_100_000)
    return factory, alphabet, concepts


class LanguageNurseryTests(unittest.TestCase):
    def test_world_has_disjoint_compositions_and_only_terminal_reward(self):
        train = LanguageNurseryWorld(4101, held_out=False, steps=3)
        held = LanguageNurseryWorld(4101, held_out=True, steps=3)
        self.assertTrue(set(train.commands).isdisjoint(held.commands))
        self.assertEqual(len(set(train.commands) | set(held.commands)), 18)
        observation = train.reset(72_000_000)
        self.assertEqual(observation.shape, (3, 78, 112))
        _, reward, done, _ = train.step(0)
        self.assertEqual(reward, 0.0); self.assertFalse(done)
        _, reward, done, _ = train.step(0)
        self.assertEqual(reward, 0.0); self.assertFalse(done)
        _, reward, done, _ = train.step(0)
        self.assertIn(reward, (-1.0, 1.0)); self.assertTrue(done)

    def test_reward_evidence_resolves_dictionary_and_generalizes(self):
        factory, alphabet, concepts = _components()
        learner = VersionSpaceLanguage(alphabet, concepts)
        result = _train_language(learner, factory, 64, 72_100_000)
        self.assertEqual(result['initial_hypotheses'], 40320)
        self.assertEqual(result['remaining_hypotheses'], 1)
        test_factory = lambda: LanguageNurseryWorld(4101, bank=ARTIFICIAL,
                                                     held_out=True, steps=3)
        evaluation = _evaluate(learner, test_factory, 32, 72_200_000)
        self.assertEqual(evaluation['read_execute_success_rate'], 1.0)
        self.assertEqual(evaluation['event_description_write_accuracy'], 1.0)

    def test_transferred_role_schema_reduces_second_language_search(self):
        factory, alphabet, concepts = _components()
        learner = VersionSpaceLanguage(alphabet, concepts)
        positions = _position_sets(factory, alphabet, samples=32, seed_base=72_300_000)
        remaining = learner.restrict_schema(positions)
        self.assertEqual(remaining, 144)
        self.assertLess(remaining, 40320)

    def test_learned_language_is_persistent_and_executable(self):
        factory, alphabet, concepts = _components()
        learner = VersionSpaceLanguage(alphabet, concepts)
        _train_language(learner, factory, 64, 72_400_000)
        frame = factory().reset(72_500_000)
        grammar = learner.grammar(_instruction_tokens(alphabet, frame))
        with tempfile.TemporaryDirectory() as directory:
            learner.save(Path(directory) / 'language_one.json', grammar)
            strategy = NurseryStrategy.load(Path(directory)); strategy.reset(frame)
            self.assertIn(strategy.act(frame), range(8))


if __name__ == '__main__':
    unittest.main()
