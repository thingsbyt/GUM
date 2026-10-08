"""Contracts for autonomous relational ontology growth."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jepa_asteroids.ontology_growth import (
    OntologyExpander,
    OntologyStrategy,
    RelationalConcepts,
    RelationalLanguageWorld,
    SymbolAlphabet,
    SymbolEchoTerminal,
    _evaluate,
    _train,
    discover_structure,
    glyph_bank,
)


class OntologyGrowthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = glyph_bank(10132)
        cls.factory = staticmethod(
            lambda: RelationalLanguageWorld(10101, bank=cls.bank, held_out=False))
        cls.alphabet = SymbolAlphabet.discover(
            lambda: SymbolEchoTerminal(cls.bank, 10102), samples=2,
            seed_base=112_000_000)
        cls.structure, cls.evidence = discover_structure(
            cls.factory, cls.alphabet, observations=64,
            seed_base=112_100_000)
        cls.concepts = RelationalConcepts.discover(
            cls.factory, observations=48, seed_base=112_200_000)

    def test_duplicate_descriptors_require_relation_and_reward_is_terminal(self):
        env = self.factory(); observation = env.reset(112_300_000)
        self.assertEqual(observation.shape, (3, 132, 180))
        target = env.commands[0][:2]
        self.assertEqual(sum(value == target for value in env.objects), 2)
        _, reward, done, _ = env.step(env.target_actions[0])
        self.assertEqual(reward, 0.0); self.assertFalse(done)
        _, reward, done, info = env.step(env.target_actions[1])
        self.assertEqual(reward, 1.0); self.assertTrue(done); self.assertTrue(info['success'])

    def test_latent_word_families_are_discovered_without_labels(self):
        self.assertEqual(self.evidence['families_discovered'], 4)
        self.assertEqual(self.evidence['family_sizes'], [2, 3, 3, 4])
        self.assertEqual(self.evidence['human_labels'], 0)

    def test_reward_promotes_four_relations_and_generalizes(self):
        learner = OntologyExpander(self.alphabet, self.concepts, self.structure)
        result = _train(learner, self.factory, 50, 112_400_000)
        self.assertEqual(result['initial_programs'], 51840)
        self.assertEqual(result['remaining_programs'], 1)
        self.assertEqual(learner.promoted_operators(), [0, 1, 2, 3])
        held = lambda: RelationalLanguageWorld(10101, bank=self.bank, held_out=True)
        self.assertEqual(_evaluate(learner, held, 32, 112_500_000)['success_rate'], 1.0)

    def test_grown_ontology_is_persistent_and_executable(self):
        learner = OntologyExpander(self.alphabet, self.concepts, self.structure)
        _train(learner, self.factory, 50, 112_600_000)
        frame = self.factory().reset(112_700_000)
        with tempfile.TemporaryDirectory() as directory:
            learner.save(Path(directory) / 'relation_language_00.json')
            strategy = OntologyStrategy.load(Path(directory)); strategy.reset(frame)
            self.assertIn(strategy.act(frame), range(12))


if __name__ == '__main__':
    unittest.main()
