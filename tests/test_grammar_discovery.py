"""Contracts for latent-family and grounded-grammar discovery."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jepa_asteroids.grammar_discovery import (
    GrammarConcepts,
    GrammarDiscoveryWorld,
    GrammarProgramLearner,
    GrammarStrategy,
    SymbolAlphabet,
    SymbolEchoTerminal,
    _evaluate,
    _train,
    discover_structure,
    glyph_bank,
)


class GrammarDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = glyph_bank(6118)
        cls.factory = staticmethod(lambda: GrammarDiscoveryWorld(6101, bank=cls.bank))
        cls.alphabet = SymbolAlphabet.discover(
            lambda: SymbolEchoTerminal(cls.bank, 6102), samples=2, seed_base=91_000_000)
        cls.structure, cls.evidence = discover_structure(
            cls.factory, cls.alphabet, observations=64, seed_base=91_100_000)
        cls.concepts = GrammarConcepts.discover(cls.factory, observations=32,
                                                seed_base=91_200_000)

    def test_variable_clauses_temporal_order_and_terminal_reward(self):
        train = GrammarDiscoveryWorld(6101, bank=self.bank, held_out=False)
        held = GrammarDiscoveryWorld(6101, bank=self.bank, held_out=True)
        self.assertTrue(set(train.command_pool).isdisjoint(held.command_pool))
        observation = train.reset(91_300_000)
        self.assertEqual(observation.shape, (3, 96, 150))
        for index, action in enumerate(train.target_actions):
            _, reward, done, _ = train.step(action)
            if index + 1 < len(train.target_actions):
                self.assertEqual(reward, 0.0); self.assertFalse(done)
        self.assertEqual(reward, 1.0); self.assertTrue(done)
        self.assertIn(len(train.target_actions), (2, 3, 4))

    def test_word_family_count_and_optional_family_are_discovered(self):
        self.assertEqual(self.evidence['families_discovered'], 5)
        self.assertEqual(self.evidence['mandatory_family_sizes'], [2, 3, 3])
        self.assertEqual(self.evidence['optional_family_sizes'], [2])
        self.assertEqual(self.evidence['human_labels'], 0)

    def test_terminal_evidence_resolves_program_and_composes(self):
        learner = GrammarProgramLearner(self.alphabet, self.concepts, self.structure)
        result = _train(learner, self.factory, 80, 91_400_000)
        self.assertEqual(result['initial_programs'], 576)
        self.assertEqual(result['remaining_programs'], 1)
        held = lambda: GrammarDiscoveryWorld(6101, bank=self.bank, held_out=True)
        evaluation = _evaluate(learner, held, 32, 91_500_000)
        self.assertEqual(evaluation['read_execute_success_rate'], 1.0)
        self.assertEqual(evaluation['semantic_description_accuracy'], 1.0)

    def test_discovered_grammar_memory_is_persistent_and_executable(self):
        learner = GrammarProgramLearner(self.alphabet, self.concepts, self.structure)
        _train(learner, self.factory, 80, 91_600_000)
        frame = self.factory().reset(91_700_000)
        with tempfile.TemporaryDirectory() as directory:
            learner.save(Path(directory) / 'language_00.json')
            strategy = GrammarStrategy.load(Path(directory)); strategy.reset(frame)
            self.assertIn(strategy.act(frame), range(8))


if __name__ == '__main__':
    unittest.main()
