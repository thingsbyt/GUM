"""Game integrity contracts for Immune Savior."""
from __future__ import annotations
import unittest

from jepa_asteroids.immune_savior import ImmuneSaviorGame


class ImmuneSaviorTests(unittest.TestCase):
    def _interact_at(self, game, cell):
        game.position = cell
        return game.step(game.interact_action)

    def test_public_observation_and_hidden_recipe(self):
        game = ImmuneSaviorGame(101); frame = game.reset()
        self.assertEqual(frame.shape, (3, 126, 112)); self.assertEqual(game.action_dim, 5)
        self.assertFalse(any(name.encode() in frame.tobytes() for name in game.element_names))

    def test_only_correct_two_element_pair_makes_treatment(self):
        game = ImmuneSaviorGame(202); game.reset()
        correct = sorted(game.correct_pair)
        for name in correct: self._interact_at(game, game.fixed[name])
        _, reward, done, info = self._interact_at(game, game.fixed["synthesizer"])
        self.assertTrue(game.treatment); self.assertEqual(reward, 0.0); self.assertFalse(done)
        other = [name for name in game.element_names if name not in game.correct_pair]
        game.reset()
        for name in other[:2]: self._interact_at(game, game.fixed[name])
        self._interact_at(game, game.fixed["synthesizer"])
        self.assertFalse(game.treatment); self.assertEqual(game.failed_mixtures, 1)

    def test_cure_requires_all_disease_and_preserves_healthy(self):
        game = ImmuneSaviorGame(303, spread_interval=10_000); game.reset()
        game.treatment = True
        for cell in list(game.disease): frame, reward, done, info = self._interact_at(game, cell)
        self.assertTrue(done); self.assertEqual(reward, 1.0); self.assertEqual(info["healthy_damaged"], 0)

    def test_treating_healthy_cell_is_terminal_failure(self):
        game = ImmuneSaviorGame(404); game.reset(); game.treatment = True
        _, reward, done, info = self._interact_at(game, next(iter(game.healthy)))
        self.assertTrue(done); self.assertEqual(reward, -1.0); self.assertEqual(info["healthy_damaged"], 1)


if __name__ == "__main__": unittest.main()
