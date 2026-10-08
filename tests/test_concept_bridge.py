from pathlib import Path

import numpy as np
import torch

from jepa_asteroids.concept_bridge import ConceptBridge, CONCEPTS, _explore_until_success, render_glyph, render_word


def test_rendered_representations_are_real_rgb_and_distinct():
    apple = np.asarray(render_glyph(0, 1)); bicycle = np.asarray(render_glyph(1, 1))
    word = np.asarray(render_word(0, 1))
    assert apple.shape == (64, 64, 3); assert word.shape == (64, 96, 3)
    assert apple.dtype == np.uint8; assert not np.array_equal(apple, bicycle)


def test_reward_exploration_discovers_only_successful_actions():
    labels = torch.arange(len(CONCEPTS))
    discovered, attempts = _explore_until_success(labels, np.random.default_rng(4))
    assert torch.equal(discovered.cpu(), labels)
    assert all(1 <= value <= len(CONCEPTS) for value in attempts)


def test_three_modalities_share_embedding_dimension():
    model = ConceptBridge()
    assert model.encode("photo", torch.zeros(2, 3, 64, 64)).shape == (2, 64)
    assert model.encode("glyph", torch.zeros(2, 3, 64, 64)).shape == (2, 64)
    assert model.encode("word", torch.zeros(2, 3, 64, 96)).shape == (2, 64)
