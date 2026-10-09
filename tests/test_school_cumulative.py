"""Contracts for the cumulative recurrent GUM School candidate."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from gum.protocol import PublicWorldSpec
from gum.school.cumulative import (
    CAUSAL_FILENAME,
    CumulativeLearnerError,
    CumulativeSchoolLearner,
)
from gum.school.learner import CrossSeedSchoolLearner
from gum.school.recurrent_meta import RecurrentCausalLearner
from gum.school.worlds import CAUSAL_WORKSHOP_ADAPTER, OBJECT_LABORATORY_ADAPTER


pytestmark = pytest.mark.neural


def _bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    base = source / "base.json"
    causal = source / "causal.pt"
    CrossSeedSchoolLearner(11).save(base)
    RecurrentCausalLearner(12).save(causal)
    target = tmp_path / "bundle"
    CumulativeSchoolLearner.create_bundle(
        target, base_path=base, causal_path=causal
    )
    return target


def _spec(adapter: str, shape: tuple[int, int, int], actions: int) -> PublicWorldSpec:
    return PublicWorldSpec(
        "test-world", "test-family", adapter, 1, "rgb", shape,
        "discrete-anonymous", actions, 200, (-1.0, 1.0),
    )


def test_bundle_routes_causal_and_object_episodes(tmp_path: Path):
    learner = CumulativeSchoolLearner.load_bundle(_bundle(tmp_path))
    causal_pixels = np.zeros((72, 72, 3), dtype=np.uint8)
    learner.begin(
        _spec(CAUSAL_WORKSHOP_ADAPTER, causal_pixels.shape, 8),
        causal_pixels,
        training=False,
    )
    assert learner._active is learner.causal
    assert 0 <= learner.act(causal_pixels, training=False) < 8

    object_pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    learner.begin(
        _spec(OBJECT_LABORATORY_ADAPTER, object_pixels.shape, 6),
        object_pixels,
        training=False,
    )
    assert learner._active is learner.base
    assert 0 <= learner.act(object_pixels, training=False) < 6


def test_bundle_rejects_component_tampering(tmp_path: Path):
    bundle = _bundle(tmp_path)
    component = bundle / CAUSAL_FILENAME
    component.write_bytes(component.read_bytes() + b"tampered")
    with pytest.raises(CumulativeLearnerError, match="hash differs"):
        CumulativeSchoolLearner.load_bundle(bundle)


def test_bundle_manifest_names_only_reviewed_components(tmp_path: Path):
    bundle = _bundle(tmp_path)
    manifest = json.loads(
        (bundle / "CUMULATIVE_LEARNER.json").read_text(encoding="utf-8")
    )
    assert manifest["training_complete_before_bundle"] is True
    assert set(manifest["components"]) == {
        "SCHOOL_LEARNER.json", "CAUSAL_META_POLICY.pt"
    }
