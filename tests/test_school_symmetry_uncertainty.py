"""Tests for symmetry-aware uncertainty measurement."""
import json
from pathlib import Path

import pytest

from gum.school.symmetry_uncertainty import (
    SymmetryUncertaintyError,
    evaluate_symmetry_uncertainty_gate,
    summarize_symmetry_aware_uncertainty,
)


def _episode(confidences, rewards, *, success=True):
    return {
        "success": success,
        "trajectory": [
            {"confidence": confidence, "reward": reward}
            for confidence, reward in zip(confidences, rewards, strict=True)
        ],
    }


def test_positive_consequence_only_changes_following_decisions():
    summary = summarize_symmetry_aware_uncertainty([
        _episode([0.0, 0.02, 0.8, 0.9], [-0.01, 0.2, 0.2, 1.0])
    ])
    assert summary["pre_causal_evidence_decisions"] == 2
    assert summary["post_causal_evidence_decisions"] == 2
    assert summary["post_causal_evidence_uncertainty_rate"] == 0.0
    assert summary["successful_episodes_without_post_causal_decisions"] == 0


def test_gate_requires_honest_initial_uncertainty_and_later_resolution():
    summary = summarize_symmetry_aware_uncertainty([
        _episode([0.0, 0.1, 0.8], [0.2, 0.2, 1.0]),
        _episode([0.0, 0.2, 0.9], [0.2, 0.2, 1.0]),
    ])
    gate = evaluate_symmetry_uncertainty_gate(
        summary, maximum_post_causal_uncertainty_rate=0.2
    )
    assert gate["passed"] is True
    assert gate["reasons"] == []


def test_gate_rejects_confident_guessing_before_evidence():
    summary = summarize_symmetry_aware_uncertainty([
        _episode([0.8, 0.9, 0.95], [0.2, 0.2, 1.0])
    ])
    gate = evaluate_symmetry_uncertainty_gate(
        summary, maximum_post_causal_uncertainty_rate=0.2
    )
    assert gate["passed"] is False
    assert "initial uncertainty is too low for symmetric controls" in gate["reasons"]


def test_gate_rejects_failure_to_resolve_after_evidence():
    summary = summarize_symmetry_aware_uncertainty([
        _episode([0.0, 0.01, 0.02], [0.2, 0.2, 1.0])
    ])
    gate = evaluate_symmetry_uncertainty_gate(
        summary, maximum_post_causal_uncertainty_rate=0.2
    )
    assert gate["passed"] is False
    assert "post-causal-evidence uncertainty exceeds its maximum" in gate["reasons"]


def test_metric_rejects_nonfinite_confidence():
    with pytest.raises(SymmetryUncertaintyError, match="confidence"):
        summarize_symmetry_aware_uncertainty([
            _episode([float("nan")], [-0.01], success=False)
        ])


def test_saved_public_validation_precommits_passing_protocol():
    path = (
        Path(__file__).resolve().parents[1]
        / "evidence" / "gum-school" / "research" / "causal-recurrent-meta-v2"
        / "SYMMETRY_UNCERTAINTY_VALIDATION.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["development_only"] is True
    assert report["sealed_data_used"] is False
    assert report["policy_weights_changed"] is False
    assert report["trials"] == 512
    assert report["validation_pass"] is True
    assert report["symmetry_aware_uncertainty"]["initial_uncertainty_rate"] == 1.0
    assert (
        report["symmetry_aware_uncertainty"][
            "post_causal_evidence_uncertainty_rate"
        ]
        == 0.0
    )
