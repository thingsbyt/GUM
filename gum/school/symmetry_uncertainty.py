"""Symmetry-aware uncertainty measurement for anonymous-control tasks.

Before an intervention, visually identical anonymous controls provide no basis
for preferring one slot.  Low confidence is therefore expected during search.
The useful calibration question is whether confidence rises after a control has
produced a positive scalar consequence.  This module computes both quantities
without consulting event labels or hidden world state.
"""
from __future__ import annotations

import math
from typing import Any


CONFIDENCE_THRESHOLD = 0.05
MINIMUM_INITIAL_UNCERTAINTY_RATE = 0.95


class SymmetryUncertaintyError(ValueError):
    pass


SUMMARY_FIELDS = {
    "format",
    "information_used",
    "forbidden_information_used",
    "confidence_threshold",
    "episodes",
    "initial_decisions",
    "initial_uncertain_decisions",
    "initial_uncertainty_rate",
    "pre_causal_evidence_decisions",
    "pre_causal_evidence_uncertain_decisions",
    "pre_causal_evidence_uncertainty_rate",
    "post_causal_evidence_decisions",
    "post_causal_evidence_uncertain_decisions",
    "post_causal_evidence_uncertainty_rate",
    "pre_causal_evidence_mean_confidence",
    "post_causal_evidence_mean_confidence",
    "confidence_gain_after_causal_evidence",
    "successful_episodes",
    "successful_episodes_with_post_causal_decisions",
    "successful_episodes_without_post_causal_decisions",
}


def _finite_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SymmetryUncertaintyError("trajectory confidence must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SymmetryUncertaintyError("trajectory confidence must be in [0, 1]")
    return result


def _finite_reward(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SymmetryUncertaintyError("trajectory reward must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SymmetryUncertaintyError("trajectory reward must be finite")
    return result


def summarize_symmetry_aware_uncertainty(
    episodes: list[dict[str, Any]],
    *,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Separate justified search uncertainty from post-consequence confidence.

    A decision is post-causal-evidence only when an earlier action in the same
    episode produced positive scalar reward.  The action that discovers the
    effect remains in the pre-evidence group because its consequence was not
    available when that action was chosen.
    """
    if not episodes:
        raise SymmetryUncertaintyError("at least one episode is required")
    if not 0.0 < confidence_threshold < 1.0:
        raise SymmetryUncertaintyError("confidence threshold must be in (0, 1)")

    initial: list[float] = []
    pre_evidence: list[float] = []
    post_evidence: list[float] = []
    successes = 0
    successful_with_post_evidence = 0
    for episode in episodes:
        if not isinstance(episode, dict):
            raise SymmetryUncertaintyError("episode must be an object")
        trajectory = episode.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            raise SymmetryUncertaintyError("episode trajectory must be non-empty")
        seen_positive_consequence = False
        episode_post_decisions = 0
        for index, row in enumerate(trajectory):
            if not isinstance(row, dict):
                raise SymmetryUncertaintyError("trajectory row must be an object")
            confidence = _finite_confidence(row.get("confidence"))
            reward = _finite_reward(row.get("reward"))
            if index == 0:
                initial.append(confidence)
            if seen_positive_consequence:
                post_evidence.append(confidence)
                episode_post_decisions += 1
            else:
                pre_evidence.append(confidence)
            if reward > 0.0:
                seen_positive_consequence = True
        if bool(episode.get("success")):
            successes += 1
            successful_with_post_evidence += int(episode_post_decisions > 0)

    def uncertain(values: list[float]) -> int:
        return sum(value < confidence_threshold for value in values)

    initial_uncertain = uncertain(initial)
    pre_uncertain = uncertain(pre_evidence)
    post_uncertain = uncertain(post_evidence)
    pre_mean = sum(pre_evidence) / len(pre_evidence)
    post_mean = sum(post_evidence) / len(post_evidence) if post_evidence else 0.0
    return {
        "format": "gum-school-symmetry-aware-uncertainty-v1",
        "information_used": ["decision-confidence", "prior-scalar-reward"],
        "forbidden_information_used": False,
        "confidence_threshold": confidence_threshold,
        "episodes": len(episodes),
        "initial_decisions": len(initial),
        "initial_uncertain_decisions": initial_uncertain,
        "initial_uncertainty_rate": initial_uncertain / len(initial),
        "pre_causal_evidence_decisions": len(pre_evidence),
        "pre_causal_evidence_uncertain_decisions": pre_uncertain,
        "pre_causal_evidence_uncertainty_rate": pre_uncertain / len(pre_evidence),
        "post_causal_evidence_decisions": len(post_evidence),
        "post_causal_evidence_uncertain_decisions": post_uncertain,
        "post_causal_evidence_uncertainty_rate": (
            post_uncertain / len(post_evidence) if post_evidence else 1.0
        ),
        "pre_causal_evidence_mean_confidence": pre_mean,
        "post_causal_evidence_mean_confidence": post_mean,
        "confidence_gain_after_causal_evidence": post_mean - pre_mean,
        "successful_episodes": successes,
        "successful_episodes_with_post_causal_decisions": successful_with_post_evidence,
        "successful_episodes_without_post_causal_decisions": (
            successes - successful_with_post_evidence
        ),
    }


def validate_symmetry_uncertainty_summary(summary: Any) -> dict[str, Any]:
    """Strictly validate a persisted symmetry-aware summary and its arithmetic."""
    if not isinstance(summary, dict) or set(summary) != SUMMARY_FIELDS:
        raise SymmetryUncertaintyError(
            "symmetry-aware uncertainty summary fields differ from the reviewed format"
        )
    if summary["format"] != "gum-school-symmetry-aware-uncertainty-v1":
        raise SymmetryUncertaintyError("unsupported symmetry-aware uncertainty format")
    if summary["information_used"] != ["decision-confidence", "prior-scalar-reward"]:
        raise SymmetryUncertaintyError("symmetry-aware metric used an unreviewed input boundary")
    if summary["forbidden_information_used"] is not False:
        raise SymmetryUncertaintyError("symmetry-aware metric used forbidden information")

    integer_fields = (
        "episodes",
        "initial_decisions",
        "initial_uncertain_decisions",
        "pre_causal_evidence_decisions",
        "pre_causal_evidence_uncertain_decisions",
        "post_causal_evidence_decisions",
        "post_causal_evidence_uncertain_decisions",
        "successful_episodes",
        "successful_episodes_with_post_causal_decisions",
        "successful_episodes_without_post_causal_decisions",
    )
    for field in integer_fields:
        value = summary[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SymmetryUncertaintyError(f"{field} must be a non-negative integer")
    numeric_fields = (
        "confidence_threshold",
        "initial_uncertainty_rate",
        "pre_causal_evidence_uncertainty_rate",
        "post_causal_evidence_uncertainty_rate",
        "pre_causal_evidence_mean_confidence",
        "post_causal_evidence_mean_confidence",
        "confidence_gain_after_causal_evidence",
    )
    for field in numeric_fields:
        value = summary[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SymmetryUncertaintyError(f"{field} must be numeric")
        if not math.isfinite(float(value)):
            raise SymmetryUncertaintyError(f"{field} must be finite")
    if not 0.0 < float(summary["confidence_threshold"]) < 1.0:
        raise SymmetryUncertaintyError("confidence_threshold must be in (0, 1)")
    for field in (
        "initial_uncertainty_rate",
        "pre_causal_evidence_uncertainty_rate",
        "post_causal_evidence_uncertainty_rate",
        "pre_causal_evidence_mean_confidence",
        "post_causal_evidence_mean_confidence",
    ):
        if not 0.0 <= float(summary[field]) <= 1.0:
            raise SymmetryUncertaintyError(f"{field} must be in [0, 1]")

    if summary["episodes"] < 1 or summary["initial_decisions"] != summary["episodes"]:
        raise SymmetryUncertaintyError("initial-decision coverage must equal episode count")
    count_pairs = (
        ("initial_uncertain_decisions", "initial_decisions"),
        ("pre_causal_evidence_uncertain_decisions", "pre_causal_evidence_decisions"),
        ("post_causal_evidence_uncertain_decisions", "post_causal_evidence_decisions"),
        ("successful_episodes", "episodes"),
        ("successful_episodes_with_post_causal_decisions", "successful_episodes"),
        ("successful_episodes_without_post_causal_decisions", "successful_episodes"),
    )
    for numerator, denominator in count_pairs:
        if summary[numerator] > summary[denominator]:
            raise SymmetryUncertaintyError(f"{numerator} exceeds {denominator}")
    if (
        summary["successful_episodes_with_post_causal_decisions"]
        + summary["successful_episodes_without_post_causal_decisions"]
        != summary["successful_episodes"]
    ):
        raise SymmetryUncertaintyError("successful-episode coverage counts are inconsistent")

    expected_rates = {
        "initial_uncertainty_rate": (
            summary["initial_uncertain_decisions"] / summary["initial_decisions"]
        ),
        "pre_causal_evidence_uncertainty_rate": (
            summary["pre_causal_evidence_uncertain_decisions"]
            / max(1, summary["pre_causal_evidence_decisions"])
        ),
        "post_causal_evidence_uncertainty_rate": (
            summary["post_causal_evidence_uncertain_decisions"]
            / max(1, summary["post_causal_evidence_decisions"])
            if summary["post_causal_evidence_decisions"]
            else 1.0
        ),
    }
    for field, expected in expected_rates.items():
        if not math.isclose(float(summary[field]), expected, rel_tol=0.0, abs_tol=1e-12):
            raise SymmetryUncertaintyError(f"{field} differs from its recorded counts")
    expected_gain = (
        float(summary["post_causal_evidence_mean_confidence"])
        - float(summary["pre_causal_evidence_mean_confidence"])
    )
    if not math.isclose(
        float(summary["confidence_gain_after_causal_evidence"]),
        expected_gain,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SymmetryUncertaintyError("confidence gain differs from recorded means")
    return summary


def evaluate_symmetry_uncertainty_gate(
    summary: dict[str, Any],
    *,
    maximum_post_causal_uncertainty_rate: float,
    minimum_initial_uncertainty_rate: float = MINIMUM_INITIAL_UNCERTAINTY_RATE,
) -> dict[str, Any]:
    """Evaluate the precommitted symmetry-aware calibration requirements."""
    reasons: list[str] = []
    if summary.get("forbidden_information_used") is not False:
        reasons.append("uncertainty metric used forbidden information")
    if summary.get("initial_decisions") != summary.get("episodes"):
        reasons.append("initial-decision coverage is incomplete")
    if float(summary.get("initial_uncertainty_rate", 0.0)) < minimum_initial_uncertainty_rate:
        reasons.append("initial uncertainty is too low for symmetric controls")
    if int(summary.get("post_causal_evidence_decisions", 0)) < 1:
        reasons.append("no decisions followed positive causal evidence")
    if (
        float(summary.get("post_causal_evidence_uncertainty_rate", 1.0))
        > maximum_post_causal_uncertainty_rate
    ):
        reasons.append("post-causal-evidence uncertainty exceeds its maximum")
    if float(summary.get("confidence_gain_after_causal_evidence", 0.0)) <= 0.0:
        reasons.append("confidence did not increase after causal evidence")
    if int(summary.get("successful_episodes_without_post_causal_decisions", 0)):
        reasons.append("successful episodes lack post-causal-evidence coverage")
    return {
        "format": "gum-school-symmetry-uncertainty-gate-v1",
        "passed": not reasons,
        "reasons": reasons,
        "thresholds": {
            "minimum_initial_uncertainty_rate": minimum_initial_uncertainty_rate,
            "maximum_post_causal_uncertainty_rate": maximum_post_causal_uncertainty_rate,
            "minimum_confidence_gain_after_causal_evidence": 0.0,
        },
    }
