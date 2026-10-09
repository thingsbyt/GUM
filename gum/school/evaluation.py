"""Precommitted GUM School evaluation record and promotion-gate logic."""
from __future__ import annotations

import math
from pathlib import PurePosixPath
import re
from typing import Any

from .symmetry_uncertainty import (
    SymmetryUncertaintyError,
    evaluate_symmetry_uncertainty_gate,
    validate_symmetry_uncertainty_summary,
)


class SchoolEvaluationError(ValueError):
    pass


SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _strict_object(value: Any, label: str, fields: set[str]) -> dict:
    if not isinstance(value, dict):
        raise SchoolEvaluationError(f"{label} must be an object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing:
        raise SchoolEvaluationError(f"{label} is missing fields: {missing}")
    if unknown:
        raise SchoolEvaluationError(f"{label} has unknown fields: {unknown}")
    return value


def _number(value: Any, label: str, *, minimum: float = 0, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchoolEvaluationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        raise SchoolEvaluationError(f"{label} is outside its valid range")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SchoolEvaluationError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SchoolEvaluationError(f"{label} must be boolean")
    return value


def _relative_evidence_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchoolEvaluationError(f"{label} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise SchoolEvaluationError(f"{label} must stay inside the run evidence directory")
    return value


def _interval(value: Any, label: str, *, maximum: float | None = None) -> dict:
    row = _strict_object(value, label, {"lower", "upper", "method"})
    lower = _number(row["lower"], f"{label}.lower", maximum=maximum)
    upper = _number(row["upper"], f"{label}.upper", maximum=maximum)
    if lower > upper:
        raise SchoolEvaluationError(f"{label}.lower must not exceed upper")
    if row["method"] not in {"wilson", "paired-bootstrap"}:
        raise SchoolEvaluationError(f"{label}.method is unsupported")
    return row


def validate_training_summary(summary: Any) -> dict:
    row = _strict_object(summary, "training summary", {
        "format", "completed", "interactions", "wall_clock_seconds", "peak_memory_mb",
        "artifact_storage_mb", "evidence_path", "evidence_sha256",
    })
    if row["format"] != "gum-school-training-summary-v1":
        raise SchoolEvaluationError("training summary has an unsupported format")
    if _boolean(row["completed"], "training summary.completed") is not True:
        raise SchoolEvaluationError("training summary.completed must be true")
    _integer(row["interactions"], "training summary.interactions", minimum=1)
    for field in ("wall_clock_seconds", "peak_memory_mb", "artifact_storage_mb"):
        _number(row[field], f"training summary.{field}")
    _relative_evidence_path(row["evidence_path"], "training summary.evidence_path")
    if not isinstance(row["evidence_sha256"], str) or not SHA256_PATTERN.fullmatch(row["evidence_sha256"]):
        raise SchoolEvaluationError("training summary.evidence_sha256 must be a SHA-256 identifier")
    return row


def validate_evaluation_record(record: Any) -> dict:
    if not isinstance(record, dict):
        raise SchoolEvaluationError("evaluation must be an object")
    record_format = record.get("format")
    fields = {
        "format", "sealed", "retention", "evidence", "resources", "input_boundary",
        "replay", "transfer_results",
    }
    if record_format == "gum-school-evaluation-v2":
        fields.add("uncertainty_calibration")
    row = _strict_object(record, "evaluation", fields)
    if row["format"] not in {"gum-school-evaluation-v1", "gum-school-evaluation-v2"}:
        raise SchoolEvaluationError("evaluation has an unsupported format")

    sealed = _strict_object(row["sealed"], "evaluation.sealed", {
        "protocol_verified", "trials", "success_rate", "fresh_success_rate",
        "uncertainty_rate", "unnecessary_action_rate", "same_perception",
        "same_resource_limits", "confidence_level", "success_interval",
        "fresh_advantage_interval",
    })
    for field in ("protocol_verified", "same_perception", "same_resource_limits"):
        _boolean(sealed[field], f"evaluation.sealed.{field}")
    _integer(sealed["trials"], "evaluation.sealed.trials", minimum=1)
    for field in ("success_rate", "fresh_success_rate", "uncertainty_rate", "unnecessary_action_rate"):
        _number(sealed[field], f"evaluation.sealed.{field}", maximum=1)
    _number(sealed["confidence_level"], "evaluation.sealed.confidence_level", minimum=0.8, maximum=0.999)
    success_interval = _interval(
        sealed["success_interval"], "evaluation.sealed.success_interval", maximum=1
    )
    if not success_interval["lower"] <= sealed["success_rate"] <= success_interval["upper"]:
        raise SchoolEvaluationError("evaluation.sealed.success_rate is outside success_interval")
    if success_interval["method"] != "wilson":
        raise SchoolEvaluationError("evaluation.sealed.success_interval must use wilson")
    advantage_interval = _interval(
        sealed["fresh_advantage_interval"], "evaluation.sealed.fresh_advantage_interval"
    )
    if advantage_interval["method"] != "paired-bootstrap":
        raise SchoolEvaluationError(
            "evaluation.sealed.fresh_advantage_interval must use paired-bootstrap"
        )

    if row["format"] == "gum-school-evaluation-v2":
        try:
            calibration = validate_symmetry_uncertainty_summary(
                row["uncertainty_calibration"]
            )
        except SymmetryUncertaintyError as error:
            raise SchoolEvaluationError(
                f"evaluation.uncertainty_calibration is invalid: {error}"
            ) from error
        if not math.isclose(
            float(sealed["uncertainty_rate"]),
            float(calibration["post_causal_evidence_uncertainty_rate"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SchoolEvaluationError(
                "evaluation.sealed.uncertainty_rate must equal the "
                "post-causal-evidence uncertainty rate"
            )
        expected_successes = int(round(float(sealed["success_rate"]) * sealed["trials"]))
        if calibration["successful_episodes"] != expected_successes:
            raise SchoolEvaluationError(
                "evaluation.uncertainty_calibration success count differs from sealed result"
            )

    if not isinstance(row["retention"], list):
        raise SchoolEvaluationError("evaluation.retention must be an array")
    seen_skills: set[str] = set()
    for index, item in enumerate(row["retention"]):
        item = _strict_object(item, f"evaluation.retention[{index}]", {
            "skill", "baseline_success_rate", "current_success_rate", "evidence_verified",
        })
        if not isinstance(item["skill"], str) or not item["skill"]:
            raise SchoolEvaluationError(f"evaluation.retention[{index}].skill must be non-empty")
        if item["skill"] in seen_skills:
            raise SchoolEvaluationError(f"evaluation.retention contains duplicate skill {item['skill']!r}")
        seen_skills.add(item["skill"])
        _number(item["baseline_success_rate"], f"evaluation.retention[{index}].baseline_success_rate", maximum=1)
        _number(item["current_success_rate"], f"evaluation.retention[{index}].current_success_rate", maximum=1)
        _boolean(item["evidence_verified"], f"evaluation.retention[{index}].evidence_verified")

    evidence = _strict_object(row["evidence"], "evaluation.evidence", {
        "source_hashes_verified", "protocol_hashes_verified", "trajectory_verified",
        "ledger_verified",
    })
    for field, value in evidence.items():
        _boolean(value, f"evaluation.evidence.{field}")

    resources = _strict_object(row["resources"], "evaluation.resources", {
        "wall_clock_seconds", "peak_memory_mb", "artifact_storage_mb", "interactions",
    })
    for field in ("wall_clock_seconds", "peak_memory_mb", "artifact_storage_mb"):
        _number(resources[field], f"evaluation.resources.{field}")
    _integer(resources["interactions"], "evaluation.resources.interactions", minimum=0)

    boundary = _strict_object(row["input_boundary"], "evaluation.input_boundary", {
        "verified", "violations",
    })
    _boolean(boundary["verified"], "evaluation.input_boundary.verified")
    if not isinstance(boundary["violations"], list) or not all(
        isinstance(value, str) and value for value in boundary["violations"]
    ):
        raise SchoolEvaluationError("evaluation.input_boundary.violations must be an array of strings")

    replay = _strict_object(row["replay"], "evaluation.replay", {
        "verified", "deterministic", "artifact_path", "artifact_sha256",
    })
    _boolean(replay["verified"], "evaluation.replay.verified")
    _boolean(replay["deterministic"], "evaluation.replay.deterministic")
    _relative_evidence_path(replay["artifact_path"], "evaluation.replay.artifact_path")
    if not isinstance(replay["artifact_sha256"], str) or not SHA256_PATTERN.fullmatch(replay["artifact_sha256"]):
        raise SchoolEvaluationError("evaluation.replay.artifact_sha256 must be a SHA-256 identifier")

    if not isinstance(row["transfer_results"], list):
        raise SchoolEvaluationError("evaluation.transfer_results must be an array")
    seen_targets: set[str] = set()
    for index, item in enumerate(row["transfer_results"]):
        item = _strict_object(item, f"evaluation.transfer_results[{index}]", {
            "target_school", "candidate_success_rate", "fresh_success_rate",
            "candidate_mean_interactions", "fresh_mean_interactions", "uncertainty_rate",
            "unnecessary_action_rate", "trials", "evidence_path", "evidence_sha256",
        })
        target = item["target_school"]
        if not isinstance(target, str) or not target:
            raise SchoolEvaluationError(f"evaluation.transfer_results[{index}].target_school must be non-empty")
        if target in seen_targets:
            raise SchoolEvaluationError(f"duplicate transfer target {target!r}")
        seen_targets.add(target)
        for field in ("candidate_success_rate", "fresh_success_rate", "uncertainty_rate", "unnecessary_action_rate"):
            _number(item[field], f"evaluation.transfer_results[{index}].{field}", maximum=1)
        for field in ("candidate_mean_interactions", "fresh_mean_interactions"):
            _number(item[field], f"evaluation.transfer_results[{index}].{field}")
        _integer(item["trials"], f"evaluation.transfer_results[{index}].trials", minimum=1)
        _relative_evidence_path(item["evidence_path"], f"evaluation.transfer_results[{index}].evidence_path")
        if not isinstance(item["evidence_sha256"], str) or not SHA256_PATTERN.fullmatch(item["evidence_sha256"]):
            raise SchoolEvaluationError(
                f"evaluation.transfer_results[{index}].evidence_sha256 must be a SHA-256 identifier"
            )
    return row


def evaluate_promotion_gates(
    lesson: dict,
    evaluation: dict,
    *,
    required_protected_skills: set[str] | None = None,
) -> dict:
    """Evaluate all seven precommitted gates; never accept a caller's pass flag."""
    validate_evaluation_record(evaluation)
    sealed = evaluation["sealed"]
    promotion = lesson["promotion"]
    required_trials = int(lesson["evaluation"]["trials"])
    sealed_reasons: list[str] = []
    if not sealed["protocol_verified"]:
        sealed_reasons.append("sealed protocol was not verified")
    if sealed["confidence_level"] != promotion["confidence_level"]:
        sealed_reasons.append("confidence level differs from the precommitted level")
    if sealed["trials"] < required_trials:
        sealed_reasons.append(f"sealed trials {sealed['trials']} are below required {required_trials}")
    if sealed["success_rate"] < promotion["minimum_success"]:
        sealed_reasons.append("sealed success is below minimum_success")
    if sealed["success_interval"]["lower"] < promotion["minimum_success_lower_bound"]:
        sealed_reasons.append("sealed success confidence lower bound is below its minimum")
    if sealed["uncertainty_rate"] > promotion["maximum_uncertainty_rate"]:
        sealed_reasons.append("uncertainty rate exceeds its maximum")
    if evaluation["format"] == "gum-school-evaluation-v2":
        symmetry_gate = evaluate_symmetry_uncertainty_gate(
            evaluation["uncertainty_calibration"],
            maximum_post_causal_uncertainty_rate=promotion[
                "maximum_uncertainty_rate"
            ],
        )
        sealed_reasons.extend(symmetry_gate["reasons"])
    if sealed["unnecessary_action_rate"] > promotion["maximum_unnecessary_action_rate"]:
        sealed_reasons.append("unnecessary-action rate exceeds its maximum")

    retention_by_skill = {row["skill"]: row for row in evaluation["retention"]}
    retention_reasons: list[str] = []
    regressions: dict[str, float | None] = {}
    protected_skills = set(lesson["retention"]["protected_skills"])
    if required_protected_skills is not None:
        protected_skills.update(required_protected_skills)
    for skill in sorted(protected_skills):
        row = retention_by_skill.get(skill)
        if row is None:
            retention_reasons.append(f"missing protected skill {skill}")
            regressions[skill] = None
            continue
        regression = float(row["baseline_success_rate"]) - float(row["current_success_rate"])
        regressions[skill] = regression
        if not row["evidence_verified"]:
            retention_reasons.append(f"retention evidence not verified for {skill}")
        if regression > lesson["retention"]["maximum_regression"]:
            retention_reasons.append(f"retention regression exceeds tolerance for {skill}")

    evidence_reasons = [
        f"{field} is false" for field, passed in evaluation["evidence"].items() if not passed
    ]

    limits = lesson["resource_limits"]
    resource_map = {
        "wall_clock_seconds": "wall_clock_seconds",
        "peak_memory_mb": "peak_memory_mb",
        "artifact_storage_mb": "artifact_storage_mb",
        "interactions": "max_interactions",
    }
    resource_reasons = []
    for measured_field, limit_field in resource_map.items():
        if evaluation["resources"][measured_field] > limits[limit_field]:
            resource_reasons.append(f"{measured_field} exceeds {limit_field}")

    denominator = max(float(sealed["fresh_success_rate"]), 1.0 / float(sealed["trials"]))
    fresh_advantage = float(sealed["success_rate"]) / denominator
    control_reasons: list[str] = []
    if not sealed["same_perception"]:
        control_reasons.append("matched fresh learner did not use the same perception")
    if not sealed["same_resource_limits"]:
        control_reasons.append("matched fresh learner did not use the same resource limits")
    if fresh_advantage < promotion["minimum_fresh_advantage"]:
        control_reasons.append("fresh advantage is below minimum_fresh_advantage")
    if not (sealed["fresh_advantage_interval"]["lower"] <= fresh_advantage
            <= sealed["fresh_advantage_interval"]["upper"]):
        control_reasons.append("computed fresh advantage is outside its reported confidence interval")
    if (sealed["fresh_advantage_interval"]["lower"]
            < promotion["minimum_fresh_advantage_lower_bound"]):
        control_reasons.append("fresh-advantage confidence lower bound is below its minimum")

    boundary = evaluation["input_boundary"]
    boundary_reasons = list(boundary["violations"])
    if not boundary["verified"]:
        boundary_reasons.append("input boundary was not verified")

    replay = evaluation["replay"]
    replay_reasons = []
    if not replay["verified"]:
        replay_reasons.append("replay artifact was not verified")
    if not replay["deterministic"]:
        replay_reasons.append("replay was not deterministic within tolerance")

    reason_map = {
        "sealed-performance": sealed_reasons,
        "retention": retention_reasons,
        "evidence-integrity": evidence_reasons,
        "resource-limits": resource_reasons,
        "control-advantage": control_reasons,
        "input-boundary": boundary_reasons,
        "replay": replay_reasons,
    }
    gates = {
        gate: {"passed": not reason_map[gate], "reasons": reason_map[gate]}
        for gate in lesson["promotion"]["required_gates"]
    }
    gates["retention"]["regressions"] = regressions
    gates["control-advantage"]["fresh_advantage"] = fresh_advantage
    return {
        "format": "gum-school-gate-result-v1",
        "all_passed": all(row["passed"] for row in gates.values()),
        "gates": gates,
    }
