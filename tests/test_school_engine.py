from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from gum.school import SchoolEngine, SchoolEngineError, SnapshotStore
from gum.school.evaluation import evaluate_promotion_gates
from gum.school.symmetry_uncertainty import summarize_symmetry_aware_uncertainty
from gum.school.validation import DEFAULT_CURRICULUM, load_json


def _sha(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _initial_state(root: Path) -> Path:
    state = root / "initial"
    state.mkdir()
    (state / "mind.json").write_text('{"generation": 0}', encoding="utf-8")
    return state


def _training_summary(engine: SchoolEngine, run_id: str, *, interactions: int = 100) -> dict:
    folder = engine._run_folder(run_id)
    candidate = folder / "candidate"
    value = json.loads((candidate / "mind.json").read_text(encoding="utf-8"))
    value["generation"] += 1
    (candidate / "mind.json").write_text(json.dumps(value), encoding="utf-8")
    artifact = folder / "evidence" / "training.json"
    artifact.write_text('{"bounded": true}', encoding="utf-8")
    return {
        "format": "gum-school-training-summary-v1",
        "completed": True,
        "interactions": interactions,
        "wall_clock_seconds": 2.0,
        "peak_memory_mb": 64.0,
        "artifact_storage_mb": 1.0,
        "evidence_path": "evidence/training.json",
        "evidence_sha256": _sha(artifact),
    }


def _evaluation(
    engine: SchoolEngine,
    run_id: str,
    *,
    success: float = 0.9,
    fresh: float = 0.5,
    retention: list[dict] | None = None,
    boundary_violations: list[str] | None = None,
    resources: dict | None = None,
    transfer_target: str | None = None,
) -> dict:
    folder = engine._run_folder(run_id)
    replay = folder / "evidence" / "replay.json"
    replay.write_text('{"verified": true}', encoding="utf-8")
    transfers = []
    if transfer_target is not None:
        transfer = folder / "evidence" / f"transfer-{transfer_target}.json"
        transfer.write_text('{"verified": true}', encoding="utf-8")
        transfers.append({
            "target_school": transfer_target,
            "candidate_success_rate": success,
            "fresh_success_rate": fresh,
            "candidate_mean_interactions": 20.0,
            "fresh_mean_interactions": 40.0,
            "uncertainty_rate": 0.1,
            "unnecessary_action_rate": 0.1,
            "trials": 32,
            "evidence_path": f"evidence/transfer-{transfer_target}.json",
            "evidence_sha256": _sha(transfer),
        })
    advantage = success / max(fresh, 1 / 32)
    return {
        "format": "gum-school-evaluation-v1",
        "sealed": {
            "protocol_verified": True,
            "trials": 32,
            "success_rate": success,
            "fresh_success_rate": fresh,
            "uncertainty_rate": 0.1,
            "unnecessary_action_rate": 0.1,
            "same_perception": True,
            "same_resource_limits": True,
            "confidence_level": 0.95,
            "success_interval": {
                "lower": max(0.0, success - 0.15),
                "upper": min(1.0, success + 0.08),
                "method": "wilson",
            },
            "fresh_advantage_interval": {
                "lower": advantage * 0.67,
                "upper": advantage * 1.34,
                "method": "paired-bootstrap",
            },
        },
        "retention": [] if retention is None else retention,
        "evidence": {
            "source_hashes_verified": True,
            "protocol_hashes_verified": True,
            "trajectory_verified": True,
            "ledger_verified": True,
        },
        "resources": resources or {
            "wall_clock_seconds": 3.0,
            "peak_memory_mb": 64.0,
            "artifact_storage_mb": 1.0,
            "interactions": 100,
        },
        "input_boundary": {
            "verified": not boundary_violations,
            "violations": [] if boundary_violations is None else boundary_violations,
        },
        "replay": {
            "verified": True,
            "deterministic": True,
            "artifact_path": "evidence/replay.json",
            "artifact_sha256": _sha(replay),
        },
        "transfer_results": transfers,
    }


def _engine(tmp_path: Path) -> SchoolEngine:
    engine = SchoolEngine(tmp_path / "school", DEFAULT_CURRICULUM)
    engine.initialize_promoted(_initial_state(tmp_path))
    return engine


def _transfer_cell(engine: SchoolEngine, source: str, target: str) -> dict:
    evidence_root = engine.workspace / "transfer-evidence"
    evidence_root.mkdir(exist_ok=True)
    artifact = evidence_root / f"{source}--{target}.json"
    artifact.write_text('{"isolated": true}', encoding="utf-8")
    return {
        "format": "gum-school-transfer-cell-v1",
        "source_school": source,
        "target_school": target,
        "origin_snapshot_id": engine._progress()["initial_snapshot_id"],
        "trained_schools": [source],
        "target_evaluation_training": False,
        "matched_fresh": True,
        "same_perception": True,
        "same_resource_limits": True,
        "trials": 32,
        "candidate_success_rate": 0.9,
        "fresh_success_rate": 0.5,
        "candidate_mean_interactions": 20.0,
        "fresh_mean_interactions": 40.0,
        "uncertainty_rate": 0.1,
        "unnecessary_action_rate": 0.1,
        "confidence_level": 0.95,
        "success_interval": {"lower": 0.75, "upper": 0.98, "method": "wilson"},
        "fresh_advantage_interval": {
            "lower": 1.2, "upper": 2.4, "method": "paired-bootstrap"
        },
        "branch_verification": {
            "source_only_training_verified": True,
            "target_frozen_verified": True,
            "matched_fresh_verified": True,
            "source_hashes_verified": True,
            "protocol_hashes_verified": True,
            "trajectory_verified": True,
        },
        "evidence_path": f"transfer-evidence/{source}--{target}.json",
        "evidence_sha256": _sha(artifact),
    }


def _prepare_evaluated_run(engine: SchoolEngine, **evaluation_options) -> str:
    started = engine.begin_lesson()
    run_id = started["run_id"]
    engine.mark_training_complete(run_id, _training_summary(engine, run_id))
    engine.record_evaluation(run_id, _evaluation(engine, run_id, **evaluation_options))
    return run_id


def test_snapshot_store_is_content_addressed_and_detects_tampering(tmp_path: Path):
    source = _initial_state(tmp_path)
    store = SnapshotStore(tmp_path / "snapshots")
    manifest = store.create(source)
    assert store.verify(manifest["snapshot_id"])["valid"]
    materialized = tmp_path / "candidate"
    store.materialize(manifest["snapshot_id"], materialized)
    (materialized / "mind.json").write_text('{"generation": 99}', encoding="utf-8")
    assert store.verify(manifest["snapshot_id"])["valid"]
    snapshot_file = store.snapshot_path(manifest["snapshot_id"]) / "state" / "mind.json"
    snapshot_file.write_text("tampered", encoding="utf-8")
    assert not store.verify(manifest["snapshot_id"])["valid"]


def test_engine_clones_promoted_state_and_enforces_authored_order(tmp_path: Path):
    engine = _engine(tmp_path)
    before = engine.status()
    assert before["next_lesson_id"] == "object-laboratory.occlusion.001"
    with pytest.raises(SchoolEngineError, match="lesson order is fixed"):
        engine.begin_lesson("causal-workshop.controls.001")
    started = engine.begin_lesson()
    assert json.loads((started["candidate_directory"] / "mind.json").read_text())["generation"] == 0
    with pytest.raises(SchoolEngineError, match="already active"):
        engine.begin_lesson()
    assert engine.snapshots.promoted()["snapshot_id"] == before["promoted_snapshot_id"]


def test_engine_can_start_verified_track_from_inherited_curriculum_prefix(tmp_path: Path):
    workspace = tmp_path / "inherited-school"
    engine = SchoolEngine(workspace, DEFAULT_CURRICULUM)
    inherited = [
        "object-laboratory.occlusion.001",
        "object-laboratory.functional-category.002",
    ]
    engine.initialize_promoted(
        _initial_state(tmp_path),
        inherited_lessons=inherited,
        inheritance_evidence={"source": "verified-official-school", "verified": True},
    )
    assert engine.status()["inherited_lessons"] == inherited
    assert engine.status()["promoted_lessons"] == inherited
    assert engine.next_lesson()["lesson_id"] == "causal-workshop.controls.001"
    reopened = SchoolEngine(workspace, DEFAULT_CURRICULUM)
    assert reopened.status()["inherited_lessons"] == inherited


def test_engine_rejects_nonprefix_inheritance(tmp_path: Path):
    engine = SchoolEngine(tmp_path / "invalid-inheritance", DEFAULT_CURRICULUM)
    with pytest.raises(SchoolEngineError, match="exact curriculum prefix"):
        engine.initialize_promoted(
            _initial_state(tmp_path),
            inherited_lessons=["object-laboratory.functional-category.002"],
            inheritance_evidence={"verified": True},
        )


def test_v2_evaluation_uses_symmetry_aware_uncertainty_gate(tmp_path: Path):
    engine = _engine(tmp_path)
    started = engine.begin_lesson()
    run_id = started["run_id"]
    engine.mark_training_complete(run_id, _training_summary(engine, run_id))
    evaluation = _evaluation(engine, run_id, success=1.0, fresh=0.5)
    calibration = summarize_symmetry_aware_uncertainty([
        {
            "success": True,
            "trajectory": [
                {"confidence": 0.0, "reward": 0.2},
                {"confidence": 0.8, "reward": 1.0},
            ],
        }
        for _ in range(32)
    ])
    evaluation["format"] = "gum-school-evaluation-v2"
    evaluation["sealed"]["uncertainty_rate"] = 0.0
    evaluation["uncertainty_calibration"] = calibration
    engine.record_evaluation(run_id, evaluation)
    decision = engine.finalize(run_id)
    assert decision["outcome"] == "promote"
    assert decision["gates"]["gates"]["sealed-performance"]["passed"] is True


def test_all_seven_gates_promote_candidate_atomically(tmp_path: Path):
    engine = _engine(tmp_path)
    before = engine.snapshots.promoted()
    run_id = _prepare_evaluated_run(
        engine, transfer_target="object-laboratory"
    )
    decision = engine.finalize(run_id)
    after = engine.snapshots.promoted()
    assert decision["outcome"] == "promote"
    assert decision["gates"]["all_passed"]
    assert set(decision["gates"]["gates"]) == {
        "sealed-performance", "retention", "evidence-integrity", "resource-limits",
        "control-advantage", "input-boundary", "replay",
    }
    assert after["snapshot_id"] != before["snapshot_id"]
    assert after["generation"] == before["generation"] + 1
    assert engine.status()["next_lesson_id"] == "object-laboratory.functional-category.002"
    assert engine.write_report()["transfer_matrix"]["complete"] is False


def test_failed_gate_quarantines_without_changing_promoted_pointer(tmp_path: Path):
    engine = _engine(tmp_path)
    before = engine.snapshots.promoted()
    run_id = _prepare_evaluated_run(engine, success=0.6, fresh=0.5)
    decision = engine.finalize(run_id)
    assert decision["outcome"] == "quarantine"
    assert not decision["gates"]["gates"]["sealed-performance"]["passed"]
    assert engine.snapshots.promoted()["snapshot_id"] == before["snapshot_id"]
    assert (engine.quarantine / f"{run_id}.json").is_file()
    assert engine.status()["next_lesson_id"] == "object-laboratory.occlusion.001"


def test_resource_and_input_boundaries_are_computed_not_caller_graded(tmp_path: Path):
    curriculum = load_json(DEFAULT_CURRICULUM)
    lesson = curriculum["lessons"][0]
    engine = _engine(tmp_path)
    started = engine.begin_lesson()
    run_id = started["run_id"]
    engine.mark_training_complete(run_id, _training_summary(engine, run_id))
    evaluation = _evaluation(
        engine,
        run_id,
        boundary_violations=["audit-only-state observed"],
        resources={
            "wall_clock_seconds": 99999.0,
            "peak_memory_mb": 64.0,
            "artifact_storage_mb": 1.0,
            "interactions": 100,
        },
    )
    result = evaluate_promotion_gates(lesson, evaluation)
    assert not result["gates"]["resource-limits"]["passed"]
    assert not result["gates"]["input-boundary"]["passed"]
    assert not result["all_passed"]


def test_confidence_lower_bound_can_fail_when_point_estimate_passes(tmp_path: Path):
    curriculum = load_json(DEFAULT_CURRICULUM)
    lesson = curriculum["lessons"][0]
    engine = _engine(tmp_path)
    run_id = engine.begin_lesson()["run_id"]
    evaluation = _evaluation(engine, run_id, success=0.9, fresh=0.5)
    evaluation["sealed"]["success_interval"]["lower"] = 0.6
    result = evaluate_promotion_gates(lesson, evaluation)
    assert evaluation["sealed"]["success_rate"] >= lesson["promotion"]["minimum_success"]
    assert not result["gates"]["sealed-performance"]["passed"]
    assert any("confidence lower bound" in reason
               for reason in result["gates"]["sealed-performance"]["reasons"])


def test_finalize_is_idempotent_and_ledger_records_one_decision(tmp_path: Path):
    engine = _engine(tmp_path)
    run_id = _prepare_evaluated_run(engine)
    first = engine.finalize(run_id)
    second = engine.finalize(run_id)
    assert first == second
    rows = [json.loads(line) for line in engine.ledger.path.read_text().splitlines()]
    decisions = [row for row in rows if row["payload"].get("event_key") == f"promotion-decided:{run_id}"]
    assert len(decisions) == 1


def test_interruption_after_pointer_update_recovers_without_double_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    engine = _engine(tmp_path)
    run_id = _prepare_evaluated_run(engine)

    def interrupt(_run_id):
        raise RuntimeError("simulated power loss")

    monkeypatch.setattr(engine, "_after_promotion_pointer_updated", interrupt)
    with pytest.raises(RuntimeError, match="simulated power loss"):
        engine.finalize(run_id)
    assert engine.snapshots.promoted()["generation"] == 2

    reopened = SchoolEngine(engine.workspace, DEFAULT_CURRICULUM)
    assert reopened.snapshots.promoted()["generation"] == 2
    assert reopened._run_state(run_id)["status"] == "promoted"
    assert reopened.status()["active"]["status"] == "idle"
    rows = [json.loads(line) for line in reopened.ledger.path.read_text().splitlines()]
    assert sum(
        row["payload"].get("event_key") == f"promotion-decided:{run_id}" for row in rows
    ) == 1


def test_executor_exception_is_preserved_as_abort_without_promotion(tmp_path: Path):
    engine = _engine(tmp_path)
    before = engine.snapshots.promoted()["snapshot_id"]

    def trainer(_candidate, _lesson, _evidence):
        raise RuntimeError("trainer failed")

    with pytest.raises(RuntimeError, match="trainer failed"):
        engine.run_lesson(trainer, lambda *_: {})
    decision_files = list(engine.runs.glob("*/DECISION.json"))
    assert len(decision_files) == 1
    assert json.loads(decision_files[0].read_text())["outcome"] == "abort"
    assert engine.snapshots.promoted()["snapshot_id"] == before


def test_evaluator_cannot_mutate_candidate_snapshot(tmp_path: Path):
    engine = _engine(tmp_path)
    promoted_id = engine.snapshots.promoted()["snapshot_id"]

    def trainer(candidate, _lesson, evidence):
        value = json.loads((candidate / "mind.json").read_text())
        value["generation"] += 1
        (candidate / "mind.json").write_text(json.dumps(value), encoding="utf-8")
        artifact = evidence / "training.json"
        artifact.write_text("{}", encoding="utf-8")
        return {
            "format": "gum-school-training-summary-v1", "completed": True,
            "interactions": 10, "wall_clock_seconds": 1.0, "peak_memory_mb": 1.0,
            "artifact_storage_mb": 1.0, "evidence_path": "evidence/training.json",
            "evidence_sha256": _sha(artifact),
        }

    def evaluator(snapshot, _lesson, _evidence):
        (snapshot / "mind.json").write_text("tampered", encoding="utf-8")
        return {}

    with pytest.raises(Exception):
        engine.run_lesson(trainer, evaluator)
    decision = json.loads(next(engine.runs.glob("*/DECISION.json")).read_text())
    assert decision["outcome"] == "abort"
    assert engine.snapshots.verify(promoted_id)["valid"]


def test_all_previously_promoted_capabilities_are_retention_protected(tmp_path: Path):
    engine = _engine(tmp_path)
    first_run = _prepare_evaluated_run(engine)
    assert engine.finalize(first_run)["outcome"] == "promote"

    second = engine.begin_lesson()
    second_run = second["run_id"]
    engine.mark_training_complete(second_run, _training_summary(engine, second_run))
    engine.record_evaluation(second_run, _evaluation(engine, second_run, retention=[]))
    decision = engine.finalize(second_run)
    assert decision["outcome"] == "quarantine"
    reasons = decision["gates"]["gates"]["retention"]["reasons"]
    assert any("object-persistence" in reason for reason in reasons)
    assert any("occlusion-tracking" in reason for reason in reasons)


def test_workspace_rejects_curriculum_drift(tmp_path: Path):
    engine = _engine(tmp_path)
    changed = deepcopy(load_json(DEFAULT_CURRICULUM))
    changed["title"] = "A materially different but structurally valid curriculum title"
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(SchoolEngineError, match="differs from the workspace's frozen curriculum"):
        SchoolEngine(engine.workspace, changed_path)


def test_training_summary_rejects_evidence_path_traversal(tmp_path: Path):
    engine = _engine(tmp_path)
    run_id = engine.begin_lesson()["run_id"]
    summary = _training_summary(engine, run_id)
    summary["evidence_path"] = "../outside.json"
    with pytest.raises(SchoolEngineError, match="stay inside"):
        engine.mark_training_complete(run_id, summary)


def test_initialization_recovers_if_pointer_was_written_before_progress(tmp_path: Path):
    workspace = tmp_path / "school"
    engine = SchoolEngine(workspace, DEFAULT_CURRICULUM)
    manifest = engine.snapshots.create(_initial_state(tmp_path))
    engine.snapshots.promote(manifest["snapshot_id"], expected_current=None, run_id="initialization")
    assert not engine.progress_path.exists()
    reopened = SchoolEngine(workspace, DEFAULT_CURRICULUM)
    assert reopened._progress()["initial_snapshot_id"] == manifest["snapshot_id"]
    assert reopened.ledger.verify()["valid"]


def test_only_verified_isolated_source_cells_complete_transfer_matrix(tmp_path: Path):
    engine = _engine(tmp_path)
    policy = engine.curriculum["transfer_matrix"]
    sources = policy["source_schools"]
    targets = [row["school"] for row in policy["targets"]]
    for source in sources:
        for target in targets:
            stored = engine.record_transfer_cell(_transfer_cell(engine, source, target))
            assert stored["fresh_advantage"] == pytest.approx(1.8)
    report = engine.write_report()
    assert report["transfer_matrix"]["complete"]
    assert len(report["transfer_matrix"]["cells"]) == 9
    assert report["transfer_matrix"]["missing_pairs"] == []


def test_transfer_cell_rejects_nonisolated_or_wrong_origin_branch(tmp_path: Path):
    engine = _engine(tmp_path)
    record = _transfer_cell(engine, "object-laboratory", "causal-workshop")
    record["trained_schools"] = ["object-laboratory", "causal-workshop"]
    with pytest.raises(SchoolEngineError, match="exactly its declared source"):
        engine.record_transfer_cell(record)
    record = _transfer_cell(engine, "object-laboratory", "causal-workshop")
    record["origin_snapshot_id"] = "sha256-" + "0" * 64
    with pytest.raises(SchoolEngineError, match="pre-curriculum snapshot"):
        engine.record_transfer_cell(record)


def test_progress_tampering_is_detected_against_anchored_ledger(tmp_path: Path):
    engine = _engine(tmp_path)
    progress = engine._progress()
    progress["promoted_lessons"] = ["object-laboratory.occlusion.001"]
    engine._write_progress(progress)
    with pytest.raises(SchoolEngineError, match="progress differs from the school ledger"):
        SchoolEngine(engine.workspace, DEFAULT_CURRICULUM)


def test_prepared_abort_is_completed_after_reopen(tmp_path: Path):
    engine = _engine(tmp_path)
    run_id = engine.begin_lesson()["run_id"]
    state = engine._run_state(run_id)
    state["abort_intent"] = {"reason": "simulated interruption", "prepared_at_utc": "test"}
    state["status"] = "abort-prepared"
    engine._write_run_state(run_id, state)
    reopened = SchoolEngine(engine.workspace, DEFAULT_CURRICULUM)
    assert reopened._run_state(run_id)["status"] == "aborted"
    decision = json.loads((reopened._run_folder(run_id) / "DECISION.json").read_text())
    assert decision["outcome"] == "abort"
    assert reopened.status()["active"]["status"] == "idle"
