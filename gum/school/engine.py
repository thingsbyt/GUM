"""Crash-resumable candidate, evaluation, and promotion lifecycle for GUM School."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Callable
import uuid

from gum.lineage import HashLedger, canonical, file_sha256
from gum.storage import atomic_write_json

from .evaluation import (
    SchoolEvaluationError,
    evaluate_promotion_gates,
    validate_evaluation_record,
    validate_training_summary,
)
from .snapshots import SnapshotError, SnapshotStore
from .validation import load_json, validate_curriculum


FINAL_RUN_STATUSES = frozenset({"promoted", "quarantined", "aborted"})


class SchoolEngineError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SchoolEngineError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise SchoolEngineError(f"{label} must contain a JSON object")
    return value


class SchoolEngine:
    """Single-writer Phase 1 engine. It does not register worlds or train by itself."""

    format = "gum-school-engine-v1"

    def __init__(self, workspace: Path, curriculum_path: Path):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.runs = self.workspace / "runs"
        self.quarantine = self.workspace / "quarantine"
        self.transfer_cells = self.workspace / "transfer-matrix"
        self.runs.mkdir(exist_ok=True)
        self.quarantine.mkdir(exist_ok=True)
        self.transfer_cells.mkdir(exist_ok=True)
        self.config_path = self.workspace / "SCHOOL.json"
        self.curriculum_path = self.workspace / "CURRICULUM.json"
        self.active_path = self.workspace / "ACTIVE.json"
        self.progress_path = self.workspace / "PROGRESS.json"
        self.report_path = self.workspace / "SCHOOL_REPORT.json"

        curriculum = validate_curriculum(load_json(curriculum_path))
        self.curriculum = curriculum
        self.curriculum_sha256 = hashlib.sha256(canonical(curriculum)).hexdigest()
        self.lesson_by_id = {lesson["lesson_id"]: lesson for lesson in curriculum["lessons"]}
        self.ordered_lessons = sorted(curriculum["lessons"], key=lambda row: row["sequence"])
        self._lock_curriculum()

        self.ledger = HashLedger(self.workspace / "SCHOOL_LEDGER.jsonl")
        verification = self.ledger.verify()
        if not verification["valid"]:
            raise SchoolEngineError(f"school ledger failed verification: {verification['errors']}")
        self.snapshots = SnapshotStore(self.workspace)
        self.recover()
        self._verify_consistency()

    def _lock_curriculum(self) -> None:
        if self.config_path.exists():
            config = _read_json(self.config_path, "school configuration")
            if config.get("format") != self.format:
                raise SchoolEngineError("unsupported school configuration")
            if config.get("curriculum_sha256") != self.curriculum_sha256:
                raise SchoolEngineError("curriculum differs from the workspace's frozen curriculum")
            locked = validate_curriculum(load_json(self.curriculum_path))
            if hashlib.sha256(canonical(locked)).hexdigest() != self.curriculum_sha256:
                raise SchoolEngineError("stored curriculum differs from its locked hash")
            return
        atomic_write_json(self.curriculum_path, self.curriculum, backup=False, sort_keys=True)
        atomic_write_json(
            self.config_path,
            {
                "format": self.format,
                "curriculum_id": self.curriculum["curriculum_id"],
                "curriculum_sha256": self.curriculum_sha256,
                "created_at_utc": _utc_now(),
                "single_writer_required": True,
            },
            backup=False,
            sort_keys=True,
        )

    def _append_once(self, event: str, payload: dict, *, event_key: str, run_id: str | None = None) -> dict:
        if self.ledger.path.exists():
            with self.ledger.path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("payload", {}).get("event_key") == event_key:
                        return row
        return self.ledger.append(event, {"event_key": event_key, **payload}, run_id=run_id)

    def _active(self) -> dict:
        if not self.active_path.exists():
            return {"format": "gum-school-active-v1", "status": "idle", "run_id": None}
        value = _read_json(self.active_path, "active-run pointer")
        if value.get("format") != "gum-school-active-v1":
            raise SchoolEngineError("unsupported active-run pointer")
        return value

    def _set_active(self, run_id: str | None) -> None:
        atomic_write_json(
            self.active_path,
            {
                "format": "gum-school-active-v1",
                "status": "idle" if run_id is None else "active",
                "run_id": run_id,
                "updated_at_utc": _utc_now(),
            },
            sort_keys=True,
        )

    def _progress(self) -> dict:
        if not self.progress_path.exists():
            return {
                "format": "gum-school-progress-v1",
                "initial_snapshot_id": None,
                "promoted_lessons": [],
                "updated_at_utc": _utc_now(),
            }
        value = _read_json(self.progress_path, "school progress")
        if value.get("format") != "gum-school-progress-v1":
            raise SchoolEngineError("unsupported school progress format")
        return value

    def _write_progress(self, progress: dict) -> None:
        progress = dict(progress)
        progress["updated_at_utc"] = _utc_now()
        atomic_write_json(self.progress_path, progress, sort_keys=True)

    def _run_folder(self, run_id: str) -> Path:
        try:
            parsed = uuid.UUID(str(run_id))
        except ValueError as error:
            raise SchoolEngineError(f"invalid run identifier {run_id!r}") from error
        return self.runs / str(parsed)

    def _run_state(self, run_id: str) -> dict:
        value = _read_json(self._run_folder(run_id) / "RUN.json", f"run {run_id}")
        if value.get("format") != "gum-school-run-v1" or value.get("run_id") != run_id:
            raise SchoolEngineError(f"run {run_id} has an invalid state record")
        return value

    def _write_run_state(self, run_id: str, state: dict) -> None:
        state = dict(state)
        state["updated_at_utc"] = _utc_now()
        atomic_write_json(self._run_folder(run_id) / "RUN.json", state, sort_keys=True)

    def _nonfinal_runs(self) -> list[str]:
        result = []
        for path in sorted(self.runs.glob("*/RUN.json")):
            state = _read_json(path, f"run state {path.parent.name}")
            if state.get("status") not in FINAL_RUN_STATUSES:
                result.append(path.parent.name)
        return result

    def initialize_promoted(self, state_directory: Path) -> dict:
        if self.snapshots.promoted() is not None:
            raise SchoolEngineError("the promoted learner is already initialized")
        manifest = self.snapshots.create(state_directory)
        pointer = self.snapshots.promote(
            manifest["snapshot_id"], expected_current=None, run_id="initialization"
        )
        self._write_progress(
            {
                "format": "gum-school-progress-v1",
                "initial_snapshot_id": manifest["snapshot_id"],
                "promoted_lessons": [],
            }
        )
        self._append_once(
            "school-initialized",
            {"snapshot_id": manifest["snapshot_id"], "curriculum_sha256": self.curriculum_sha256},
            event_key="school-initialized",
        )
        self.write_report()
        return pointer

    def next_lesson(self) -> dict | None:
        promoted = set(self._progress()["promoted_lessons"])
        for lesson in self.ordered_lessons:
            if lesson["lesson_id"] not in promoted:
                return deepcopy(lesson)
        return None

    def begin_lesson(self, lesson_id: str | None = None) -> dict:
        pointer = self.snapshots.promoted()
        if pointer is None:
            raise SchoolEngineError("initialize a promoted learner before beginning lessons")
        active = self._active()
        if active["status"] != "idle":
            raise SchoolEngineError(f"run {active['run_id']} is already active")
        expected = self.next_lesson()
        if expected is None:
            raise SchoolEngineError("all curriculum lessons are already promoted")
        selected_id = expected["lesson_id"] if lesson_id is None else lesson_id
        if selected_id != expected["lesson_id"]:
            raise SchoolEngineError(
                f"lesson order is fixed; next lesson is {expected['lesson_id']!r}"
            )

        run_id = str(uuid.uuid4())
        folder = self._run_folder(run_id)
        folder.mkdir(parents=True)
        candidate = folder / "candidate"
        evidence = folder / "evidence"
        evidence.mkdir()
        self.snapshots.materialize(pointer["snapshot_id"], candidate)
        lesson = self.lesson_by_id[selected_id]
        state = {
            "format": "gum-school-run-v1",
            "run_id": run_id,
            "status": "candidate-prepared",
            "lesson_id": selected_id,
            "lesson_sha256": hashlib.sha256(canonical(lesson)).hexdigest(),
            "curriculum_sha256": self.curriculum_sha256,
            "base_snapshot_id": pointer["snapshot_id"],
            "candidate_snapshot_id": None,
            "created_at_utc": _utc_now(),
        }
        self._write_run_state(run_id, state)
        self._set_active(run_id)
        self._append_once(
            "lesson-started",
            {"lesson_id": selected_id, "base_snapshot_id": pointer["snapshot_id"]},
            event_key=f"lesson-started:{run_id}",
            run_id=run_id,
        )
        return {"run_id": run_id, "lesson": deepcopy(lesson), "candidate_directory": candidate}

    def _verify_artifact(self, run_id: str, relative: str, expected: str) -> None:
        path = self._run_folder(run_id) / PureRelativePath(relative).path
        if not path.is_file():
            raise SchoolEngineError(f"evidence artifact is missing: {relative}")
        if f"sha256:{file_sha256(path)}" != expected:
            raise SchoolEngineError(f"evidence artifact hash differs: {relative}")

    def _verify_workspace_artifact(self, relative: str, expected: str) -> None:
        path = self.workspace / PureRelativePath(relative).path
        if not path.is_file():
            raise SchoolEngineError(f"workspace evidence artifact is missing: {relative}")
        if f"sha256:{file_sha256(path)}" != expected:
            raise SchoolEngineError(f"workspace evidence artifact hash differs: {relative}")

    def mark_training_complete(self, run_id: str, summary: dict) -> dict:
        state = self._run_state(run_id)
        path = self._run_folder(run_id) / "TRAINING.json"
        if state["status"] != "candidate-prepared":
            if path.exists():
                existing = _read_json(path, "training summary")
                if canonical(existing) == canonical(summary):
                    return existing
            raise SchoolEngineError(f"run {run_id} cannot record training from status {state['status']!r}")
        try:
            validate_training_summary(summary)
        except SchoolEvaluationError as error:
            raise SchoolEngineError(str(error)) from error
        self._verify_artifact(run_id, summary["evidence_path"], summary["evidence_sha256"])
        lesson = self.lesson_by_id[state["lesson_id"]]
        if summary["interactions"] > lesson["training"]["interaction_budget"]:
            raise SchoolEngineError("training interactions exceed the precommitted budget")
        manifest = self.snapshots.create(self._run_folder(run_id) / "candidate")
        evaluation_state = self._run_folder(run_id) / "evaluation-state"
        self.snapshots.materialize(manifest["snapshot_id"], evaluation_state)
        atomic_write_json(path, summary, backup=False, sort_keys=True)
        state["candidate_snapshot_id"] = manifest["snapshot_id"]
        state["status"] = "training-complete"
        self._write_run_state(run_id, state)
        self._append_once(
            "training-completed",
            {"lesson_id": state["lesson_id"], "candidate_snapshot_id": manifest["snapshot_id"],
             "summary_sha256": file_sha256(path)},
            event_key=f"training-completed:{run_id}",
            run_id=run_id,
        )
        return summary

    def candidate_snapshot_directory(self, run_id: str) -> Path:
        state = self._run_state(run_id)
        snapshot_id = state.get("candidate_snapshot_id")
        if not snapshot_id:
            raise SchoolEngineError("candidate has not completed training")
        verification = self.snapshots.verify_materialization(
            snapshot_id, self._run_folder(run_id) / "evaluation-state"
        )
        if not verification["valid"]:
            raise SchoolEngineError(f"candidate evaluation state failed verification: {verification['errors']}")
        return self._run_folder(run_id) / "evaluation-state"

    def record_evaluation(self, run_id: str, evaluation: dict) -> dict:
        state = self._run_state(run_id)
        path = self._run_folder(run_id) / "EVALUATION.json"
        if state["status"] != "training-complete":
            if path.exists():
                existing = _read_json(path, "evaluation")
                if canonical(existing) == canonical(evaluation):
                    return existing
            raise SchoolEngineError(f"run {run_id} cannot record evaluation from status {state['status']!r}")
        try:
            validate_evaluation_record(evaluation)
        except SchoolEvaluationError as error:
            raise SchoolEngineError(str(error)) from error
        verification = self.snapshots.verify(state["candidate_snapshot_id"])
        if not verification["valid"]:
            raise SchoolEngineError(f"evaluation mutated or corrupted the candidate: {verification['errors']}")
        materialization = self.snapshots.verify_materialization(
            state["candidate_snapshot_id"], self._run_folder(run_id) / "evaluation-state"
        )
        if not materialization["valid"]:
            raise SchoolEngineError(
                f"evaluation mutated its frozen state copy: {materialization['errors']}"
            )
        self._verify_artifact(
            run_id, evaluation["replay"]["artifact_path"], evaluation["replay"]["artifact_sha256"]
        )
        for transfer in evaluation["transfer_results"]:
            self._verify_artifact(run_id, transfer["evidence_path"], transfer["evidence_sha256"])
        training = _read_json(self._run_folder(run_id) / "TRAINING.json", "training summary")
        for field in ("interactions", "wall_clock_seconds", "peak_memory_mb", "artifact_storage_mb"):
            if evaluation["resources"][field] < training[field]:
                raise SchoolEngineError(
                    f"evaluation.resources.{field} is lower than the recorded training use"
                )
        atomic_write_json(path, evaluation, backup=False, sort_keys=True)
        state["status"] = "evaluation-recorded"
        self._write_run_state(run_id, state)
        self._append_once(
            "evaluation-recorded",
            {"lesson_id": state["lesson_id"], "evaluation_sha256": file_sha256(path)},
            event_key=f"evaluation-recorded:{run_id}",
            run_id=run_id,
        )
        return evaluation

    def _required_protected_skills(self) -> set[str]:
        progress = self._progress()
        return {
            capability
            for lesson_id in progress["promoted_lessons"]
            for capability in self.lesson_by_id[lesson_id]["capabilities"]
        }

    def finalize(self, run_id: str) -> dict:
        state = self._run_state(run_id)
        decision_path = self._run_folder(run_id) / "DECISION.json"
        if state["status"] in FINAL_RUN_STATUSES:
            return _read_json(decision_path, "decision")
        if state["status"] == "evaluation-recorded":
            evaluation = _read_json(self._run_folder(run_id) / "EVALUATION.json", "evaluation")
            lesson = self.lesson_by_id[state["lesson_id"]]
            gates = evaluate_promotion_gates(
                lesson,
                evaluation,
                required_protected_skills=self._required_protected_skills(),
            )
            state["decision_intent"] = {
                "outcome": "promote" if gates["all_passed"] else "quarantine",
                "gates": gates,
                "prepared_at_utc": _utc_now(),
            }
            state["status"] = "decision-prepared"
            self._write_run_state(run_id, state)
        elif state["status"] != "decision-prepared":
            raise SchoolEngineError(f"run {run_id} cannot finalize from status {state['status']!r}")
        return self._finish_prepared_decision(run_id)

    def _after_promotion_pointer_updated(self, run_id: str) -> None:
        """Test hook: interruption here must be recoverable and idempotent."""

    def _finish_prepared_decision(self, run_id: str) -> dict:
        state = self._run_state(run_id)
        if state["status"] != "decision-prepared":
            raise SchoolEngineError("run has no prepared decision")
        intent = state["decision_intent"]
        outcome = intent["outcome"]
        if outcome == "promote":
            self.snapshots.promote(
                state["candidate_snapshot_id"],
                expected_current=state["base_snapshot_id"],
                run_id=run_id,
            )
            self._after_promotion_pointer_updated(run_id)
            progress = self._progress()
            if state["lesson_id"] not in progress["promoted_lessons"]:
                progress["promoted_lessons"].append(state["lesson_id"])
                progress["promoted_lessons"].sort(
                    key=lambda lesson_id: self.lesson_by_id[lesson_id]["sequence"]
                )
                self._write_progress(progress)
            final_status = "promoted"
        else:
            quarantine_record = {
                "format": "gum-school-quarantine-v1",
                "run_id": run_id,
                "lesson_id": state["lesson_id"],
                "candidate_snapshot_id": state["candidate_snapshot_id"],
                "base_snapshot_id": state["base_snapshot_id"],
                "failed_gates": [
                    gate for gate, result in intent["gates"]["gates"].items()
                    if not result["passed"]
                ],
                "quarantined_at_utc": _utc_now(),
            }
            atomic_write_json(
                self.quarantine / f"{run_id}.json", quarantine_record, backup=False, sort_keys=True
            )
            final_status = "quarantined"

        decision = {
            "format": "gum-school-decision-v1",
            "run_id": run_id,
            "lesson_id": state["lesson_id"],
            "outcome": outcome,
            "base_snapshot_id": state["base_snapshot_id"],
            "candidate_snapshot_id": state["candidate_snapshot_id"],
            "curriculum_sha256": self.curriculum_sha256,
            "gates": intent["gates"],
            "decided_at_utc": _utc_now(),
        }
        decision_path = self._run_folder(run_id) / "DECISION.json"
        if decision_path.exists():
            existing = _read_json(decision_path, "decision")
            decision = existing
        else:
            atomic_write_json(decision_path, decision, backup=False, sort_keys=True)
        state["status"] = final_status
        state["decision_intent"] = intent
        self._write_run_state(run_id, state)
        self._append_once(
            "promotion-decided",
            {"lesson_id": state["lesson_id"], "outcome": outcome,
             "decision_sha256": file_sha256(decision_path),
             "candidate_snapshot_id": state["candidate_snapshot_id"]},
            event_key=f"promotion-decided:{run_id}",
            run_id=run_id,
        )
        self._set_active(None)
        self.write_report()
        return decision

    def abort(self, run_id: str, reason: str) -> dict:
        state = self._run_state(run_id)
        decision_path = self._run_folder(run_id) / "DECISION.json"
        if state["status"] in FINAL_RUN_STATUSES:
            return _read_json(decision_path, "decision")
        if state["status"] == "decision-prepared":
            raise SchoolEngineError("a prepared promotion decision must be recovered, not aborted")
        if state["status"] == "abort-prepared":
            return self._finish_abort(run_id)
        if not isinstance(reason, str) or not reason.strip():
            raise SchoolEngineError("abort reason must be non-empty")
        state["abort_intent"] = {"reason": reason.strip()[:1000], "prepared_at_utc": _utc_now()}
        state["status"] = "abort-prepared"
        self._write_run_state(run_id, state)
        return self._finish_abort(run_id)

    def _finish_abort(self, run_id: str) -> dict:
        state = self._run_state(run_id)
        if state["status"] != "abort-prepared":
            raise SchoolEngineError("run has no prepared abort")
        decision_path = self._run_folder(run_id) / "DECISION.json"
        reason = state["abort_intent"]["reason"]
        decision = {
            "format": "gum-school-decision-v1",
            "run_id": run_id,
            "lesson_id": state["lesson_id"],
            "outcome": "abort",
            "base_snapshot_id": state["base_snapshot_id"],
            "candidate_snapshot_id": state.get("candidate_snapshot_id"),
            "curriculum_sha256": self.curriculum_sha256,
            "reason": reason,
            "decided_at_utc": _utc_now(),
        }
        atomic_write_json(decision_path, decision, backup=False, sort_keys=True)
        atomic_write_json(
            self.quarantine / f"{run_id}.json",
            {"format": "gum-school-quarantine-v1", "run_id": run_id,
             "lesson_id": state["lesson_id"], "reason": decision["reason"],
             "quarantined_at_utc": _utc_now()},
            backup=False,
            sort_keys=True,
        )
        state["status"] = "aborted"
        self._write_run_state(run_id, state)
        self._append_once(
            "lesson-aborted",
            {"lesson_id": state["lesson_id"], "reason": decision["reason"],
             "decision_sha256": file_sha256(decision_path)},
            event_key=f"lesson-aborted:{run_id}",
            run_id=run_id,
        )
        self._set_active(None)
        self.write_report()
        return decision

    def run_lesson(
        self,
        trainer: Callable[[Path, dict, Path], dict],
        evaluator: Callable[[Path, dict, Path], dict],
        *,
        lesson_id: str | None = None,
    ) -> dict:
        """Run source-reviewed callables; curriculum data never names executable code."""
        started = self.begin_lesson(lesson_id)
        run_id = started["run_id"]
        folder = self._run_folder(run_id)
        try:
            summary = trainer(started["candidate_directory"], deepcopy(started["lesson"]), folder / "evidence")
            self.mark_training_complete(run_id, summary)
            evaluation = evaluator(
                self.candidate_snapshot_directory(run_id),
                deepcopy(started["lesson"]),
                folder / "evidence",
            )
            self.record_evaluation(run_id, evaluation)
            return self.finalize(run_id)
        except Exception as error:
            state = self._run_state(run_id)
            if state["status"] not in FINAL_RUN_STATUSES and state["status"] != "decision-prepared":
                self.abort(run_id, f"{type(error).__name__}: {error}")
            raise

    def recover(self) -> dict:
        pointer = self.snapshots.promoted()
        if pointer is not None and not self.progress_path.exists():
            self._write_progress(
                {
                    "format": "gum-school-progress-v1",
                    "initial_snapshot_id": pointer["snapshot_id"],
                    "promoted_lessons": [],
                }
            )
        elif pointer is None and self.progress_path.exists():
            raise SchoolEngineError("school progress exists without a promoted snapshot")
        if pointer is not None:
            progress = self._progress()
            initial_snapshot_id = progress.get("initial_snapshot_id")
            if not initial_snapshot_id or not self.snapshots.verify(initial_snapshot_id)["valid"]:
                raise SchoolEngineError("initial snapshot recorded in progress is missing or invalid")
            self._append_once(
                "school-initialized",
                {"snapshot_id": initial_snapshot_id, "curriculum_sha256": self.curriculum_sha256},
                event_key="school-initialized",
            )
        active = self._active()
        if active["status"] == "idle":
            incomplete = self._nonfinal_runs()
            if len(incomplete) > 1:
                raise SchoolEngineError(f"multiple incomplete runs require manual review: {incomplete}")
            if incomplete:
                self._set_active(incomplete[0])
                active = self._active()
        if active["status"] == "idle":
            return {"status": "idle", "run_id": None}
        run_id = active["run_id"]
        state = self._run_state(run_id)
        self._append_once(
            "lesson-started",
            {"lesson_id": state["lesson_id"], "base_snapshot_id": state["base_snapshot_id"]},
            event_key=f"lesson-started:{run_id}",
            run_id=run_id,
        )
        if state["status"] == "decision-prepared":
            self._finish_prepared_decision(run_id)
            return {"status": "recovered-final", "run_id": run_id}
        if state["status"] == "abort-prepared":
            self._finish_abort(run_id)
            return {"status": "recovered-final", "run_id": run_id}
        if state["status"] in FINAL_RUN_STATUSES:
            decision_path = self._run_folder(run_id) / "DECISION.json"
            if decision_path.exists():
                decision = _read_json(decision_path, "decision")
                event = "lesson-aborted" if decision["outcome"] == "abort" else "promotion-decided"
                self._append_once(
                    event,
                    {"lesson_id": state["lesson_id"], "outcome": decision["outcome"],
                     "decision_sha256": file_sha256(decision_path)},
                    event_key=f"{event}:{run_id}",
                    run_id=run_id,
                )
            self._set_active(None)
            self.write_report()
            return {"status": "recovered-final", "run_id": run_id}
        return {"status": state["status"], "run_id": run_id}

    def _verify_consistency(self) -> None:
        pointer = self.snapshots.promoted()
        if pointer is None:
            return
        progress = self._progress()
        initialized_snapshot = None
        promoted_events = []
        if self.ledger.path.exists():
            with self.ledger.path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    event_key = row.get("payload", {}).get("event_key")
                    if event_key == "school-initialized":
                        initialized_snapshot = row["payload"].get("snapshot_id")
                    elif (row.get("event") == "promotion-decided"
                          and row.get("payload", {}).get("outcome") == "promote"):
                        promoted_events.append(row)
        if initialized_snapshot != progress.get("initial_snapshot_id"):
            raise SchoolEngineError("progress initial snapshot differs from the school ledger")
        expected_lessons = [row["payload"]["lesson_id"] for row in promoted_events]
        if progress.get("promoted_lessons") != expected_lessons:
            raise SchoolEngineError("promoted lesson progress differs from the school ledger")
        expected_snapshot = (
            initialized_snapshot if not promoted_events
            else promoted_events[-1]["payload"].get("candidate_snapshot_id")
        )
        if pointer["snapshot_id"] != expected_snapshot:
            raise SchoolEngineError("promoted pointer differs from the last ledger decision")

    def write_report(self) -> dict:
        decisions = []
        diagnostic_cells = []
        for path in sorted(self.runs.glob("*/DECISION.json")):
            decision = _read_json(path, f"decision {path.parent.name}")
            decisions.append(decision)
            evaluation_path = path.parent / "EVALUATION.json"
            if not evaluation_path.exists():
                continue
            evaluation = _read_json(evaluation_path, f"evaluation {path.parent.name}")
            source_school = self.lesson_by_id[decision["lesson_id"]]["school"]
            for row in evaluation.get("transfer_results", []):
                diagnostic_cells.append({"run_id": decision["run_id"], "source_school": source_school, **row})
        official_cells = [
            _read_json(path, f"transfer cell {path.stem}")
            for path in sorted(self.transfer_cells.glob("*.json"))
        ]
        pointer = self.snapshots.promoted()
        progress = self._progress()
        required_sources = self.curriculum["transfer_matrix"]["source_schools"]
        required_targets = [row["school"] for row in self.curriculum["transfer_matrix"]["targets"]]
        observed_pairs = {(row["source_school"], row["target_school"]) for row in official_cells}
        missing_pairs = [
            [source, target]
            for source in required_sources
            for target in required_targets
            if (source, target) not in observed_pairs
        ]
        report = {
            "format": "gum-school-report-v1",
            "curriculum_id": self.curriculum["curriculum_id"],
            "curriculum_sha256": self.curriculum_sha256,
            "promoted_snapshot_id": None if pointer is None else pointer["snapshot_id"],
            "initial_snapshot_id": progress["initial_snapshot_id"],
            "promoted_lessons": progress["promoted_lessons"],
            "next_lesson_id": None if self.next_lesson() is None else self.next_lesson()["lesson_id"],
            "decisions": decisions,
            "transfer_matrix": {
                "protocol": self.curriculum["transfer_matrix"],
                "cells": official_cells,
                "missing_pairs": missing_pairs,
                "complete": not missing_pairs,
                "diagnostic_sequential_results": diagnostic_cells,
                "warning": "Only verified isolated-source branch cells count toward completeness.",
            },
            "ledger": self.ledger.verify(),
            "generated_at_utc": _utc_now(),
        }
        atomic_write_json(self.report_path, report, sort_keys=True)
        return report

    def record_transfer_cell(self, record: dict) -> dict:
        fields = {
            "format", "source_school", "target_school", "origin_snapshot_id",
            "trained_schools", "target_evaluation_training", "matched_fresh",
            "same_perception", "same_resource_limits", "trials", "candidate_success_rate",
            "fresh_success_rate", "candidate_mean_interactions", "fresh_mean_interactions",
            "uncertainty_rate", "unnecessary_action_rate", "confidence_level",
            "success_interval", "fresh_advantage_interval", "branch_verification",
            "evidence_path", "evidence_sha256",
        }
        if not isinstance(record, dict) or set(record) != fields:
            missing = sorted(fields - set(record)) if isinstance(record, dict) else sorted(fields)
            unknown = sorted(set(record) - fields) if isinstance(record, dict) else []
            raise SchoolEngineError(f"transfer cell fields differ; missing={missing}, unknown={unknown}")
        if record["format"] != "gum-school-transfer-cell-v1":
            raise SchoolEngineError("unsupported transfer-cell format")
        policy = self.curriculum["transfer_matrix"]
        sources = set(policy["source_schools"])
        targets = {row["school"] for row in policy["targets"]}
        source = record["source_school"]
        target = record["target_school"]
        if source not in sources or target not in targets:
            raise SchoolEngineError("transfer cell source or target is outside the curriculum matrix")
        progress = self._progress()
        if record["origin_snapshot_id"] != progress["initial_snapshot_id"]:
            raise SchoolEngineError("transfer branch did not originate from the pre-curriculum snapshot")
        if record["trained_schools"] != [source]:
            raise SchoolEngineError("transfer branch must train exactly its declared source school")
        required_constants = {
            "target_evaluation_training": False,
            "matched_fresh": True,
            "same_perception": True,
            "same_resource_limits": True,
        }
        for field, expected in required_constants.items():
            if record[field] is not expected:
                raise SchoolEngineError(f"transfer cell {field} must be {expected}")
        if record["trials"] != policy["trials_per_cell"]:
            raise SchoolEngineError("transfer cell trial count differs from the precommitted count")
        if record["confidence_level"] != policy["confidence_level"]:
            raise SchoolEngineError("transfer cell confidence level differs from the precommitted level")
        for field in ("candidate_success_rate", "fresh_success_rate", "uncertainty_rate",
                      "unnecessary_action_rate"):
            value = record[field]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or not 0 <= value <= 1):
                raise SchoolEngineError(f"transfer cell {field} must be between zero and one")
        for field in ("candidate_mean_interactions", "fresh_mean_interactions"):
            value = record[field]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or value < 0):
                raise SchoolEngineError(f"transfer cell {field} must be finite and nonnegative")
        advantage = float(record["candidate_success_rate"]) / max(
            float(record["fresh_success_rate"]), 1.0 / float(record["trials"])
        )
        for field, point, maximum, method in (
            ("success_interval", float(record["candidate_success_rate"]), 1.0, "wilson"),
            ("fresh_advantage_interval", advantage, None, "paired-bootstrap"),
        ):
            interval = record[field]
            if not isinstance(interval, dict) or set(interval) != {"lower", "upper", "method"}:
                raise SchoolEngineError(f"transfer cell {field} is malformed")
            lower, upper = interval["lower"], interval["upper"]
            if (isinstance(lower, bool) or isinstance(upper, bool)
                    or not isinstance(lower, (int, float)) or not isinstance(upper, (int, float))
                    or not math.isfinite(float(lower)) or not math.isfinite(float(upper))
                    or lower < 0 or lower > upper or (maximum is not None and upper > maximum)
                    or not lower <= point <= upper or interval["method"] != method):
                raise SchoolEngineError(f"transfer cell {field} is invalid")
        verification = record["branch_verification"]
        verification_fields = {
            "source_only_training_verified", "target_frozen_verified", "matched_fresh_verified",
            "source_hashes_verified", "protocol_hashes_verified", "trajectory_verified",
        }
        if (not isinstance(verification, dict) or set(verification) != verification_fields
                or not all(value is True for value in verification.values())):
            raise SchoolEngineError("transfer branch verification is incomplete")
        if not isinstance(record["evidence_path"], str) or not record["evidence_path"].startswith(
            "transfer-evidence/"
        ):
            raise SchoolEngineError("transfer evidence must stay under transfer-evidence/")
        self._verify_workspace_artifact(record["evidence_path"], record["evidence_sha256"])
        stored = dict(record)
        stored["success_difference"] = (
            float(record["candidate_success_rate"]) - float(record["fresh_success_rate"])
        )
        stored["fresh_advantage"] = advantage
        stored["recorded_at_utc"] = _utc_now()
        destination = self.transfer_cells / f"{source}--{target}.json"
        if destination.exists():
            existing = _read_json(destination, "transfer cell")
            comparable = {key: value for key, value in existing.items() if key != "recorded_at_utc"}
            proposed = {key: value for key, value in stored.items() if key != "recorded_at_utc"}
            if canonical(comparable) != canonical(proposed):
                raise SchoolEngineError("refusing to overwrite an existing transfer-matrix cell")
            return existing
        atomic_write_json(destination, stored, backup=False, sort_keys=True)
        self._append_once(
            "transfer-cell-recorded",
            {"source_school": source, "target_school": target,
             "cell_sha256": file_sha256(destination)},
            event_key=f"transfer-cell:{source}:{target}",
        )
        self.write_report()
        return stored

    def status(self) -> dict:
        pointer = self.snapshots.promoted()
        active = self._active()
        progress = self._progress()
        return {
            "format": self.format,
            "curriculum_id": self.curriculum["curriculum_id"],
            "curriculum_sha256": self.curriculum_sha256,
            "promoted_snapshot_id": None if pointer is None else pointer["snapshot_id"],
            "promoted_generation": None if pointer is None else pointer["generation"],
            "promoted_lessons": progress["promoted_lessons"],
            "active": active,
            "next_lesson_id": None if self.next_lesson() is None else self.next_lesson()["lesson_id"],
            "ledger": self.ledger.verify(),
        }


class PureRelativePath:
    """Reject absolute and parent-traversing artifact paths before joining."""

    def __init__(self, value: str):
        if not isinstance(value, str) or not value or "\\" in value:
            raise SchoolEngineError("artifact path must be a relative POSIX path")
        parts = value.split("/")
        if value.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            raise SchoolEngineError("artifact path must stay inside the run directory")
        self.path = Path(*parts)
