"""Validated curriculum and crash-resumable GUM School lifecycle."""

from .engine import SchoolEngine, SchoolEngineError
from .evaluation import SchoolEvaluationError, evaluate_promotion_gates
from .snapshots import SnapshotError, SnapshotStore

from .validation import (
    CurriculumValidationError,
    load_json,
    validate_curriculum,
    validate_curriculum_file,
    validate_document,
)

__all__ = [
    "CurriculumValidationError",
    "SchoolEngine",
    "SchoolEngineError",
    "SchoolEvaluationError",
    "SnapshotError",
    "SnapshotStore",
    "evaluate_promotion_gates",
    "load_json",
    "validate_curriculum",
    "validate_curriculum_file",
    "validate_document",
]
