"""Validated curriculum and crash-resumable GUM School lifecycle."""

from .engine import SchoolEngine, SchoolEngineError
from .evaluation import SchoolEvaluationError, evaluate_promotion_gates
from .snapshots import SnapshotError, SnapshotStore
from .admission import WorldAdmissionError, run_foundational_admission
from .worlds import (
    CAUSAL_WORKSHOP_ADAPTER,
    CHANGING_MAZE_ADAPTER,
    FOUNDATIONAL_ADAPTERS,
    FOUNDATIONAL_LOADERS,
    GENERATOR_MECHANISMS,
    OBJECT_LABORATORY_ADAPTER,
    FoundationalWorldError,
    create_foundational_world,
    create_lesson_world,
)

from .validation import (
    CurriculumValidationError,
    load_json,
    validate_curriculum,
    validate_curriculum_file,
    validate_document,
)

__all__ = [
    "CurriculumValidationError",
    "CAUSAL_WORKSHOP_ADAPTER",
    "CHANGING_MAZE_ADAPTER",
    "FOUNDATIONAL_ADAPTERS",
    "FOUNDATIONAL_LOADERS",
    "GENERATOR_MECHANISMS",
    "FoundationalWorldError",
    "OBJECT_LABORATORY_ADAPTER",
    "SchoolEngine",
    "SchoolEngineError",
    "SchoolEvaluationError",
    "SnapshotError",
    "SnapshotStore",
    "WorldAdmissionError",
    "create_foundational_world",
    "create_lesson_world",
    "evaluate_promotion_gates",
    "load_json",
    "run_foundational_admission",
    "validate_curriculum",
    "validate_curriculum_file",
    "validate_document",
]
