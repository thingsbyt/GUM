"""Validated curriculum and crash-resumable GUM School lifecycle."""

from .engine import SchoolEngine, SchoolEngineError
from .evaluation import SchoolEvaluationError, evaluate_promotion_gates
from .snapshots import SnapshotError, SnapshotStore
from .admission import WorldAdmissionError, run_foundational_admission
from .learner import CrossSeedSchoolLearner, SchoolLearnerError
from .rehearsal import run_nonpromoting_rehearsal
from .training import (
    TrainingLaneConfig,
    initialize_school_learner,
    make_development_rehearsal_evaluator,
    make_rehearsal_trainer,
)
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
    "CrossSeedSchoolLearner",
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
    "SchoolLearnerError",
    "SnapshotError",
    "SnapshotStore",
    "TrainingLaneConfig",
    "WorldAdmissionError",
    "create_foundational_world",
    "create_lesson_world",
    "evaluate_promotion_gates",
    "initialize_school_learner",
    "load_json",
    "make_development_rehearsal_evaluator",
    "make_rehearsal_trainer",
    "run_foundational_admission",
    "run_nonpromoting_rehearsal",
    "validate_curriculum",
    "validate_curriculum_file",
    "validate_document",
]
