"""Specification-time validation for GUM School.

Phase 0 deliberately contains no training, scheduling, or promotion runner.
"""

from .validation import (
    CurriculumValidationError,
    load_json,
    validate_curriculum,
    validate_curriculum_file,
    validate_document,
)

__all__ = [
    "CurriculumValidationError",
    "load_json",
    "validate_curriculum",
    "validate_curriculum_file",
    "validate_document",
]
