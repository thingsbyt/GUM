"""Strict structural and semantic validation for GUM School specifications."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator
from referencing import Registry, Resource


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_DIRECTORY = REPOSITORY_ROOT / "curriculum" / "schemas"
DEFAULT_CURRICULUM = REPOSITORY_ROOT / "curriculum" / "gum-school-v1.json"

SCHEMA_FILES = {
    "gum-school-curriculum-v1": "gum-school-curriculum-v1.schema.json",
    "gum-school-lesson-v1": "gum-school-lesson-v1.schema.json",
    "gum-school-examination-v1": "gum-school-examination-v1.schema.json",
    "gum-school-world-admission-v1": "gum-school-world-admission-v1.schema.json",
}

# These names are rejected at every nesting level, even when a future schema
# accidentally becomes permissive. They describe task answers or audit-only
# state rather than learner-visible specification.
PROHIBITED_SOLUTION_KEYS = frozenset(
    {
        "action_map",
        "answer",
        "answer_key",
        "answers",
        "concept_assignment",
        "control_map",
        "correct_sequence",
        "goal_coordinates",
        "hazard_coordinates",
        "mechanism_description",
        "sealed_seed",
        "sealed_seeds",
        "solution",
        "solutions",
        "successful_sequence",
        "target_program",
        "task_solution",
    }
)


class CurriculumValidationError(ValueError):
    """Raised when a school specification fails one or more checks."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("GUM School validation failed:\n- " + "\n- ".join(self.errors))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def load_json(path: Path | str) -> Any:
    """Load UTF-8 JSON while rejecting ambiguous duplicate object fields."""
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise CurriculumValidationError([f"{source}: malformed JSON: {error}"]) from error


def _schema_resources(schema_directory: Path) -> tuple[dict[str, dict[str, Any]], Registry]:
    schemas: dict[str, dict[str, Any]] = {}
    registry = Registry()
    for schema_name, filename in SCHEMA_FILES.items():
        schema = load_json(schema_directory / filename)
        if not isinstance(schema, dict):
            raise CurriculumValidationError([f"schema {filename} must contain a JSON object"])
        Draft202012Validator.check_schema(schema)
        resource = Resource.from_contents(schema)
        if resource.id() is None:
            raise CurriculumValidationError([f"schema {filename} has no $id"])
        registry = registry.with_resource(resource.id(), resource)
        schemas[schema_name] = schema
    return schemas, registry


def _format_path(parts: Iterable[Any]) -> str:
    rendered = "$"
    for part in parts:
        rendered += f"[{part}]" if isinstance(part, int) else f".{part}"
    return rendered


def _schema_errors(
    document: Any,
    schema_name: str,
    *,
    schema_directory: Path,
) -> list[str]:
    schemas, registry = _schema_resources(schema_directory)
    schema = schemas.get(schema_name)
    if schema is None:
        return [f"unsupported schema identifier {schema_name!r}"]
    validator = Draft202012Validator(schema, registry=registry)
    return [
        f"{_format_path(error.absolute_path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(document),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    ]


def _scan_for_solution_fields(value: Any, path: tuple[Any, ...] = ()) -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            child_path = path + (key,)
            if normalized in PROHIBITED_SOLUTION_KEYS:
                errors.append(
                    f"{_format_path(child_path)}: prohibited solution-bearing field {key!r}"
                )
            errors.extend(_scan_for_solution_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_scan_for_solution_fields(child, path + (index,)))
    return errors


def _duplicates(rows: list[dict[str, Any]], key: str, label: str) -> list[str]:
    errors: list[str] = []
    seen: set[Any] = set()
    for index, row in enumerate(rows):
        value = row.get(key)
        if value in seen:
            errors.append(f"$.{label}[{index}].{key}: duplicate identifier {value!r}")
        seen.add(value)
    return errors


def _overlap_error(label: str, partitions: dict[str, set[int]]) -> list[str]:
    errors: list[str] = []
    names = list(partitions)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = sorted(partitions[left_name] & partitions[right_name])
            if overlap:
                errors.append(
                    f"{label}: overlapping seed partitions {left_name!r} and "
                    f"{right_name!r}: {overlap}"
                )
    return errors


def _semantic_curriculum_errors(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    lessons = document.get("lessons", [])
    examinations = document.get("examinations", [])
    admissions = document.get("world_admissions", [])
    if not all(isinstance(value, list) for value in (lessons, examinations, admissions)):
        return errors

    errors.extend(_duplicates(lessons, "lesson_id", "lessons"))
    errors.extend(_duplicates(lessons, "sequence", "lessons"))
    errors.extend(_duplicates(examinations, "examination_id", "examinations"))
    errors.extend(_duplicates(admissions, "admission_id", "world_admissions"))

    exam_by_id = {row.get("examination_id"): row for row in examinations if isinstance(row, dict)}
    admission_by_id = {row.get("admission_id"): row for row in admissions if isinstance(row, dict)}
    adapters = set(document.get("adapter_catalog", []))
    capabilities = set(document.get("capability_catalog", []))
    global_seed_partitions = {"training": set(), "development": set(), "retention": set()}
    total_training_interactions = 0
    total_evaluation_trials = 0
    budgets = document.get("policies", {}).get("budgets", {})

    for index, lesson in enumerate(lessons):
        if not isinstance(lesson, dict):
            continue
        label = f"$.lessons[{index}]"
        adapter = lesson.get("adapter")
        if adapter not in adapters:
            errors.append(f"{label}.adapter: adapter {adapter!r} is not in adapter_catalog")
        for capability in lesson.get("capabilities", []):
            if capability not in capabilities:
                errors.append(
                    f"{label}.capabilities: capability {capability!r} is not in capability_catalog"
                )

        admission_id = lesson.get("world_admission_id")
        admission = admission_by_id.get(admission_id)
        if admission is None:
            errors.append(f"{label}.world_admission_id: unknown admission {admission_id!r}")
        elif admission.get("adapter") != adapter or admission.get("school") != lesson.get("school"):
            errors.append(f"{label}: lesson and world admission adapter/school do not match")

        has_primary_examination = False
        for examination_id in lesson.get("evaluation", {}).get("examination_ids", []):
            examination = exam_by_id.get(examination_id)
            if examination is None:
                errors.append(f"{label}.evaluation.examination_ids: unknown examination {examination_id!r}")
            elif examination.get("adapter") == adapter and examination.get("school") == lesson.get("school"):
                has_primary_examination = True
        if not has_primary_examination:
            errors.append(f"{label}: lesson has no examination for its own adapter and school")

        training = lesson.get("training", {})
        evaluation = lesson.get("evaluation", {})
        retention = lesson.get("retention", {})
        global_seed_partitions["training"].update(training.get("seeds", []))
        global_seed_partitions["development"].update(evaluation.get("development_seeds", []))
        global_seed_partitions["retention"].update(retention.get("evaluation_seeds", []))
        errors.extend(
            _overlap_error(
                label,
                {
                    "training": set(training.get("seeds", [])),
                    "development": set(evaluation.get("development_seeds", [])),
                    "retention": set(retention.get("evaluation_seeds", [])),
                },
            )
        )

        interaction_budget = training.get("interaction_budget")
        max_interactions = lesson.get("resource_limits", {}).get("max_interactions")
        if isinstance(interaction_budget, int):
            total_training_interactions += interaction_budget
            maximum_lesson = budgets.get("maximum_lesson_interactions")
            if isinstance(maximum_lesson, int) and interaction_budget > maximum_lesson:
                errors.append(
                    f"{label}.training.interaction_budget: exceeds curriculum maximum_lesson_interactions"
                )
            if isinstance(max_interactions, int) and interaction_budget > max_interactions:
                errors.append(
                    f"{label}.resource_limits.max_interactions: impossible budget; lower than training.interaction_budget"
                )
        trials = evaluation.get("trials")
        new_seed_trials = evaluation.get("new_seed_trials")
        if isinstance(trials, int):
            total_evaluation_trials += trials
        if isinstance(new_seed_trials, int):
            total_evaluation_trials += new_seed_trials

        allowed = set(lesson.get("learner_inputs", []))
        prohibited = set(lesson.get("prohibited_inputs", []))
        collision = sorted(allowed & prohibited)
        if collision:
            errors.append(f"{label}: learner_inputs and prohibited_inputs overlap: {collision}")

    maximum_total = budgets.get("maximum_total_training_interactions")
    if isinstance(maximum_total, int) and total_training_interactions > maximum_total:
        errors.append(
            "$.policies.budgets.maximum_total_training_interactions: impossible budget; "
            f"less than declared lesson total {total_training_interactions}"
        )
    maximum_trials = budgets.get("maximum_evaluation_trials")
    if isinstance(maximum_trials, int) and total_evaluation_trials > maximum_trials:
        errors.append(
            "$.policies.budgets.maximum_evaluation_trials: impossible budget; "
            f"less than declared lesson total {total_evaluation_trials}"
        )

    errors.extend(_overlap_error("$", global_seed_partitions))

    expected_sequences = list(range(1, len(lessons) + 1))
    actual_sequences = sorted(
        lesson.get("sequence") for lesson in lessons
        if isinstance(lesson, dict) and isinstance(lesson.get("sequence"), int)
    )
    if actual_sequences != expected_sequences:
        errors.append(
            f"$.lessons: sequence values must be contiguous from 1; found {actual_sequences}"
        )

    for index, examination in enumerate(examinations):
        if not isinstance(examination, dict):
            continue
        label = f"$.examinations[{index}]"
        if examination.get("adapter") not in adapters:
            errors.append(f"{label}.adapter: adapter is not in adapter_catalog")
        for capability in examination.get("capabilities", []):
            if capability not in capabilities:
                errors.append(
                    f"{label}.capabilities: capability {capability!r} is not in capability_catalog"
                )
        collision = sorted(
            set(examination.get("learner_inputs", []))
            & set(examination.get("prohibited_inputs", []))
        )
        if collision:
            errors.append(f"{label}: learner_inputs and prohibited_inputs overlap: {collision}")

    for index, admission in enumerate(admissions):
        if not isinstance(admission, dict):
            continue
        label = f"$.world_admissions[{index}]"
        if admission.get("adapter") not in adapters:
            errors.append(f"{label}.adapter: adapter is not in adapter_catalog")
        errors.extend(
            _overlap_error(
                label,
                {
                    "training": set(admission.get("seed_partitions", {}).get("training", [])),
                    "development": set(admission.get("seed_partitions", {}).get("development", [])),
                },
            )
        )
        declaration = admission.get("declaration", {})
        minimum = declaration.get("reward_minimum")
        maximum = declaration.get("reward_maximum")
        if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)) and minimum >= maximum:
            errors.append(f"{label}.declaration: reward_minimum must be lower than reward_maximum")

    return errors


def validate_document(
    document: Any,
    *,
    schema_name: str | None = None,
    schema_directory: Path | str = DEFAULT_SCHEMA_DIRECTORY,
) -> list[str]:
    """Return every structural, leakage, and semantic error in a specification."""
    errors = _scan_for_solution_fields(document)
    if not isinstance(document, dict):
        return errors + ["$: document must be a JSON object"]
    selected_schema = schema_name or document.get("schema")
    if not isinstance(selected_schema, str):
        return errors + ["$.schema: missing schema identifier"]
    errors.extend(
        _schema_errors(document, selected_schema, schema_directory=Path(schema_directory))
    )
    if selected_schema == "gum-school-curriculum-v1":
        errors.extend(_semantic_curriculum_errors(document))
    return sorted(set(errors))


def validate_curriculum(
    document: Any,
    *,
    schema_directory: Path | str = DEFAULT_SCHEMA_DIRECTORY,
) -> dict[str, Any]:
    """Validate a curriculum and return it, or raise with all discovered errors."""
    errors = validate_document(
        document,
        schema_name="gum-school-curriculum-v1",
        schema_directory=schema_directory,
    )
    if errors:
        raise CurriculumValidationError(errors)
    return document


def validate_curriculum_file(
    path: Path | str = DEFAULT_CURRICULUM,
    *,
    schema_directory: Path | str = DEFAULT_SCHEMA_DIRECTORY,
) -> dict[str, Any]:
    return validate_curriculum(load_json(path), schema_directory=schema_directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a GUM School JSON specification")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_CURRICULUM)
    parser.add_argument("--schema-directory", type=Path, default=DEFAULT_SCHEMA_DIRECTORY)
    arguments = parser.parse_args(argv)
    try:
        document = load_json(arguments.path)
        errors = validate_document(document, schema_directory=arguments.schema_directory)
    except CurriculumValidationError as error:
        errors = list(error.errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"VALID: {arguments.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
