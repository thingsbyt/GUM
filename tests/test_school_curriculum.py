from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from gum.school.validation import (
    DEFAULT_CURRICULUM,
    DEFAULT_SCHEMA_DIRECTORY,
    SCHEMA_FILES,
    load_json,
    validate_curriculum,
    validate_curriculum_file,
    validate_document,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "gum_school"


def _apply_fixture(fixture_path: Path) -> tuple[dict, str]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    document = deepcopy(load_json(REPOSITORY_ROOT / fixture["base"]))
    for operation in fixture["operations"]:
        target = document
        for part in operation["path"][:-1]:
            target = target[part]
        final = operation["path"][-1]
        if operation["operation"] in {"add", "replace"}:
            target[final] = operation["value"]
        else:
            raise AssertionError(f"unsupported fixture operation {operation['operation']!r}")
    return document, fixture["expected_error"]


def _keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _keys(child)


def _object_schemas(value):
    if isinstance(value, dict):
        if value.get("type") == "object" and "properties" in value:
            yield value
        for child in value.values():
            yield from _object_schemas(child)
    elif isinstance(value, list):
        for child in value:
            yield from _object_schemas(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys(child)


def test_all_school_schemas_are_valid_draft_2020_12_documents():
    for filename in SCHEMA_FILES.values():
        schema = load_json(DEFAULT_SCHEMA_DIRECTORY / filename)
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False
        assert all(row.get("additionalProperties") is False for row in _object_schemas(schema))


def test_official_curriculum_passes_structural_and_semantic_validation():
    curriculum = validate_curriculum_file()
    assert curriculum["curriculum_id"] == "gum-school-v1"
    assert curriculum["phase"] == 2
    assert curriculum["status"] == "adapters-admitted-not-trained"


def test_curriculum_is_an_authored_foundational_sequence_not_random_tasks():
    curriculum = validate_curriculum_file()
    assert {lesson["school"] for lesson in curriculum["lessons"]} == {
        "object-laboratory",
        "causal-workshop",
        "changing-maze",
    }
    assert [lesson["sequence"] for lesson in curriculum["lessons"]] == list(
        range(1, len(curriculum["lessons"]) + 1)
    )
    for lesson in curriculum["lessons"]:
        assert lesson["training"]["generator"].startswith("authored-")
        assert "random" not in lesson["authored_family"]
        assert len(lesson["rationale"]) >= 40


def test_local_language_model_is_optional_disabled_and_untrusted():
    policy = validate_curriculum_file()["policies"]["local_llm"]
    assert policy == {
        "optional": True,
        "enabled_by_default": False,
        "trusted": False,
        "may_execute_code": False,
        "may_grade": False,
        "may_view_sealed_data": False,
    }


def test_prerequisites_retention_and_transfer_design_are_machine_enforced():
    curriculum = validate_curriculum_file()
    lessons = sorted(curriculum["lessons"], key=lambda row: row["sequence"])
    assert lessons[0]["prerequisites"] == []
    assert lessons[0]["retention"] == {
        "scope": "all-promoted-skills",
        "protected_skills": [],
        "evaluation_seeds": [11201, 11202, 11203, 11204],
        "trials": 24,
        "maximum_regression": 0.05,
    }
    assert all(lesson["prerequisites"] for lesson in lessons[1:])
    matrix = curriculum["transfer_matrix"]
    assert matrix["design"] == "isolated-source-branches"
    assert matrix["branch_origin"] == "pre-curriculum-promoted-snapshot"
    assert len(matrix["source_schools"]) * len(matrix["targets"]) == 9
    assert matrix["target_evaluation_training"] is False
    assert curriculum["policies"]["budgets"]["status"] == "frozen-after-adapter-admission"
    declared_trials = sum(
        lesson["baseline"]["trials"]
        + lesson["evaluation"]["trials"]
        + lesson["evaluation"]["new_seed_trials"]
        + lesson["retention"]["trials"]
        for lesson in lessons
    ) + len(matrix["source_schools"]) * len(matrix["targets"]) * matrix["trials_per_cell"]
    assert declared_trials == 864
    assert declared_trials <= curriculum["policies"]["budgets"]["maximum_evaluation_trials"]


def test_public_curriculum_contains_commitments_not_sealed_seed_lists_or_answers():
    curriculum = validate_curriculum_file()
    normalized_keys = {key.lower().replace("-", "_") for key in _keys(curriculum)}
    assert "sealed_seeds" not in normalized_keys
    assert "answer_key" not in normalized_keys
    assert "solution" not in normalized_keys
    for examination in curriculum["examinations"]:
        assert examination["selection"]["status"] == "selection-protocol-committed"
        assert examination["selection"]["commitment_scope"] == "selection-protocol"


def test_worlds_are_admitted_only_with_phase_two_evidence():
    curriculum = validate_curriculum_file()
    for admission in curriculum["world_admissions"]:
        assert admission["status"] == "admitted"
        assert admission["evidence_hashes"]
        assert set(admission["checks"].values()) == {"passed", "nontrivial"}


@pytest.mark.parametrize("fixture_path", sorted(FIXTURE_DIRECTORY.glob("*.json")), ids=lambda path: path.stem)
def test_adversarial_curriculum_fixtures_are_rejected(fixture_path: Path):
    document, expected_error = _apply_fixture(fixture_path)
    errors = validate_document(document)
    assert errors, fixture_path.name
    assert any(expected_error in error for error in errors), "\n".join(errors)


def test_duplicate_json_fields_are_rejected(tmp_path: Path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"gum-school-curriculum-v1","schema":"duplicate"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON field"):
        load_json(path)


def test_malformed_json_fixture_is_rejected_before_schema_validation():
    with pytest.raises(ValueError, match="malformed JSON"):
        load_json(FIXTURE_DIRECTORY / "malformed-json.txt")


def test_cross_family_exam_is_allowed_only_with_a_primary_exam():
    curriculum = validate_curriculum_file()
    final_lesson = curriculum["lessons"][-1]
    assert len(final_lesson["evaluation"]["examination_ids"]) == 2
    changed = deepcopy(curriculum)
    changed["lessons"][-1]["evaluation"]["examination_ids"] = [
        "exam.causal-workshop.transfer.v1"
    ]
    errors = validate_document(changed)
    assert any("no examination for its own adapter and school" in error for error in errors)


def test_direct_lesson_and_admission_documents_validate_against_their_schemas():
    curriculum = validate_curriculum_file()
    assert validate_document(curriculum["lessons"][0]) == []
    assert validate_document(curriculum["examinations"][0]) == []
    assert validate_document(curriculum["world_admissions"][0]) == []


def test_validate_curriculum_returns_the_original_document():
    curriculum = load_json(DEFAULT_CURRICULUM)
    assert validate_curriculum(curriculum) is curriculum
