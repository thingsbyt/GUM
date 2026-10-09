"""Official-engine integration for the recurrent causal specialist."""
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from gum.school.recurrent_official import run_official_recurrent_promotion


pytestmark = pytest.mark.neural


class _DeterministicSystemRandom:
    next_seed = 1_700_000_003

    def randrange(self, start: int, stop: int) -> int:
        value = self.__class__.next_seed
        self.__class__.next_seed += 7_919
        assert start <= value < stop
        return value


def test_recurrent_candidate_is_promoted_by_official_engine(tmp_path: Path, monkeypatch):
    _DeterministicSystemRandom.next_seed = 1_700_000_003
    monkeypatch.setattr(
        "gum.school.recurrent_official.secrets.SystemRandom",
        _DeterministicSystemRandom,
    )
    monkeypatch.setattr(
        "gum.school.recurrent_official.secrets.token_hex", lambda count: "ef" * count
    )
    monkeypatch.setattr("gum.school.recurrent_official._git_clean", lambda: True)
    monkeypatch.setattr("gum.school.recurrent_official._git_head", lambda: "1" * 40)
    root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "official-recurrent"
    report = run_official_recurrent_promotion(
        workspace,
        runner_path=root / "scripts" / "run_school_recurrent_official_promotion.py",
        trials=32,
    )
    assert report["qualification"] == "pass"
    assert report["official_curriculum_run"] is True
    assert report["decision"]["outcome"] == "promote"
    assert report["candidate"]["successes"] == 32
    assert report["uncertainty_calibration"]["initial_uncertainty_rate"] == 1.0
    assert (
        report["uncertainty_calibration"][
            "post_causal_evidence_uncertainty_rate"
        ]
        == 0.0
    )
    assert report["retention"]["current_summary"]["successes"] == 48
    assert report["next_lesson_id"] == "causal-workshop.composition.002"
    school = json.loads((workspace / "SCHOOL_REPORT.json").read_text(encoding="utf-8"))
    assert school["inherited_lessons"] == [
        "object-laboratory.occlusion.001",
        "object-laboratory.functional-category.002",
    ]
    assert school["promoted_lessons"][:3] == [
        "object-laboratory.occlusion.001",
        "object-laboratory.functional-category.002",
        "causal-workshop.controls.001",
    ]
    run = workspace / "runs" / report["run_id"]
    evaluation = json.loads((run / "EVALUATION.json").read_text(encoding="utf-8"))
    assert evaluation["format"] == "gum-school-evaluation-v2"
    assert evaluation["sealed"]["uncertainty_rate"] == 0.0
    assert set(report["seed_manifest"]["seeds"]).isdisjoint(
        report["source_freeze"]["excluded_prior_sealed_seeds"]
    )


def test_saved_official_recurrent_promotion_passed_all_gates():
    root = Path(__file__).resolve().parents[1]
    workspace = root / "evidence" / "gum-school" / "sealed-recurrent-official-v2"
    report = json.loads(
        (workspace / "OFFICIAL_RECURRENT_PROMOTION.json").read_text(encoding="utf-8")
    )
    assert report["qualification"] == "pass"
    assert report["official_curriculum_run"] is True
    assert report["candidate"]["successes"] == 61
    assert report["candidate"]["trials"] == 64
    assert report["candidate_before_training"]["successes"] == 8
    assert report["matched_fresh"]["successes"] == 7
    assert report["uniform_random"]["successes"] == 4
    assert report["retention"]["current_summary"]["successes"] == 48
    assert report["decision"]["outcome"] == "promote"
    assert report["decision"]["gates"]["all_passed"] is True
    assert report["next_lesson_id"] == "causal-workshop.composition.002"
    assert report["source_freeze"]["git_head"] == (
        "5da84f02f1e4c9a1f283d9ba664cd65152d8f4b7"
    )
    assert len(report["source_freeze"]["excluded_prior_sealed_seeds"]) == 224
    assert set(report["seed_manifest"]["seeds"]).isdisjoint(
        report["source_freeze"]["excluded_prior_sealed_seeds"]
    )
    progress = json.loads((workspace / "PROGRESS.json").read_text(encoding="utf-8"))
    assert progress["promoted_lessons"][:3] == [
        "object-laboratory.occlusion.001",
        "object-laboratory.functional-category.002",
        "causal-workshop.controls.001",
    ]
