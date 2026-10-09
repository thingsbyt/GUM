"""Official-engine contracts for recurrent causal composition."""
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from gum.lineage import file_sha256
from gum.school.composition_official import run_official_composition_promotion
from gum.school.cumulative import CAUSAL_FILENAME, CumulativeSchoolLearner


pytestmark = pytest.mark.neural


class _DeterministicSystemRandom:
    next_seed = 1_810_000_003

    def randrange(self, start: int, stop: int) -> int:
        value = self.__class__.next_seed
        self.__class__.next_seed += 10_007
        assert start <= value < stop
        return value


def test_replace_causal_policy_preserves_base_component(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    official = root / "evidence" / "gum-school" / "sealed-recurrent-official-v2"
    pointer = json.loads((official / "PROMOTED.json").read_text(encoding="utf-8"))
    source = official / "snapshots" / pointer["snapshot_id"] / "state"
    candidate = tmp_path / "candidate"
    import shutil
    shutil.copytree(source, candidate)
    before = CumulativeSchoolLearner.load_bundle(candidate)
    base_hash = file_sha256(candidate / "SCHOOL_LEARNER.json")
    policy = (
        root / "evidence" / "gum-school" / "research"
        / "causal-composition-reward-outcome-v2"
        / "COMPOSITION_REWARD_OUTCOME_POLICY.pt"
    )
    CumulativeSchoolLearner.replace_causal_policy(candidate, causal_path=policy)
    after = CumulativeSchoolLearner.load_bundle(candidate)
    assert file_sha256(candidate / "SCHOOL_LEARNER.json") == base_hash
    assert file_sha256(candidate / CAUSAL_FILENAME) == file_sha256(policy)
    assert after.base.status() == before.base.status()


def test_composition_candidate_passes_official_engine_protocol(
    tmp_path: Path, monkeypatch
):
    _DeterministicSystemRandom.next_seed = 1_810_000_003
    monkeypatch.setattr(
        "gum.school.composition_official.secrets.SystemRandom",
        _DeterministicSystemRandom,
    )
    monkeypatch.setattr(
        "gum.school.composition_official.secrets.token_hex",
        lambda count: "ab" * count,
    )
    monkeypatch.setattr("gum.school.composition_official._git_clean", lambda: True)
    monkeypatch.setattr(
        "gum.school.composition_official._git_head", lambda: "2" * 40
    )
    root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "official-composition"
    report = run_official_composition_promotion(
        workspace,
        runner_path=root / "scripts" / "run_school_composition_official_promotion.py",
        trials=32,
    )
    assert report["qualification"] == "pass"
    assert report["decision"]["outcome"] == "promote"
    assert report["candidate"]["successes"] >= 26
    assert report["decision"]["gates"]["all_passed"] is True
    assert report["next_lesson_id"] == "changing-maze.memory.001"
    assert report["retention"]["current_summary"]["successes"] >= 69
    assert report["source_freeze"]["git_head"] == "2" * 40
    assert set(report["seed_manifest"]["seeds"]).isdisjoint(
        report["source_freeze"]["excluded_prior_sealed_seeds"]
    )
    progress = json.loads((workspace / "PROGRESS.json").read_text(encoding="utf-8"))
    assert progress["promoted_lessons"][:4] == [
        "object-laboratory.occlusion.001",
        "object-laboratory.functional-category.002",
        "causal-workshop.controls.001",
        "causal-workshop.composition.002",
    ]


def test_saved_official_composition_promotion_passed_every_gate():
    root = Path(__file__).resolve().parents[1]
    workspace = (
        root / "evidence" / "gum-school"
        / "sealed-recurrent-composition-official-v1"
    )
    report = json.loads(
        (workspace / "OFFICIAL_COMPOSITION_PROMOTION.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["qualification"] == "pass"
    assert report["official_curriculum_run"] is True
    assert report["sealed_data_used_for_training"] is False
    assert report["candidate"]["successes"] == 64
    assert report["candidate_before_training"]["successes"] == 21
    assert report["matched_fresh"]["successes"] == 17
    assert report["uniform_random"]["successes"] == 17
    assert report["retention"]["current_summary"]["successes"] == 72
    assert report["decision"]["outcome"] == "promote"
    assert report["decision"]["gates"]["all_passed"] is True
    assert report["next_lesson_id"] == "changing-maze.memory.001"
    assert report["source_freeze"]["git_head"] == (
        "e5cd6719aff968a260ae99dac925479bdab4a615"
    )
    assert len(report["source_freeze"]["excluded_prior_sealed_seeds"]) == 288
    assert set(report["seed_manifest"]["seeds"]).isdisjoint(
        report["source_freeze"]["excluded_prior_sealed_seeds"]
    )
    progress = json.loads((workspace / "PROGRESS.json").read_text(encoding="utf-8"))
    assert progress["promoted_lessons"][:4] == [
        "object-laboratory.occlusion.001",
        "object-laboratory.functional-category.002",
        "causal-workshop.controls.001",
        "causal-workshop.composition.002",
    ]
