from __future__ import annotations

import json
from pathlib import Path


def test_saved_strict_causal_ablation_preserves_negative_result():
    root = Path(__file__).resolve().parents[1]
    path = (
        root / "evidence" / "gum-school" / "strict"
        / "causal-controls-unscaffolded-v1" / "STRICT_CAUSAL_ABLATION.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["development_only"] is True
    assert report["sealed_data_used"] is False
    assert report["strict_pass"] is False
    assert report["scaffolded_promoted_control"]["successes"] == 64
    assert report["promoted_snapshot_without_scaffold"]["successes"] == 4
    trained = report["full_budget_training_without_scaffold"]
    assert trained["training"]["interactions"] == 12_000
    assert trained["development"]["successes"] == 10
    assert trained["development"]["success_interval"]["upper"] < 0.8
    assert report["interpretation"] == (
        "The unscaffolded learner did not discover a general causal intervention policy."
    )
