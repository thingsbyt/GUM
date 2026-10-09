"""Run the frozen recurrent composition policy through the School engine."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gum.school.composition_official import run_official_composition_promotion


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = (
    ROOT / "evidence" / "gum-school" / "sealed-recurrent-composition-official-v1"
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--trials", type=int, default=64)
    args = parser.parse_args(argv)
    report = run_official_composition_promotion(
        args.workspace,
        runner_path=Path(__file__).resolve(),
        trials=args.trials,
    )
    print(json.dumps({
        "qualification": report["qualification"],
        "lesson_id": report["lesson_id"],
        "candidate_success_rate": report["candidate"]["success_rate"],
        "candidate_before_training_success_rate": report[
            "candidate_before_training"
        ]["success_rate"],
        "matched_fresh_success_rate": report["matched_fresh"]["success_rate"],
        "initial_uncertainty_rate": report["uncertainty_calibration"][
            "initial_uncertainty_rate"
        ],
        "post_causal_uncertainty_rate": report["uncertainty_calibration"][
            "post_causal_evidence_uncertainty_rate"
        ],
        "retention_success_rate": report["retention"]["current_summary"][
            "success_rate"
        ],
        "next_lesson_id": report["next_lesson_id"],
        "report": str(
            Path(args.workspace).resolve() / "OFFICIAL_COMPOSITION_PROMOTION.json"
        ),
    }, indent=2, sort_keys=True))
    return 0 if report["qualification"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
