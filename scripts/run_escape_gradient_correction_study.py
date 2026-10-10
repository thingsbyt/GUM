"""Run the locked four-team cooperative gradient-correction study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gum.school.escape_gradient_study import run_correction_study
from gum.storage import atomic_write_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    report = run_correction_study(
        repository,
        args.output_root,
        device=args.device,
        progress=lambda message: print(message, flush=True),
    )
    target = args.output_root / "STUDY_RESULTS.json"
    atomic_write_json(target, report, backup=False, sort_keys=True)
    print(json.dumps({
        "report": str(target.resolve()),
        "aggregate": report["aggregate"],
        "mechanism_check": report["mechanism_check"],
        "held_out_check": report["held_out_check"],
        "decision": report["decision"],
        "wall_time_seconds": report["wall_time_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
