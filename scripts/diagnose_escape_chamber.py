"""Measure preserved Quattro Forti experience without changing the learners."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gum.school.escape_diagnostics import diagnose_team, probe_visual_mechanism
from gum.storage import atomic_write_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("team_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-configurations", type=int, default=48)
    args = parser.parse_args(argv)
    report = diagnose_team(args.team_root)
    report["perception_probe"] = probe_visual_mechanism(
        args.team_root, configurations=args.probe_configurations
    )
    if args.output:
        atomic_write_json(args.output, report, backup=False, sort_keys=True)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
