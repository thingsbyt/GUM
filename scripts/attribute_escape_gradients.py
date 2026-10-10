"""Reproduce and decompose one cooperative-escape policy update."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gum.school.escape_gradient_attribution import attribute_successful_episode
from gum.storage import atomic_write_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("team_root", type=Path)
    parser.add_argument("episode_id")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    print("Reproducing the original update before attribution...", flush=True)
    report = attribute_successful_episode(
        args.team_root,
        args.episode_id,
        args.output_root,
        device=args.device,
    )
    target = args.output_root / "ATTRIBUTION.json"
    atomic_write_json(target, report, backup=False, sort_keys=True)
    contexts = [
        {
            "member": member["name"],
            **context,
        }
        for member in report["members"]
        for context in member["diagnostic_contexts"]
    ]
    print(json.dumps({
        "report": str(target.resolve()),
        "source_commit": report["source_commit"],
        "reproduction_passed": report["reproduction_gate"]["passed"],
        "diagnostic_contexts": contexts,
        "wall_time_seconds": report["wall_time_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
