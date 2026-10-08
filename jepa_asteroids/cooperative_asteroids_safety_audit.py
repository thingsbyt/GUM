"""Second precommit after the first audit exposed a friendly-fire regression."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .cooperative_asteroids import run_audit


FROZEN_SHA = "E54676E09B9AD66F0F794F673AE376D9652AD339965603F8CDB9D9C3F1660E60"
SEALED_SEED = 1_880_001


def _hash():
    return hashlib.sha256(Path(__file__).with_name("cooperative_asteroids.py").read_bytes()).hexdigest().upper()


def run_frozen_safety_audit(output, *, worlds=8, max_steps=240):
    before = _hash()
    if before != FROZEN_SHA: raise RuntimeError(f"frozen safety implementation mismatch: {before}")
    report = run_audit(output, worlds=worlds, seed=SEALED_SEED, max_steps=max_steps)
    after = _hash(); aggregate = report["aggregate"]
    both_contributed = sum(all(value > 0 for value in row["coordinated"]["hits_by_agent"])
                           for row in report["per_world"])
    coordinated_wins = sum(row["coordinated"]["hits"] > row["uncommunicative"]["hits"]
                           for row in report["per_world"])
    passed = (before == after == FROZEN_SHA and report["controls_grounded_worlds"] == worlds
              and aggregate["coordinated"]["total_hits"] > aggregate["uncommunicative"]["total_hits"]
              and aggregate["coordinated"]["total_hits"] > aggregate["solo"]["total_hits"]
              and aggregate["coordinated"]["total_hits"] > aggregate["random"]["total_hits"]
              and aggregate["coordinated"]["both_survive"] == worlds
              and aggregate["coordinated"]["friendly_fire"] == 0
              and both_contributed == worlds and coordinated_wins >= worlds // 2)
    report["format"] = "wailah-two-rocket-asteroids-v20-safety-frozen-audit-v1"
    report["classification"] = "second precommit after disclosed safety failure; new untouched seeds"
    report["both_rockets_contributed_worlds"] = both_contributed
    report["coordinated_hit_wins"] = coordinated_wins
    report["frozen_implementation"] = {"expected": FROZEN_SHA, "before": before, "after": after,
                                        "unchanged": before == after}
    report["precommitted_criteria"] = ["all anonymous controls visually grounded",
        "coordinated total hits exceed all three matched baselines", "both rockets score and survive every world",
        "zero coordinated friendly-fire incidents", "coordination wins at least half of per-world hit comparisons",
        "implementation hash unchanged"]
    report["precommitted_pass"] = bool(passed)
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8"); return report


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=8); parser.add_argument("--max-steps", type=int, default=240)
    args = parser.parse_args(argv); report = run_frozen_safety_audit(args.output, worlds=args.worlds, max_steps=args.max_steps)
    print(json.dumps({"aggregate": report["aggregate"], "controls_grounded_worlds": report["controls_grounded_worlds"],
        "both_rockets_contributed_worlds": report["both_rockets_contributed_worlds"],
        "coordinated_hit_wins": report["coordinated_hit_wins"],
        "frozen_unchanged": report["frozen_implementation"]["unchanged"],
        "precommitted_pass": report["precommitted_pass"]}, indent=2))


if __name__ == "__main__": raise SystemExit(main())
