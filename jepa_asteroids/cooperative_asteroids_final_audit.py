"""Final sealed replication after resolving long-wrap friendly fire."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .cooperative_asteroids import run_audit


FROZEN_SHA = "0E43CAA81B2A1BDF139E58238C28D7E67A2AEFE2BE7F0F9F587542D264159947"
SEALED_SEED = 2_330_001


def _hash():
    return hashlib.sha256(Path(__file__).with_name("cooperative_asteroids.py").read_bytes()).hexdigest().upper()


def run_final_audit(output, *, worlds=8, max_steps=240):
    before = _hash()
    if before != FROZEN_SHA: raise RuntimeError(f"frozen final implementation mismatch: {before}")
    report = run_audit(output, worlds=worlds, seed=SEALED_SEED, max_steps=max_steps)
    after = _hash(); aggregate = report["aggregate"]
    both_contributed = sum(all(value > 0 for value in row["coordinated"]["hits_by_agent"])
                           for row in report["per_world"])
    team_wins = sum(row["coordinated"]["return"] > row["uncommunicative"]["return"]
                    for row in report["per_world"])
    co, mute = aggregate["coordinated"], aggregate["uncommunicative"]
    passed = (before == after == FROZEN_SHA and report["controls_grounded_worlds"] == worlds
              and co["mean_team_return"] > mute["mean_team_return"]
              and co["mean_team_return"] > aggregate["solo"]["mean_team_return"]
              and co["mean_team_return"] > aggregate["random"]["mean_team_return"]
              and co["total_hits"] >= .85 * mute["total_hits"]
              and co["both_survive"] == worlds and co["friendly_fire"] == 0
              and both_contributed == worlds and team_wins >= worlds // 2)
    report["format"] = "wailah-two-rocket-asteroids-v20-final-frozen-audit-v1"
    report["classification"] = "fourth precommit after three disclosed failed seals; new untouched seeds"
    report["team_objective"] = "hits - 10*friendly_fire - 20*life_loss + survival ticks"
    report["cooperation_decision"] = {"necessary_for_raw_shooting": False,
        "useful_for_shared_safe_objective": co["mean_team_return"] > mute["mean_team_return"],
        "basis": "matched communicating versus message-disabled pair"}
    report["both_rockets_contributed_worlds"] = both_contributed; report["team_utility_wins"] = team_wins
    report["frozen_implementation"] = {"expected": FROZEN_SHA, "before": before, "after": after,
                                        "unchanged": before == after}
    report["precommitted_criteria"] = ["all controls visually grounded", "coordinated return beats all baselines",
        "at least 85 percent of uncommunicative hit throughput", "both rockets score and stay active in every world",
        "zero coordinated friendly fire", "at least half of matched team-utility comparisons won", "hash unchanged"]
    report["precommitted_pass"] = bool(passed)
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8"); return report


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=8); parser.add_argument("--max-steps", type=int, default=240)
    args = parser.parse_args(argv); report = run_final_audit(args.output, worlds=args.worlds, max_steps=args.max_steps)
    print(json.dumps({"aggregate": report["aggregate"], "controls_grounded_worlds": report["controls_grounded_worlds"],
        "both_rockets_contributed_worlds": report["both_rockets_contributed_worlds"],
        "team_utility_wins": report["team_utility_wins"], "cooperation_decision": report["cooperation_decision"],
        "frozen_unchanged": report["frozen_implementation"]["unchanged"],
        "precommitted_pass": report["precommitted_pass"]}, indent=2))


if __name__ == "__main__": raise SystemExit(main())
