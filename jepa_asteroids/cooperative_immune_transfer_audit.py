"""Precommitted frozen replication for autonomous two-agent cooperation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .cooperative_immune_savior import run_cooperative_audit


FROZEN_SHA = "E78EA65A95303CDA93107444ADC1E1F3E50B4C57F5A3323CE5CC142AEEAAD35E"
SEALED_SEED = 1_310_003


def _hash():
    return hashlib.sha256(Path(__file__).with_name("cooperative_immune_savior.py").read_bytes()).hexdigest().upper()


def run_frozen_cooperative_audit(output, worlds=12, budget=1500):
    before = _hash()
    if before != FROZEN_SHA: raise RuntimeError(f"frozen cooperative implementation mismatch: {before}")
    report = run_cooperative_audit(output, worlds=worlds, seed=SEALED_SEED, budget=budget)
    after = _hash(); decisions = [row["team"].get("cooperation_decision") for row in report["per_world"]]
    decision_count = sum(bool(row and row.get("necessary")) for row in decisions)
    aggregate = report["aggregate"]
    passed = (aggregate["communicating_pair"]["successes"] == worlds
              and aggregate["communicating_pair"]["zero_damage_worlds"] == worlds
              and aggregate["same_agents_cooperation_disabled"]["successes"] == 0
              and aggregate["random_pair"]["successes"] == 0
              and decision_count == worlds and before == after == FROZEN_SHA)
    report["format"] = "wailah-cooperative-immune-v19-frozen-audit-v1"
    report["classification"] = "precommitted frozen two-agent cooperation decision and execution"
    report["cooperation_decisions"] = {"necessary_decisions": decision_count,
        "worlds": worlds, "all_evidence_based": decision_count == worlds, "rows": decisions}
    report["frozen_implementation"] = {"expected": FROZEN_SHA, "before": before,
                                        "after": after, "unchanged": before == after}
    report["precommitted_pass"] = passed
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8"); return report


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv); report = run_frozen_cooperative_audit(args.output)
    print(json.dumps({"aggregate": report["aggregate"],
        "cooperation_decisions": {k: v for k, v in report["cooperation_decisions"].items() if k != "rows"},
        "frozen_unchanged": report["frozen_implementation"]["unchanged"],
        "pass": report["precommitted_pass"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
