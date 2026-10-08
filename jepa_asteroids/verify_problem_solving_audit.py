"""Independent arithmetic/transition verifier for problem-solving audits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .semantic_world_model import semantic_event


def verify(path: Path) -> dict:
    path = Path(path); raw = path.read_bytes(); report = json.loads(raw)
    if report.get("format") != "wailah-compositional-semantic-mind-audit-v1":
        raise ValueError("wrong audit format")
    horizon = int(report["protocol"]["horizon"]); transitions = successes = 0
    conditions = {}
    for name, aggregate in report["aggregate"].items():
        verified_successes = 0
        for row in aggregate["rows"]:
            target = row["target_events_audit_only"]; phase = 0
            trace = row["trace"]
            if int(row["steps"]) != len(trace) or len(trace) > horizon:
                raise AssertionError(f"invalid trace length in {name}")
            for index, step in enumerate(trace, 1):
                if int(step["step"]) != index:
                    raise AssertionError(f"nonconsecutive trace in {name}")
                event = semantic_event(step["observed_event"])
                if event == target[phase]: phase += 1
                else: phase = 1 if event == target[0] else 0
                terminal = phase == len(target)
                if float(step["reward"]) != float(terminal):
                    raise AssertionError(f"reward does not match solved state in {name}")
                if terminal and index != len(trace):
                    raise AssertionError(f"trace continues after solution in {name}")
            solved = phase == len(target)
            if bool(row["success"]) != solved:
                raise AssertionError(f"success claim does not match trace in {name}")
            verified_successes += int(solved); transitions += len(trace)
        if verified_successes != int(aggregate["successes"]):
            raise AssertionError(f"aggregate mismatch in {name}")
        conditions[name] = verified_successes; successes += verified_successes
    return {"verified": True, "audit_sha256": hashlib.sha256(raw).hexdigest(),
            "conditions": conditions, "verified_transitions": transitions,
            "verified_successes": successes}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("audit", type=Path)
    args = parser.parse_args(argv); print(json.dumps(verify(args.audit), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
