"""External A→B→A continual-learning audit on independently authored worlds."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys

import numpy as np

from .arc_agi3_learning_audit import _quiet_logger, _trial
from .universal_explorer import UniversalExplorer


def _phase(arcade, game: str, seed: int, attempts: int, agent: UniversalExplorer,
           max_actions: int) -> list[dict]:
    return [_trial(arcade, game, seed + index, agent, max_actions,
                   random.Random(seed + index + 71)) for index in range(attempts)]


def _summary(rows: list[dict], max_actions: int) -> dict:
    successes = [row for row in rows if row["levels_completed"] > 0]
    costs = [row["progress_steps"][0] if row["progress_steps"] else max_actions for row in rows]
    return {"attempts": len(rows), "successful_attempts": len(successes),
            "levels": sum(row["levels_completed"] for row in rows),
            "success_rate": len(successes) / max(1, len(rows)),
            "mean_censored_steps_to_progress": float(np.mean(costs)),
            "rows": rows}


def run(project: Path, output: Path, a_attempts: int = 6, b_attempts: int = 20,
        max_actions: int = 512) -> dict:
    os.environ.setdefault("MPLCONFIGDIR", str(project / ".mplconfig"))
    sys.path.insert(0, str(project / ".arc_agi_deps"))
    from arc_agi import Arcade
    arcade = Arcade(environments_dir=str(project / "arc_environments"),
                    recordings_dir=str(project / "arc_recordings"), logger=_quiet_logger())
    system = random.SystemRandom(); a_seed = system.randrange(1, 2**30)
    b_seed = system.randrange(2**30, 2**31 - b_attempts - 2)
    revisit_seed = system.randrange(1, 2**30)
    shared = UniversalExplorer(seed=202_610_05)
    a_train_rows = _phase(arcade, "ls20", a_seed, a_attempts, shared, max_actions)
    b_transfer_rows = _phase(arcade, "sp80", b_seed, b_attempts, shared, max_actions)
    scratch = UniversalExplorer(seed=202_610_05)
    b_scratch_rows = _phase(arcade, "sp80", b_seed, b_attempts, scratch, max_actions)
    a_revisit_rows = _phase(arcade, "ls20", revisit_seed, a_attempts, shared, max_actions)
    a_train = _summary(a_train_rows, max_actions); b_transfer = _summary(b_transfer_rows, max_actions)
    b_scratch = _summary(b_scratch_rows, max_actions); a_revisit = _summary(a_revisit_rows, max_actions)
    report = {
        "format": "wailah-arc-agi3-lifelong-transfer-v1",
        "protocol": {"sequence": "ls20(A) -> sp80(B) -> ls20(A)",
                     "worlds": "independently authored public ARC-AGI-3 environments",
                     "agent_input": "pixels, available controls, progress delta, termination",
                     "withheld": ["game identity", "instructions", "goal", "source code", "human solutions"],
                     "shared_agent_has_no_game_identifier": True,
                     "scratch_B_uses_identical_seeds": True,
                     "architecture_frozen_during_recorded_run": True,
                     "evaluation_status": "development benchmark; not untouched heldout confirmation",
                     "worlds_previously_used_for_architecture_debugging": True},
        "seed_commitments": {"a_train": hashlib.sha256(str(a_seed).encode()).hexdigest(),
                             "b": hashlib.sha256(str(b_seed).encode()).hexdigest(),
                             "a_revisit": hashlib.sha256(str(revisit_seed).encode()).hexdigest()},
        "a_train": a_train, "b_transfer": b_transfer, "b_scratch": b_scratch,
        "a_revisit": a_revisit,
        "transfer": {"success_rate_delta": b_transfer["success_rate"] - b_scratch["success_rate"],
                     "censored_step_reduction": (b_scratch["mean_censored_steps_to_progress"] -
                                                  b_transfer["mean_censored_steps_to_progress"])},
        "retention": {"success_rate_delta": a_revisit["success_rate"] - a_train["success_rate"],
                      "retained": a_revisit["success_rate"] >= .95 * a_train["success_rate"]},
        "final_brain": shared.status(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--a-attempts", type=int, default=6)
    parser.add_argument("--b-attempts", type=int, default=20); parser.add_argument("--max-actions", type=int, default=512)
    args = parser.parse_args(argv); report = run(args.project.resolve(), args.output.resolve(),
                                                  args.a_attempts, args.b_attempts, args.max_actions)
    compact = {name: {key: value for key, value in report[name].items() if key != "rows"}
               for name in ("a_train", "b_transfer", "b_scratch", "a_revisit")}
    compact["transfer"] = report["transfer"]; compact["retention"] = report["retention"]
    print(json.dumps(compact, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
