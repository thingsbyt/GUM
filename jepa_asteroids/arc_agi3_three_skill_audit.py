"""Frozen A→B→C→A/B/C continual-learning audit on external ARC worlds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

from .arc_agi3_learning_audit import _quiet_logger, _trial
from .universal_explorer import UniversalExplorer


AGENT_SEED = 202_610_05
SEEDS = {"ls20": 310_000_000, "sp80": 320_000_000, "ft09": 330_000_000,
         "ls20_revisit": 340_000_000, "sp80_revisit": 350_000_000,
         "ft09_revisit": 360_000_000}


def _phase(arcade, game, seed, attempts, agent, max_actions):
    return [_trial(arcade, game, seed + i, agent, max_actions, random.Random(seed + i + 71))
            for i in range(attempts)]


def _summary(rows, max_actions):
    costs = [row["progress_steps"][0] if row["progress_steps"] else max_actions for row in rows]
    return {"attempts": len(rows), "successes": sum(row["levels_completed"] > 0 for row in rows),
            "levels": sum(row["levels_completed"] for row in rows),
            "wins": sum(bool(row["won"]) for row in rows),
            "mean_censored_steps": float(np.mean(costs)), "rows": rows}


def run(project: Path, output: Path, max_actions: int = 512) -> dict:
    os.environ.setdefault("MPLCONFIGDIR", str(project / ".mplconfig"))
    sys.path.insert(0, str(project / ".arc_agi_deps"))
    from arc_agi import Arcade
    arcade = Arcade(environments_dir=str(project / "arc_environments"),
                    recordings_dir=str(project / "arc_recordings"), logger=_quiet_logger())
    shared = UniversalExplorer(AGENT_SEED)
    a = _phase(arcade, "ls20", SEEDS["ls20"], 6, shared, max_actions)
    b = _phase(arcade, "sp80", SEEDS["sp80"], 20, shared, max_actions)
    c = _phase(arcade, "ft09", SEEDS["ft09"], 20, shared, max_actions)
    b_scratch = _phase(arcade, "sp80", SEEDS["sp80"], 20,
                       UniversalExplorer(AGENT_SEED), max_actions)
    c_scratch = _phase(arcade, "ft09", SEEDS["ft09"], 20,
                       UniversalExplorer(AGENT_SEED), max_actions)
    ar = _phase(arcade, "ls20", SEEDS["ls20_revisit"], 4, shared, max_actions)
    br = _phase(arcade, "sp80", SEEDS["sp80_revisit"], 4, shared, max_actions)
    cr = _phase(arcade, "ft09", SEEDS["ft09_revisit"], 4, shared, max_actions)
    names = {"a_ls20": a, "b_sp80": b, "c_ft09": c, "b_sp80_scratch": b_scratch,
             "c_ft09_scratch": c_scratch, "a_revisit": ar, "b_revisit": br, "c_revisit": cr}
    hashes = {name: hashlib.sha256((project / "jepa_asteroids" / name).read_bytes()).hexdigest()
              for name in ("universal_explorer.py", "causal_discovery.py", "hypothesis_engine.py",
                           "concept_memory.py", "role_schema.py", "visual_goal_planner.py")}
    report = {"format": "wailah-arc3-three-skill-lifelong-v1",
              "protocol": {"sequence": "ls20 -> sp80 -> ft09 -> scratch controls -> revisit all",
                           "agent_seed": AGENT_SEED, "phase_seeds": SEEDS,
                           "same_B_C_seeds_for_shared_and_scratch": True,
                           "agent_input": "pixels, available controls, progress delta, termination",
                           "withheld": ["game identity", "instructions", "goal", "human solutions"],
                           "status": "development benchmark, frozen during recorded run",
                           "source_sha256": hashes},
              "phases": {name: _summary(rows, max_actions) for name, rows in names.items()},
              "final_brain": shared.status()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    shared.save(output.with_name("WAILAH_THREE_SKILL_BRAIN.json"))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--max-actions", type=int, default=512)
    args = parser.parse_args(argv); report = run(args.project.resolve(), args.output.resolve(), args.max_actions)
    print(json.dumps({name: {key: value for key, value in row.items() if key != "rows"}
                      for name, row in report["phases"].items()}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
