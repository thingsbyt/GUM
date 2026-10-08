"""Frozen full-local-suite ARC-AGI-3 evaluation with lifelong, scratch, and random conditions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

from .arc_agi3_learning_audit import _quiet_logger, _trial
from .universal_explorer import UniversalExplorer


GAMES = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
         "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
         "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"]
ROOT_SEED = 202_610_050
AGENT_SEED = 202_610_05


def _seed(game_index: int, attempt: int, revisit: bool = False) -> int:
    return ROOT_SEED + game_index * 10_000 + (5_000 if revisit else 0) + attempt


def _summary(rows: list[dict]) -> dict:
    fractions = [row["levels_completed"] / max(1, row["win_levels"]) for row in rows]
    return {"trials": len(rows),
            "levels_completed": int(sum(row["levels_completed"] for row in rows)),
            "any_progress_trials": int(sum(row["levels_completed"] > 0 for row in rows)),
            "any_progress_rate": float(np.mean([row["levels_completed"] > 0 for row in rows])),
            "complete_wins": int(sum(bool(row["won"]) for row in rows)),
            "complete_win_rate": float(np.mean([bool(row["won"]) for row in rows])),
            "normalized_level_score": float(np.mean(fractions)),
            "mean_actions": float(np.mean([row["actions"] for row in rows]))}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(project: Path, output: Path, attempts: int = 6, revisits: int = 2,
        max_actions: int = 512) -> dict:
    os.environ.setdefault("MPLCONFIGDIR", str(project / ".mplconfig"))
    sys.path.insert(0, str(project / ".arc_agi_deps"))
    from arc_agi import Arcade
    arcade = Arcade(environments_dir=str(project / "arc_environments"),
                    recordings_dir=str(project / "arc_recordings"), logger=_quiet_logger())
    hashes = {name: _hash(project / "jepa_asteroids" / name) for name in
              ("universal_explorer.py", "hypothesis_engine.py", "concept_memory.py", "role_schema.py")}
    report = {"format": "wailah-arc-agi3-frozen-full-local-suite-v1",
              "protocol": {"games": GAMES, "attempts_per_game_per_condition": attempts,
                           "revisits_per_game": revisits, "max_actions": max_actions,
                           "root_seed": ROOT_SEED, "agent_seed": AGENT_SEED,
                           "conditions": ["one_lifelong_agent", "fresh_agent_per_game", "random"],
                           "same_world_seeds_across_conditions": True,
                           "agent_input": "pixels, available controls, progress delta, termination",
                           "withheld": ["game identity", "instructions", "goal", "human solutions"],
                           "evaluation_status": "frozen full local public suite; worlds are not untouched",
                           "agent_source_sha256": hashes},
              "started_unix": time.time(), "worlds": [], "complete": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lifelong = UniversalExplorer(seed=AGENT_SEED)
    for game_index, game in enumerate(GAMES):
        shared_rows = [_trial(arcade, game, _seed(game_index, i), lifelong, max_actions,
                              random.Random(_seed(game_index, i) + 71)) for i in range(attempts)]
        scratch = UniversalExplorer(seed=AGENT_SEED)
        scratch_rows = [_trial(arcade, game, _seed(game_index, i), scratch, max_actions,
                               random.Random(_seed(game_index, i) + 71)) for i in range(attempts)]
        random_rows = [_trial(arcade, game, _seed(game_index, i), None, max_actions,
                              random.Random(_seed(game_index, i) + 71)) for i in range(attempts)]
        row = {"game": game, "lifelong": {"summary": _summary(shared_rows), "rows": shared_rows},
               "scratch": {"summary": _summary(scratch_rows), "rows": scratch_rows},
               "random": {"summary": _summary(random_rows), "rows": random_rows}}
        report["worlds"].append(row)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(game, "lifelong", row["lifelong"]["summary"],
              "scratch", row["scratch"]["summary"],
              "random", row["random"]["summary"], flush=True)
    revisits_by_game = {}
    for game_index, game in enumerate(GAMES):
        rows = [_trial(arcade, game, _seed(game_index, i, True), lifelong, max_actions,
                       random.Random(_seed(game_index, i, True) + 71)) for i in range(revisits)]
        revisits_by_game[game] = {"summary": _summary(rows), "rows": rows}
        print("revisit", game, revisits_by_game[game]["summary"], flush=True)
    for row in report["worlds"]: row["revisit"] = revisits_by_game[row["game"]]
    report["aggregate"] = {}
    for condition in ("lifelong", "scratch", "random", "revisit"):
        rows = [trial for world in report["worlds"] for trial in world[condition]["rows"]]
        report["aggregate"][condition] = _summary(rows)
    report["retention"] = {
        "revisit_minus_initial_any_progress_rate": (
            report["aggregate"]["revisit"]["any_progress_rate"] -
            report["aggregate"]["lifelong"]["any_progress_rate"]),
        "revisit_minus_initial_normalized_level_score": (
            report["aggregate"]["revisit"]["normalized_level_score"] -
            report["aggregate"]["lifelong"]["normalized_level_score"])}
    report["final_brain"] = lifelong.status(); report["complete"] = True
    report["finished_unix"] = time.time()
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lifelong.save(output.with_name("WAILAH_FROZEN_ARC3_BRAIN.json"))
    print(json.dumps({"aggregate": report["aggregate"], "retention": report["retention"],
                      "final_brain": report["final_brain"]}, indent=2), flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--revisits", type=int, default=2); parser.add_argument("--max-actions", type=int, default=512)
    args = parser.parse_args(argv); run(args.project.resolve(), args.output.resolve(), args.attempts,
                                        args.revisits, args.max_actions); return 0


if __name__ == "__main__": raise SystemExit(main())
