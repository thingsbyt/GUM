"""External, read-only intelligence audit against public ARC-AGI-3 games.

This module deliberately lives outside WAILAH's training loop.  The tested
policy sees only pixels and available controls.  It is never given the game
identifier, implementation, instructions, goal, or human action baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def _json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _quiet_logger():
    logger = logging.getLogger("wailah.arc3")
    logger.handlers.clear(); logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger


class Bridge:
    def __init__(self, python: Path, project: Path, workspace: Path):
        self.process = subprocess.Popen(
            [str(python), "-m", "jepa_asteroids.arc_agi3_bridge",
             "--workspace", str(workspace)],
            cwd=str(project), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
        )

    def act(self, response):
        request = {"frame": [layer.tolist() for layer in response.frame],
                   "available_actions": list(response.available_actions)}
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            error = self.process.stderr.read()
            raise RuntimeError(f"WAILAH bridge stopped: {error[-2000:]}")
        return json.loads(line)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close(); self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired: self.process.kill()


def _run_game(arcade, game_id: str, seed: int, max_actions: int, policy: str,
              bridge: Bridge | None = None) -> dict:
    from arcengine import GameAction, GameState
    env = arcade.make(game_id, seed=seed, save_recording=False)
    if env is None: raise RuntimeError(f"ARC-AGI-3 game unavailable: {game_id}")
    response = env.reset()
    if response is None: raise RuntimeError(f"ARC-AGI-3 reset failed: {game_id}")
    start_levels = int(response.levels_completed); rng = random.Random(seed + 9901)
    trace = []; failure = None; started = time.monotonic(); action_counts = {}
    observation_hashes = {hashlib.sha256(b"".join(layer.tobytes() for layer in response.frame)).hexdigest()}
    for step in range(max_actions):
        available = [int(value) for value in response.available_actions if int(value) != 0]
        if not available: failure = "no non-reset actions"; break
        if policy == "random":
            action_id = rng.choice(available); decision = {"policy": "random"}
        else:
            decision = bridge.act(response)
            if "error" in decision: failure = decision["error"]; break
            action_id = int(decision["action_id"])
            if action_id not in available:
                failure = f"policy emitted unavailable action {action_id}"; break
        action_counts[str(action_id)] = action_counts.get(str(action_id), 0) + 1
        before = int(response.levels_completed)
        response = env.step(GameAction.from_id(action_id))
        if response is None: failure = "environment step returned no observation"; break
        observation_hashes.add(hashlib.sha256(
            b"".join(layer.tobytes() for layer in response.frame)).hexdigest())
        after = int(response.levels_completed)
        if after != before or step < 8:
            trace.append({"step": step + 1, "action_id": action_id,
                          "levels_before": before, "levels_after": after,
                          "state": response.state.value})
        if response.state in (GameState.WIN, GameState.GAME_OVER): break
    levels = int(response.levels_completed) - start_levels
    actions_used = sum(action_counts.values())
    return {"game_id": game_id, "seed": seed,
            "seed_commitment": hashlib.sha256(str(seed).encode()).hexdigest(),
            "policy": policy, "action_budget": max_actions, "actions_used": actions_used,
            "action_counts": action_counts,
            "dominant_action_fraction": (max(action_counts.values()) / actions_used
                                         if actions_used else None),
            "unique_visual_states": len(observation_hashes),
            "levels_completed": levels, "win_levels": int(response.win_levels),
            "won": response.state == GameState.WIN, "terminal_state": response.state.value,
            "failure": failure, "seconds": time.monotonic() - started,
            "event_trace": trace}


def run_audit(project: Path, workspace: Path, output: Path, *, games: list[str],
              max_actions: int, bridge_python: Path) -> dict:
    # Import the independently installed official SDK only in this Python 3.12 process.
    environment_dir = project / "arc_environments"
    recordings_dir = project / "arc_recordings"
    os.environ.setdefault("MPLCONFIGDIR", str(project / ".mplconfig"))
    logging.getLogger("arc_agi").setLevel(logging.CRITICAL)
    logging.getLogger("arc_agi.scorecard").setLevel(logging.CRITICAL)
    sys.path.insert(0, str(project / ".arc_agi_deps"))
    from arc_agi import Arcade
    arcade = Arcade(environments_dir=str(environment_dir), recordings_dir=str(recordings_dir),
                    logger=_quiet_logger())
    # Seeds are generated at audit time and committed by hash in the report.
    # They were not available when WAILAH's checkpoints were trained.
    system_rng = random.SystemRandom()
    seeds = {game: system_rng.randrange(1, 2**31 - 1) for game in games}
    bridge = Bridge(bridge_python, project, workspace)
    try:
        rows = []
        for game in games:
            rows.append({"random": _run_game(arcade, game, seeds[game], max_actions, "random"),
                         "wailah": _run_game(arcade, game, seeds[game], max_actions,
                                              "frozen-wailah", bridge)})
    finally:
        bridge.close()
    random_levels = sum(row["random"]["levels_completed"] for row in rows)
    wailah_levels = sum(row["wailah"]["levels_completed"] for row in rows)
    report = {
        "format": "wailah-external-intelligence-audit-v1",
        "benchmark": "ARC-AGI-3 public interactive environments",
        "protocol": {"agent_input": "pixels plus currently available action IDs only",
                     "withheld": ["game identity", "instructions", "goal", "source code",
                                  "human baseline actions"],
                     "learning": "disabled; frozen pre-audit WAILAH brain",
                     "same_seed_for_random_and_wailah": True,
                     "metric": "levels completed within an equal action budget"},
        "games": rows,
        "aggregate": {"games": len(rows), "random_levels": random_levels,
                      "wailah_levels": wailah_levels,
                      "wailah_wins": sum(row["wailah"]["won"] for row in rows),
                      "beats_random": wailah_levels > random_levels},
    }
    report["report_sha256_without_self_hash"] = _json_hash(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", nargs="+", default=["ls20", "ft09", "cd82"])
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--bridge-python", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_audit(args.project.resolve(), args.workspace.resolve(), args.output.resolve(),
                       games=args.games, max_actions=args.max_actions,
                       bridge_python=args.bridge_python.resolve())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
