"""Let the universal explorer learn online in independent ARC-AGI-3 worlds."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

from .universal_explorer import UniversalExplorer


def _quiet_logger():
    logger = logging.getLogger("wailah.arc3.learning")
    logger.handlers.clear(); logger.addHandler(logging.NullHandler()); logger.setLevel(logging.CRITICAL)
    return logger


def _trial(arcade, game: str, seed: int, agent: UniversalExplorer | None,
           max_actions: int, rng: random.Random) -> dict:
    from arcengine import GameAction, GameState
    env = arcade.make(game, seed=seed, save_recording=False)
    response = env.reset()
    if response is None: raise RuntimeError(f"reset failed: {game}")
    if agent is not None: agent.reset_episode(response.levels_completed, response.frame)
    counts = {}; progress_steps = []; point_counts = {}; start = time.monotonic()
    for step in range(max_actions):
        actions = [int(x) for x in response.available_actions if int(x) != 0]
        if not actions: break
        before = response
        if agent is None:
            action = rng.choice(actions); reason = "random"
            action_data = {"x": rng.randrange(64), "y": rng.randrange(64)} if action == 6 else None
        else:
            action, evidence = agent.act(before.frame, actions, before.levels_completed); reason = evidence["reason"]
            action_data = evidence.get("action_data")
        counts[str(action)] = counts.get(str(action), 0) + 1
        game_action = GameAction.from_id(action)
        if action == 6:
            action_data = action_data or {"x": 0, "y": 0}
            game_action.set_data(action_data)
            point = f"{int(action_data['x'])},{int(action_data['y'])}"
            point_counts[point] = point_counts.get(point, 0) + 1
        # LocalEnvironmentWrapper accepts complex-action payloads through the
        # explicit data argument; mutating the enum alone does not transmit it.
        after = env.step(game_action, data=action_data)
        if after is None: break
        delta = int(after.levels_completed) - int(before.levels_completed)
        done = after.state in (GameState.WIN, GameState.GAME_OVER)
        reward = float(delta)
        if done and after.state == GameState.GAME_OVER and delta <= 0: reward = -1.0
        if agent is not None:
            agent.observe(before.frame, action, after.frame, reward, done,
                          before.levels_completed, after.levels_completed)
        if delta > 0: progress_steps.append(step + 1)
        response = after
        if done: break
    used = sum(counts.values())
    return {"seed": seed, "actions": used, "levels_completed": int(response.levels_completed),
            "win_levels": int(response.win_levels), "won": response.state == GameState.WIN,
            "terminal_state": response.state.value, "action_counts": counts,
            "unique_action6_points": len(point_counts),
            "dominant_action_fraction": max(counts.values()) / max(1, used),
            "progress_steps": progress_steps, "seconds": time.monotonic() - start}


def run(project: Path, output: Path, games: list[str], attempts: int, max_actions: int,
        seed_start: int | None = None) -> dict:
    os.environ.setdefault("MPLCONFIGDIR", str(project / ".mplconfig"))
    sys.path.insert(0, str(project / ".arc_agi_deps"))
    from arc_agi import Arcade
    arcade = Arcade(environments_dir=str(project / "arc_environments"),
                    recordings_dir=str(project / "arc_recordings"), logger=_quiet_logger())
    system_rng = random.SystemRandom(); rows = []
    memory_dir = output.parent / "arc3_universal_memories"; memory_dir.mkdir(parents=True, exist_ok=True)
    for game_index, game in enumerate(games):
        seed = (int(seed_start) + game_index * 100_000 if seed_start is not None else
                system_rng.randrange(1, 2**31 - attempts - 2))
        agent = UniversalExplorer(seed=91_000 + game_index)
        learning = []; random_rows = []
        for attempt in range(attempts):
            # The world seed changes on every attempt: exact trajectory memorization
            # alone cannot satisfy the audit.
            trial_seed = seed + attempt
            learning.append(_trial(arcade, game, trial_seed, agent, max_actions,
                                   random.Random(trial_seed + 71)))
            random_rows.append(_trial(arcade, game, trial_seed, None, max_actions,
                                      random.Random(trial_seed + 71)))
        agent.save(memory_dir / f"{game}.json")
        split = max(1, attempts // 2)
        first = np.asarray([x["levels_completed"] for x in learning[:split]], dtype=float)
        last = np.asarray([x["levels_completed"] for x in learning[split:]], dtype=float)
        rows.append({"game_id": game, "seed_start": seed,
                     "seed_commitment": hashlib.sha256(str(seed).encode()).hexdigest(),
                     "attempts": learning, "random_attempts": random_rows,
                     "learning_curve": {"first_half_mean_levels": float(first.mean()),
                                        "second_half_mean_levels": float(last.mean()) if len(last) else float(first.mean()),
                                        "best_levels": int(max(x["levels_completed"] for x in learning)),
                                        "final_levels": int(learning[-1]["levels_completed"]),
                                        "improved": bool(len(last) and last.mean() > first.mean())},
                     "brain": agent.status()})
    learned_total = sum(sum(x["levels_completed"] for x in row["attempts"]) for row in rows)
    random_total = sum(sum(x["levels_completed"] for x in row["random_attempts"]) for row in rows)
    report = {"format": "wailah-arc-agi3-online-learning-v1",
              "protocol": {"agent_input": "pixels, available controls, progress delta, termination",
                           "withheld": ["game identity", "instructions", "goal", "source code",
                                        "human solutions", "baseline action counts"],
                           "one_unchanged_mechanism_for_all_games": True,
                           "fresh_world_seed_each_attempt": True,
                           "seed_policy": "recorded fixed block" if seed_start is not None else "system random block",
                           "original_wailah_memories_mutated": False},
              "games": rows,
              "aggregate": {"games": len(rows), "attempts_per_game": attempts,
                            "learned_levels": learned_total, "random_levels": random_total,
                            "beats_random": learned_total > random_total,
                            "games_with_learning_curve_improvement": sum(
                                row["learning_curve"]["improved"] for row in rows)}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", nargs="+", default=["ls20", "tr87", "wa30", "g50t", "re86", "tu93"])
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--seed-start", type=int)
    args = parser.parse_args(argv)
    report = run(args.project.resolve(), args.output.resolve(), args.games, args.attempts,
                 args.max_actions, args.seed_start)
    print(json.dumps(report["aggregate"], indent=2))
    for row in report["games"]:
        print(row["game_id"], row["learning_curve"], row["brain"])
    return 0


if __name__ == "__main__": raise SystemExit(main())
