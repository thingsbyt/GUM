"""Generated sparse-reward benchmark for autonomous visual event skills."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .universal_explorer import UniversalExplorer


class ShiftingEventChainWorld:
    """Hidden control chain whose intermediate visual events have zero reward.

    The required controls are randomly remapped per task.  Each correct prefix
    visibly adds an object, a wrong control silently removes the prefix, and the
    entire layout translates between episodes.  Translation defeats rote
    absolute-frame replay while preserving the learned event structure.
    """

    def __init__(self, task_seed: int, action_count: int = 7,
                 chain_length: int = 5, horizon: int = 160):
        self.action_count = int(action_count); self.chain_length = int(chain_length)
        self.horizon = int(horizon)
        # Control 6 is a coordinate-bearing point action in the ARC interface;
        # this benchmark uses ordinary discrete controls so point-affordance
        # learning cannot confound event-sequence learning.
        self.control_ids = (1, 2, 3, 4, 5, 7, 8)[:self.action_count]
        rng = np.random.default_rng(task_seed)
        self.chain = [int(x) for x in rng.choice(
            self.control_ids, size=self.chain_length, replace=False)]

    @property
    def actions(self) -> list[int]:
        return list(self.control_ids)

    def reset(self, episode_seed: int) -> np.ndarray:
        rng = np.random.default_rng(episode_seed)
        self.shift_x = int(rng.integers(-2, 3)) * 3
        self.shift_y = int(rng.integers(-2, 3)) * 2
        self.stage = 0; self.steps = 0; self.resets = 0
        self.max_stage = 0; self.actions_by_stage = defaultdict(Counter)
        return self.render()

    def step(self, action: int):
        self.actions_by_stage[self.stage][int(action)] += 1
        if int(action) == self.chain[self.stage]:
            self.stage += 1
        else:
            self.stage = 0; self.resets += 1
        success = self.stage == self.chain_length
        self.max_stage = max(self.max_stage, self.stage)
        reward = 1.0 if success else 0.0
        self.steps += 1; done = success or self.steps >= self.horizon
        return self.render(), reward, done, {
            "success": success, "steps": self.steps, "resets": self.resets,
            "terminal_stage": self.stage, "max_stage": self.max_stage,
            "actions_by_stage_audit_only": {str(stage): dict(rows)
                                             for stage, rows in self.actions_by_stage.items()},
            "chain_audit_only": list(self.chain)}

    def render(self) -> np.ndarray:
        frame = np.zeros((1, 64, 64), dtype=np.uint8)
        x0, y0 = 15 + self.shift_x, 25 + self.shift_y
        frame[0, y0:y0 + 5, x0:x0 + 5] = 12
        frame[0, y0 + 13:y0 + 16, x0 + 2:x0 + 11] = 2
        for index in range(self.stage):
            x, y = x0 + 10 + 7 * index, y0 - 8 + 4 * (index % 2)
            frame[0, y:y + 3, x:x + 3] = 3 + index
        return frame


def _agent_condition(task_seed: int, episodes: int, *, options: bool,
                     horizon: int, seed_base: int) -> tuple[list[dict], dict]:
    world = ShiftingEventChainWorld(task_seed, horizon=horizon)
    agent = UniversalExplorer(seed=seed_base + task_seed % 10_000)
    rows = []
    for episode in range(episodes):
        observation = world.reset(seed_base + task_seed * 100 + episode)
        agent.reset_episode(0, observation)
        if not options:
            # Continue learning the same memories, but prevent event options
            # from selecting actions. This isolates their control contribution.
            agent.event_options.recommend = lambda *_args, **_kwargs: None
        while True:
            action, _ = agent.act(observation, world.actions, 0)
            nxt, reward, done, info = world.step(action)
            agent.observe(observation, action, nxt, reward, done, 0, 0)
            observation = nxt
            if done:
                rows.append({"episode": episode, **info}); break
    return rows, agent.status()


def _random_condition(task_seed: int, episodes: int, *, horizon: int,
                      seed_base: int) -> list[dict]:
    world = ShiftingEventChainWorld(task_seed, horizon=horizon); rows = []
    for episode in range(episodes):
        rng = np.random.default_rng(seed_base + task_seed * 100 + episode + 77)
        world.reset(seed_base + task_seed * 100 + episode)
        while True:
            _, _, done, info = world.step(int(rng.choice(world.actions)))
            if done:
                rows.append({"episode": episode, **info}); break
    return rows


def _summary(rows: list[dict]) -> dict:
    split = max(1, len(rows) // 2)
    first = rows[:split]; second = rows[split:]
    return {"episodes": len(rows), "successes": sum(row["success"] for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "first_half_success_rate": float(np.mean([row["success"] for row in first])),
            "second_half_success_rate": float(np.mean([row["success"] for row in second])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_max_stage": float(np.mean([row["max_stage"] for row in rows])),
            "rows": rows}


def run_benchmark(output: Path, *, tasks: int = 12, episodes: int = 12,
                  horizon: int = 160, seed_base: int = 730_000_000,
                  task_offset: int = 0) -> dict:
    results = []
    for index in range(tasks):
        task_number = task_offset + index
        task_seed = 81_000 + task_number * 131
        learned, brain = _agent_condition(task_seed, episodes, options=True,
                                          horizon=horizon, seed_base=seed_base)
        ablated, _ = _agent_condition(task_seed, episodes, options=False,
                                      horizon=horizon, seed_base=seed_base)
        random_rows = _random_condition(task_seed, episodes, horizon=horizon,
                                        seed_base=seed_base)
        results.append({"task": task_number, "task_seed": task_seed,
                        "event_options": _summary(learned),
                        "without_event_option_control": _summary(ablated),
                        "random": _summary(random_rows),
                        "option_memory": brain["event_options"]})
    def aggregate(name):
        rows = [result[name] for result in results]
        return {"tasks": tasks, "episodes": tasks * episodes,
                "successes": int(sum(row["successes"] for row in rows)),
                "success_rate": float(np.mean([row["success_rate"] for row in rows])),
                "first_half_success_rate": float(np.mean(
                    [row["first_half_success_rate"] for row in rows])),
                "second_half_success_rate": float(np.mean(
                    [row["second_half_success_rate"] for row in rows]))}
    report = {"format": "wailah-generated-event-option-audit-v1",
              "protocol": {"tasks": tasks, "episodes_per_task": episodes,
                           "task_offset": task_offset,
                           "actions": 7, "chain_length": 5, "horizon": horizon,
                           "reward": "zero for intermediate events; +1 only for full chain",
                           "variation": "hidden action remap per task; translated layout per episode",
                           "agent_input": "pixels, action set, terminal reward, termination",
                           "agent_not_given": ["chain", "stage", "event labels", "coordinates",
                                               "task seed", "solution"]},
              "aggregate": {name: aggregate(name) for name in
                            ("event_options", "without_event_option_control", "random")},
              "tasks": results}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--task-offset", type=int, default=0)
    args = parser.parse_args(argv)
    report = run_benchmark(args.output, tasks=args.tasks, episodes=args.episodes,
                           horizon=args.horizon, task_offset=args.task_offset)
    print(json.dumps(report["aggregate"], indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
