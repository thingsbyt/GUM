"""Frozen-style audit of semantic causal transfer under control/identity remaps.

This isolates the shared semantic layer.  It receives the same grounded event
stream produced by ConceptMemory from pixels in UniversalExplorer; task stage,
event grammar, control mapping, and solution are audit-only environment data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .semantic_world_model import SharedSemanticWorldModel


EVENTS = ("entity-appeared", "entity-moved-horizontal", "entity-moved-vertical",
          "entity-disappeared", "relation-created")
GRAMMAR = ("entity-appeared", "entity-moved-horizontal",
           "entity-disappeared", "relation-created")


def _grounded(kind: str, task_seed: int, step: int) -> str:
    """Generate identity-specific observations of an identity-free event."""
    color = 2 + ((task_seed * 17 + step * 7) % 13)
    identity = f"object:c{color}:a{2 + step % 4}:w{1 + step % 3}:h{1 + (step + 1) % 3}"
    if kind == "entity-appeared": return f"event:appeared:{identity}"
    if kind == "entity-disappeared": return f"event:disappeared:{identity}"
    if kind == "entity-moved-horizontal": return f"event:moved:{identity}:dx1:dy0"
    if kind == "entity-moved-vertical": return f"event:moved:{identity}:dx0:dy-1"
    return f"event:relation-created:{identity}:near:object:c{15-color}:a3:w2:h2"


class RemappedEventGrammarWorld:
    def __init__(self, task_seed: int, horizon: int):
        self.task_seed = int(task_seed); self.horizon = int(horizon)
        rng = np.random.default_rng(task_seed)
        permutation = list(rng.permutation(len(EVENTS)))
        self.actions = [1, 2, 3, 4, 5]
        self.outcome = {action: EVENTS[permutation[index]]
                        for index, action in enumerate(self.actions)}
        self.solution = [next(action for action, event in self.outcome.items() if event == target)
                         for target in GRAMMAR]
        self.reset()

    def reset(self):
        self.stage = 0; self.steps = 0; self.max_stage = 0

    def step(self, action: int):
        kind = self.outcome[int(action)]
        if kind == GRAMMAR[self.stage]:
            self.stage += 1
        else:
            self.stage = 1 if kind == GRAMMAR[0] else 0
        self.steps += 1; self.max_stage = max(self.max_stage, self.stage)
        success = self.stage == len(GRAMMAR)
        done = success or self.steps >= self.horizon
        return [_grounded(kind, self.task_seed, self.steps)], float(success), done


def _discover_source(model: SharedSemanticWorldModel, task_seed: int,
                     successes: int, max_steps: int) -> dict:
    world = RemappedEventGrammarWorld(task_seed, max_steps)
    rng = np.random.default_rng(task_seed + 91); model.reset_episode("source")
    found = 0; steps = 0; recent_actions = []; successful_actions = None
    while found < successes and steps < max_steps:
        action = int(rng.choice(world.actions)); recent_actions.append(action)
        events, reward, _ = world.step(action); steps += 1
        model.observe("source", action, events, bool(reward), bool(reward))
        if reward:
            found += 1; successful_actions = recent_actions[-len(GRAMMAR):]
            world.reset(); recent_actions = []; model.reset_episode("source")
    return {"steps": steps, "successes": found, "source_solution_audit_only": world.solution,
            "experienced_success_actions": successful_actions}


def _trial(model: SharedSemanticWorldModel, task_seed: int, horizon: int,
           policy: str, replay: list[int] | None = None) -> dict:
    world = RemappedEventGrammarWorld(task_seed, horizon)
    context = f"heldout-{task_seed}"; model.reset_episode(context)
    rng = np.random.default_rng(task_seed + 31337); usage = Counter(); reasons = Counter()
    while True:
        plan = model.recommend(context, world.actions) if policy == "semantic" else None
        if plan:
            action, reason = plan; reasons[reason] += 1
        elif policy == "replay":
            action = replay[world.steps % len(replay)]; reason = "replay-source-actions"
        else:
            least = min(usage[action] for action in world.actions)
            candidates = [action for action in world.actions if usage[action] == least]
            action = int(rng.choice(candidates)); reason = "balanced-exploration"
        usage[action] += 1
        events, reward, done = world.step(action)
        model.observe(context, action, events, bool(reward), done)
        if done:
            return {"success": bool(reward), "steps": world.steps,
                    "max_stage": world.max_stage, "semantic_actions": sum(reasons.values()),
                    "reasons": dict(reasons), "solution_audit_only": world.solution}


def _clone(model: SharedSemanticWorldModel) -> SharedSemanticWorldModel:
    copy = SharedSemanticWorldModel(); copy.restore(json.loads(json.dumps(model.export())))
    return copy


def _summarize(rows: list[dict]) -> dict:
    count = len(rows); successes = sum(row["success"] for row in rows)
    rate = successes / max(1, count); z = 1.959963984540054
    denominator = 1 + z * z / max(1, count)
    center = (rate + z * z / (2 * max(1, count))) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / max(1, count) +
                           z * z / (4 * max(1, count) ** 2)) / denominator
    return {"tasks": count, "successes": successes,
            "success_rate": float(rate),
            "success_rate_wilson_95": [max(0.0, center - radius), min(1.0, center + radius)],
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_max_stage": float(np.mean([row["max_stage"] for row in rows])),
            "rows": rows}


def run_benchmark(output: Path, *, tasks: int = 32, horizon: int = 80,
                  source_successes: int = 8, source_seed: int = 470011,
                  heldout_seed: int = 880031) -> dict:
    trained = SharedSemanticWorldModel()
    source = _discover_source(trained, source_seed, source_successes, 100_000)
    if source["successes"] < source_successes:
        raise RuntimeError("source discovery budget exhausted")
    target_seeds = [heldout_seed + index * 1009 for index in range(tasks)]
    semantic, no_transfer, replay = [], [], []
    for task_seed in target_seeds:
        semantic.append(_trial(_clone(trained), task_seed, horizon, "semantic"))
        no_transfer.append(_trial(SharedSemanticWorldModel(), task_seed, horizon, "scratch"))
        replay.append(_trial(SharedSemanticWorldModel(), task_seed, horizon, "replay",
                             source["experienced_success_actions"]))
    source_text = Path(__file__).read_bytes()
    report = {
        "format": "wailah-semantic-causal-transfer-audit-v1",
        "protocol": {"source_seed": source_seed, "source_successes": source_successes,
                     "heldout_seed": heldout_seed, "heldout_tasks": tasks,
                     "horizon": horizon, "actions": 5, "grammar_length": len(GRAMMAR),
                     "variation": "every held-out task remaps controls and visual identities",
                     "reward": "zero until the complete event grammar; then +1 and termination",
                     "agent_input": "grounded visual-event stream, available controls, reward, done",
                     "agent_not_given": ["event grammar", "task stage", "control mapping",
                                         "solution", "task seed"],
                     "isolation_note": "tests the shared semantic layer; the production agent's pixel-to-event parser is held constant"},
        "implementation_sha256": hashlib.sha256(source_text).hexdigest(),
        "source_discovery": source,
        "learned_theory": trained.status(),
        "aggregate": {"semantic_transfer": _summarize(semantic),
                      "no_cross_task_memory": _summarize(no_transfer),
                      "numeric_action_replay": _summarize(replay)}}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=32); parser.add_argument("--horizon", type=int, default=80)
    args = parser.parse_args(argv); report = run_benchmark(args.output, tasks=args.tasks, horizon=args.horizon)
    print(json.dumps({name: {key: value for key, value in rows.items() if key != "rows"}
                      for name, rows in report["aggregate"].items()}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
