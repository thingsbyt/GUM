"""Audit learned visual language, semantic composition, growth, and retention."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .semantic_composer import SemanticSkillComposer


EVENTS = ("entity-appeared", "entity-moved-horizontal", "entity-moved-vertical",
          "entity-disappeared", "relation-created")
GLYPHS = (
    ((1, 1, 1), (0, 1, 0), (0, 1, 0)),
    ((1, 0, 0), (1, 1, 1), (0, 0, 1)),
    ((1, 1, 0), (0, 1, 0), (0, 1, 1)),
    ((1, 0, 1), (1, 1, 1), (0, 1, 0)),
    ((1, 1, 1), (1, 0, 0), (1, 1, 1)),
)


def render_goal(tokens: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); offset = int(rng.integers(1, 3))
    frame = np.zeros((1, 9, 5 + len(tokens) * 6), dtype=np.uint8)
    for position, token in enumerate(tokens):
        mask = np.asarray(GLYPHS[token], dtype=bool); x = offset + position * 6
        frame[0, 2:5, x:x + 3][mask] = int(rng.integers(2, 16))
    return frame


def _grounded(kind: str, task_seed: int, step: int) -> str:
    color = 2 + ((task_seed * 11 + step * 7) % 13)
    identity = f"object:c{color}:a{2 + step % 4}:w{1 + step % 3}:h{1 + (step + 1) % 3}"
    if kind == "entity-appeared": return f"event:appeared:{identity}"
    if kind == "entity-disappeared": return f"event:disappeared:{identity}"
    if kind == "entity-moved-horizontal": return f"event:moved:{identity}:dx1:dy0"
    if kind == "entity-moved-vertical": return f"event:moved:{identity}:dx0:dy-1"
    return f"event:relation-created:{identity}:near:object:c{15-color}:a3:w2:h2"


class VisualCommandWorld:
    """Unknown visual command plus independently remapped causal controls."""
    actions = [1, 2, 3, 4, 5]

    def __init__(self, token_meanings: tuple[str, ...], goal_tokens: tuple[int, ...],
                 task_seed: int, horizon: int):
        self.token_meanings = token_meanings; self.goal_tokens = goal_tokens
        self.target = tuple(token_meanings[token] for token in goal_tokens)
        self.task_seed = int(task_seed); self.horizon = int(horizon)
        order = np.random.default_rng(task_seed).permutation(len(EVENTS))
        self.outcome = {action: EVENTS[int(order[index])]
                        for index, action in enumerate(self.actions)}
        self.solution = [next(action for action, event in self.outcome.items() if event == target)
                         for target in self.target]
        self.frame = render_goal(goal_tokens, task_seed + 17); self.reset()

    def reset(self):
        self.stage = 0; self.steps = 0; self.max_stage = 0

    def step(self, action: int):
        kind = self.outcome[int(action)]
        if kind == self.target[self.stage]: self.stage += 1
        else: self.stage = 1 if kind == self.target[0] else 0
        self.steps += 1; self.max_stage = max(self.max_stage, self.stage)
        success = self.stage == len(self.target); done = success or self.steps >= self.horizon
        return [_grounded(kind, self.task_seed, self.steps)], float(success), done


def _clone(model: SemanticSkillComposer) -> SemanticSkillComposer:
    result = SemanticSkillComposer(); result.restore(json.loads(json.dumps(model.export())))
    return result


def _episode(model: SemanticSkillComposer, world: VisualCommandWorld, context: str,
             policy: str, rng: np.random.Generator, replay: list[int] | None = None) -> dict:
    world.reset(); model.reset_episode(context, world.frame); usage = Counter(); reasons = Counter()
    trace = []
    while True:
        plan = model.recommend(world.actions) if policy == "compose" else None
        if plan:
            action, reason = plan; reasons[reason] += 1
        elif policy == "replay":
            action = int(replay[world.steps % len(replay)]); reason = "replay-old-buttons"
        else:
            least = min(usage[action] for action in world.actions)
            candidates = [action for action in world.actions if usage[action] == least]
            action = int(rng.choice(candidates)); reason = "balanced-exploration"
        usage[action] += 1; events, reward, done = world.step(action)
        trace.append({"step": world.steps, "action": int(action),
                      "observed_event": events[0], "reward": float(reward),
                      "controller_reason": reason})
        model.observe(action, events, bool(reward), done)
        if done:
            return {"success": bool(reward), "steps": world.steps,
                    "max_stage": world.max_stage, "composed_actions": sum(reasons.values()),
                    "reasons": dict(reasons), "trace": trace,
                    "goal_tokens_audit_only": list(world.goal_tokens),
                    "target_events_audit_only": list(world.target),
                    "solution_audit_only": world.solution}


def _teach(model: SemanticSkillComposer, meanings: tuple[str, ...], language_seed: int,
           horizon: int = 100) -> tuple[list[dict], list[int]]:
    # Every symbol occurs in multiple independently rewarded commands. The
    # learner itself chooses the order using uncertainty in its current lexicon.
    pool = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (0, 2), (1, 3), (2, 4), (3, 0), (4, 1)]
    rows = []; last_success = None
    while pool:
        frames = [render_goal(tokens, language_seed + 100 + index)
                  for index, tokens in enumerate(pool)]
        choice = model.choose_curriculum(frames); tokens = pool.pop(choice)
        task_seed = language_seed + 1009 * (len(rows) + 1)
        world = VisualCommandWorld(meanings, tokens, task_seed, horizon)
        rng = np.random.default_rng(task_seed + 71); attempts = 0
        while True:
            attempts += 1
            result = _episode(model, world, "curriculum", "explore", rng)
            if result["success"]:
                last_success = result["solution_audit_only"]
                rows.append({"curriculum_index": len(rows), "attempts": attempts,
                             "dictionary_size_after": model.status()["grounded_symbol_meanings"],
                             **result})
                break
            if attempts >= 32:
                raise RuntimeError("curriculum discovery budget exhausted")
    return rows, last_success


def _corrupt_dictionary(model: SemanticSkillComposer) -> SemanticSkillComposer:
    result = _clone(model); grounded = result.status()["grounded_dictionary"]
    tokens = sorted(grounded); meanings = [grounded[token] for token in tokens]
    rotated = meanings[1:] + meanings[:1]
    for token, meaning in zip(tokens, rotated):
        result.token_evidence[token] = Counter({meaning: 10})
    return result


def _summary(rows: list[dict]) -> dict:
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


def run_benchmark(output: Path, *, tasks: int = 128, horizon: int = 80,
                  language_seed: int = 1_220_017, heldout_seed: int = 8_810_021) -> dict:
    mapping = tuple(EVENTS[int(index)] for index in
                    np.random.default_rng(language_seed).permutation(len(EVENTS)))
    composer = SemanticSkillComposer(); dictionary_before = composer.status()["grounded_symbol_meanings"]
    curriculum, replay_actions = _teach(composer, mapping, language_seed)
    dictionary_after = composer.status()["grounded_symbol_meanings"]
    dictionary_hash = hashlib.sha256(json.dumps(
        composer.status()["grounded_dictionary"], sort_keys=True).encode()).hexdigest()

    rng = np.random.default_rng(heldout_seed); goals = []
    while len(goals) < tasks:
        candidate = tuple(int(x) for x in rng.integers(0, 5, size=4))
        # Permit reuse of a skill later in the plan, but avoid adjacent
        # duplicates whose two identical visual transitions are ambiguous.
        if all(a != b for a, b in zip(candidate, candidate[1:])) and candidate not in goals:
            goals.append(candidate)
    full_rows, scratch_rows, corrupted_rows, replay_rows = [], [], [], []
    task_records = []
    for index, tokens in enumerate(goals):
        task_seed = heldout_seed + index * 2029; context = f"heldout-{task_seed}"
        world = VisualCommandWorld(mapping, tokens, task_seed, horizon)
        full = _episode(composer, world, context, "compose",
                        np.random.default_rng(task_seed + 1))
        scratch = _episode(SemanticSkillComposer(), world, context, "explore",
                           np.random.default_rng(task_seed + 1))
        corrupted = _episode(_corrupt_dictionary(composer), world, context, "compose",
                             np.random.default_rng(task_seed + 1))
        replay = _episode(SemanticSkillComposer(), world, context, "replay",
                          np.random.default_rng(task_seed + 1), replay_actions)
        full_rows.append(full); scratch_rows.append(scratch)
        corrupted_rows.append(corrupted); replay_rows.append(replay)
        task_records.append((world, context, task_seed))

    # Revisit after all other tasks. The semantic dictionary and each local
    # causal model must survive; a mature context should execute near optimally.
    revisit_rows = [_episode(composer, world, context, "compose",
                             np.random.default_rng(task_seed + 99))
                    for world, context, task_seed in task_records]
    dictionary_hash_after = hashlib.sha256(json.dumps(
        composer.status()["grounded_dictionary"], sort_keys=True).encode()).hexdigest()
    training_pairs = {tuple(row["goal_tokens_audit_only"]) for row in curriculum}
    heldout_pairs = {(a, b) for goal in goals for a, b in zip(goal, goal[1:])}
    seen_pairs = {(a, b) for pair in training_pairs for a, b in zip(pair, pair[1:])}
    report = {"format": "wailah-compositional-semantic-mind-audit-v1",
        "protocol": {"language_seed": language_seed, "heldout_seed": heldout_seed,
            "curriculum_commands": 10, "heldout_tasks": tasks, "horizon": horizon,
            "visual_vocabulary": 5, "heldout_command_length": 4,
            "variation": "unseen symbol compositions, fresh control remap and visual identity per task",
            "reward": "zero until exact complete command; then +1 and termination",
            "agent_input": "visual command pixels, grounded visual-event stream, controls, reward, done",
            "agent_not_given": ["symbol dictionary", "target event sequence", "task stage",
                                "control mapping", "solution", "task seed"],
            "isolation_note": "visual command parsing is end-to-end; production pixel-to-event perception is held constant"},
        "curriculum": {"dictionary_before": dictionary_before, "dictionary_after": dictionary_after,
            "chosen_by_agent_uncertainty": True, "rows": curriculum,
            "learned_dictionary_audit_only": composer.status()["grounded_dictionary"]},
        "composition_novelty": {"exact_heldout_commands_seen_during_training": 0,
            "heldout_bigrams": len(heldout_pairs), "heldout_bigrams_not_seen_in_training": len(heldout_pairs - seen_pairs)},
        "aggregate": {"semantic_composer": _summary(full_rows),
            "no_cross_task_memory": _summary(scratch_rows),
            "corrupted_learned_dictionary": _summary(corrupted_rows),
            "numeric_action_replay": _summary(replay_rows),
            "revisit_after_all_tasks": _summary(revisit_rows)},
        "growth": {"first_encounter_mean_steps": float(np.mean([row["steps"] for row in full_rows])),
            "revisit_mean_steps": float(np.mean([row["steps"] for row in revisit_rows])),
            "experience_speedup": float(np.mean([row["steps"] for row in full_rows]) /
                                        np.mean([row["steps"] for row in revisit_rows])),
            "dictionary_unchanged_after_heldout_and_revisit": dictionary_hash == dictionary_hash_after},
        "final_composer": composer.status(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8"); return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=128); parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--language-seed", type=int, default=1_220_017)
    parser.add_argument("--heldout-seed", type=int, default=8_810_021)
    args = parser.parse_args(argv); report = run_benchmark(
        args.output, tasks=args.tasks, horizon=args.horizon,
        language_seed=args.language_seed, heldout_seed=args.heldout_seed)
    print(json.dumps({name: {key: value for key, value in rows.items() if key != "rows"}
                      for name, rows in report["aggregate"].items()} | {"growth": report["growth"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
