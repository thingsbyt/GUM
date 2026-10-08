"""Five-agent collective learning audit with sealed novel compositions.

Four specialists learn overlapping pieces of a visual event language in
separate worlds.  They exchange only provenance-bearing semantic claims.  A
fifth agent must combine their lessons, ground freshly remapped controls, and
execute commands no specialist ever encountered.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .collective_mind import (ExperiencePacket, SharedExperienceLibrary,
                              compile_packets, naive_digest)
from .compositional_mind_benchmark import EVENTS, VisualCommandWorld, _episode, render_goal
from .semantic_composer import SemanticSkillComposer, visual_symbol_sequence


SPECIALISTS = {
    "agent-alfa": (0, 1),
    "agent-bravo": (1, 2),
    "agent-charlie": (2, 3),
    "agent-delta": (3, 4),
}


def _meanings(seed: int) -> tuple[str, ...]:
    order = np.random.default_rng(seed).permutation(len(EVENTS))
    return tuple(EVENTS[int(index)] for index in order)


def _train_specialist(agent_id: str, tokens: tuple[int, int], meanings: tuple[str, ...], seed: int):
    mind = SemanticSkillComposer(); rows = []; successful_action_traces = []
    curriculum = [tokens, tokens[::-1], tokens, tokens[::-1]]
    for index, command in enumerate(curriculum):
        task_seed = seed + index * 1009; attempts = 0
        while True:
            attempts += 1; world = VisualCommandWorld(meanings, command, task_seed, 90)
            result = _episode(mind, world, f"{agent_id}-private-{index}", "explore",
                              np.random.default_rng(task_seed + attempts * 37))
            if result["success"]:
                rows.append({"command": list(command), "attempts": attempts,
                             "steps": result["steps"], "private_context": f"private-{index}"})
                successful_action_traces.append([int(step["action"]) for step in result["trace"]])
                break
            if attempts >= 64:
                raise RuntimeError(f"{agent_id} could not finish private curriculum")
    return mind, rows, successful_action_traces


def _raw_replay(world: VisualCommandWorld, actions: list[int]) -> dict:
    world.reset(); stage_max = 0
    for step in range(world.horizon):
        events, reward, done = world.step(actions[step % len(actions)])
        stage_max = max(stage_max, world.max_stage)
        if done:
            return {"success": bool(reward), "steps": world.steps, "max_stage": stage_max}
    raise AssertionError("world did not terminate")


def _evaluate(mode: str, meanings: tuple[str, ...], goals: list[tuple[int, ...]], seed: int,
              horizon: int, library: SharedExperienceLibrary, packets: list[ExperiencePacket],
              raw_actions: list[int]):
    rows = []
    for index, goal in enumerate(goals):
        task_seed = seed + index * 2029; world = VisualCommandWorld(meanings, goal, task_seed, horizon)
        if mode == "raw-trace-replay":
            result = _raw_replay(world, raw_actions)
        else:
            mind = SemanticSkillComposer()
            used = []
            if mode == "verified-collective": used = library.digest_into(mind)
            elif mode == "naive-unverified": naive_digest(packets, mind)
            result = _episode(mind, world, f"sealed-{task_seed}",
                              "compose" if mode != "isolated" else "explore",
                              np.random.default_rng(task_seed + 71))
            result = {key: value for key, value in result.items() if key != "trace"}
            if mode == "verified-collective":
                library.validate(used, f"integrator-{index}", bool(result["success"]))
                result["digested_packets"] = len(used)
                result["locally_grounded_controls"] = mind.status()["counterfactual_queries"]
        rows.append(result)
    return rows


def _summary(rows: list[dict]) -> dict:
    return {"tasks": len(rows), "successes": sum(row["success"] for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_max_stage": float(np.mean([row["max_stage"] for row in rows])),
            "rows": rows}


def run_collective_audit(output, *, tasks=64, horizon=42,
                         language_seed=17_440_031, heldout_seed=91_220_043) -> dict:
    meanings = _meanings(language_seed); library = SharedExperienceLibrary()
    specialist_rows = {}; packets = []; raw_actions = []
    for index, (agent_id, tokens) in enumerate(SPECIALISTS.items()):
        task = f"learn-private-symbols-{tokens[0]}-{tokens[1]}"
        library.assign(agent_id, task)
        mind, rows, action_traces = _train_specialist(
            agent_id, tokens, meanings, language_seed + (index + 1) * 100_003)
        member_packets = compile_packets(agent_id, mind)
        for packet in member_packets: library.ingest(packet)
        library.complete(agent_id, task); packets.extend(member_packets)
        # Authentic successful private traces, deliberately unusable as a
        # policy because every held-out world remaps the controls.
        raw_actions.extend(action for trace in action_traces for action in trace)
        specialist_rows[agent_id] = {"private_tokens": list(tokens), "curriculum": rows,
            "grounded_symbols": mind.status()["grounded_symbol_meanings"],
            "published_packets": [packet.to_dict() for packet in member_packets],
            "private_controls_shared": False, "private_context_models_shared": False}

    # A high-confidence but single-source contradiction.  Naive summation
    # trusts it; the collective prefers two independent honest observations.
    token_two = visual_symbol_sequence(render_goal((2,), language_seed + 7))[0]
    wrong = next(event for event in EVENTS if event != meanings[2])
    noisy = ExperiencePacket("agent-echo-noisy", token_two, wrong, 20,
                             ("unsupported-noisy-report",), confidence=1.0)
    library.ingest(noisy); packets.append(noisy)
    exchange_before_target = library.export()

    rng = np.random.default_rng(heldout_seed); goals = []
    while len(goals) < tasks:
        candidate = tuple(int(x) for x in rng.permutation(5))
        if candidate not in goals: goals.append(candidate)
    conditions = {}
    # Each condition receives an identical fresh world, budget, and random seed.
    for mode in ("isolated", "raw-trace-replay", "naive-unverified", "verified-collective"):
        conditions[mode] = _summary(_evaluate(mode, meanings, goals, heldout_seed,
                                                   horizon, library, packets, raw_actions))

    collective = conditions["verified-collective"]; isolated = conditions["isolated"]
    naive = conditions["naive-unverified"]
    speed_advantage = isolated["mean_steps"] / max(1.0, collective["mean_steps"])
    final_library = library.status(); decisions = library.accepted_claims()
    report = {"format": "wailah-collective-learning-v18-audit-v1",
        "classification": "multi-agent asynchronous collective learning with provenance and local grounding",
        "protocol": {"specialist_agents": len(SPECIALISTS), "integrator_agents": tasks,
            "sealed_tasks": tasks, "horizon": horizon, "language_seed": language_seed,
            "heldout_seed": heldout_seed, "heldout_command_length": 5,
            "every_heldout_command_uses_all_specialists": True,
            "exact_heldout_commands_seen_by_specialists": 0,
            "fresh_control_remap_per_heldout_task": True,
            "shared": ["identity-free semantic claims", "evidence counts", "provenance hashes"],
            "not_shared": ["numeric controls", "coordinates", "private context models",
                           "action traces", "heldout commands", "solutions"],
            "simultaneous_co_control_tested": False},
        "specialists": specialist_rows,
        "exchange": {"before_target": exchange_before_target, "decisions_after_target": decisions,
                     "status_after_target": final_library,
                     "injected_conflict": noisy.to_dict()},
        "aggregate": conditions,
        "effect": {"collective_minus_isolated_successes": collective["successes"] - isolated["successes"],
                   "collective_minus_naive_successes": collective["successes"] - naive["successes"],
                   "step_speed_advantage_over_isolated": speed_advantage},
        "precommitted_pass": (collective["success_rate"] >= 0.90
            and isolated["success_rate"] <= 0.20
            and collective["successes"] >= naive["successes"] + tasks // 2
            and final_library["quarantined_packets"] >= 1
            and all(row.get("locally_grounded_controls", 0) > 0
                    for row in collective["rows"] if row["success"])),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    library.save(output.with_name("COLLECTIVE_V18_LIBRARY.json")); return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=64); parser.add_argument("--horizon", type=int, default=42)
    args = parser.parse_args(argv); report = run_collective_audit(args.output, tasks=args.tasks, horizon=args.horizon)
    print(json.dumps({name: {key: value for key, value in row.items() if key != "rows"}
                      for name, row in report["aggregate"].items()} |
                     {"effect": report["effect"], "pass": report["precommitted_pass"],
                      "library": report["exchange"]["status_after_target"]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
