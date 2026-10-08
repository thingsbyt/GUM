"""Reward-independent causal experiments and novelty-frontier memory."""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np


def _delta_signature(before: np.ndarray, after: np.ndarray) -> str:
    changed = before != after
    count = int(changed.sum())
    if not count: return "none"
    ys, xs = np.nonzero(changed)
    height = int(ys.max() - ys.min() + 1); width = int(xs.max() - xs.min() + 1)
    flows = Counter(zip(before[changed].tolist(), after[changed].tolist()))
    top = ",".join(f"{a}>{b}:{min(7, int(math.log2(n + 1)))}"
                   for (a, b), n in flows.most_common(4))
    return f"n{min(15, int(math.log2(count + 1)))}:w{min(15, width//2)}:h{min(15, height//2)}:{top}"


class CausalDiscoveryEngine:
    """Separate controllable consequences from animation and retain frontiers.

    No task labels or goals are used. The engine treats actions as experiments,
    estimates their effect/novelty/hazard, and archives action prefixes that
    reach new visual states. Later episodes replay a frontier and branch from it.
    """
    format = "wailah-causal-discovery-v1"

    def __init__(self):
        self.action_attempts = Counter(); self.action_effect = Counter()
        self.action_novelty = Counter(); self.action_hazard = Counter()
        self.state_action_attempts = defaultdict(Counter)
        self.delta_actions = defaultdict(set); self.autonomous_deltas = set()
        self.known_states = set(); self.frontiers = {}
        self.successful_paths = []
        self.current_path = []; self.replay_path = []; self.replay_index = 0
        self.active_success = None; self.active_frontier = None
        self.frontier_replays = 0; self.frontier_aborts = 0
        self.success_replays = 0; self.success_aborts = 0
        self.causal_effects = 0; self.autonomous_effects = 0
        self.failure_lengths = []
        self.last_token = None

    def reset_episode(self) -> None:
        self.current_path = []; self.replay_path = []; self.replay_index = 0
        self.last_token = None; self.active_success = None; self.active_frontier = None
        reliable = [row for row in self.successful_paths
                    if row.get("successes", 1) / max(1, row.get("successes", 1) + row.get("failures", 0)) >= .35]
        if reliable:
            self.active_success = max(reliable, key=lambda row: (
                row.get("successes", 1) - row.get("failures", 0), -len(row["path"])))
            self.replay_path = list(self.active_success["path"])
            return
        candidates = [row for row in self.frontiers.values() if not row.get("hazard", False)]
        if candidates:
            if self.failure_lengths:
                replay_budget = max(1, int(float(np.median(self.failure_lengths[-32:])) * .55))
                budgeted = [row for row in candidates if row["depth"] <= replay_budget]
                if budgeted: candidates = budgeted
            selected = max(candidates, key=lambda row: (
                row.get("novelty", 0) / math.sqrt(1 + row.get("replays", 0)) + .08 * row["depth"],
                -row.get("replays", 0), row["depth"]))
            self.active_frontier = selected
            self.replay_path = list(selected["path"]); selected["replays"] = selected.get("replays", 0) + 1

    def replay(self, state: str, actions: list[int]):
        if self.replay_index >= len(self.replay_path): return None
        row = self.replay_path[self.replay_index]
        if row["state"] != state or int(row["action"]) not in actions:
            if self.active_success is not None:
                self.active_success["failures"] = self.active_success.get("failures", 0) + 1
                self.success_aborts += 1; self.active_success = None
            else:
                self.frontier_aborts += 1
                if self.active_frontier is not None:
                    self.active_frontier["failures"] = self.active_frontier.get("failures", 0) + 1
                self.active_frontier = None
            self.replay_path = []; self.replay_index = 0
            return None
        self.replay_index += 1
        if self.active_success is not None: self.success_replays += 1
        else: self.frontier_replays += 1
        reason = ("replay-successful-coordinate-procedure" if self.active_success is not None
                  else "replay-novelty-frontier")
        return int(row["action"]), row.get("data"), reason

    def experiment_action(self, state: str, actions: list[int]) -> int:
        scored = []
        for action in actions:
            attempts = self.action_attempts[action]
            local = self.state_action_attempts[state][action]
            effect = self.action_effect[action] / max(1, attempts)
            novelty = self.action_novelty[action] / max(1, attempts)
            hazard = self.action_hazard[action] / max(1, attempts)
            information = 3.0 / math.sqrt(1 + local) + 1.0 / math.sqrt(1 + attempts)
            # Visible motion is evidence, not a goal. Keep its bonus modest so
            # delayed, initially inert controls continue receiving experiments.
            scored.append((information + .75 * effect + 2.0 * novelty - 8.0 * hazard,
                           -local, -attempts, -action, action))
        return int(max(scored)[-1])

    def note_action(self, state: str, action: int, data: dict | None = None) -> None:
        self.last_token = {"state": state, "action": int(action),
                           "data": ({"x": int(data["x"]), "y": int(data["y"])} if data else None)}

    def observe(self, state: str, action: int, next_state: str, before: np.ndarray,
                after: np.ndarray, progress: bool, done: bool) -> None:
        token = self.last_token or {"state": state, "action": int(action), "data": None}
        self.action_attempts[int(action)] += 1; self.state_action_attempts[state][int(action)] += 1
        signature = _delta_signature(before, after)
        if signature != "none":
            self.delta_actions[signature].add(int(action))
            if len(self.delta_actions[signature]) >= 2: self.autonomous_deltas.add(signature)
            if signature in self.autonomous_deltas: self.autonomous_effects += 1
            else: self.action_effect[int(action)] += 1; self.causal_effects += 1
        novel = next_state not in self.known_states
        self.known_states.add(state); self.known_states.add(next_state)
        if novel: self.action_novelty[int(action)] += 1
        if done and not progress: self.action_hazard[int(action)] += 1
        step = dict(token); step["next_state"] = next_state
        self.current_path.append(step)
        if progress:
            if self.active_success is not None:
                self.active_success["successes"] = self.active_success.get("successes", 1) + 1
            if self.current_path and not any(row["path"] == self.current_path for row in self.successful_paths):
                self.successful_paths.append({"path": list(self.current_path), "successes": 1, "failures": 0})
        if novel and len(self.current_path) <= 128:
            existing = self.frontiers.get(next_state)
            if existing is None or len(self.current_path) < existing["depth"]:
                self.frontiers[next_state] = {"path": list(self.current_path),
                                              "depth": len(self.current_path),
                                              "novelty": 1, "replays": 0,
                                              "hazard": bool(done and not progress)}
        if done and not progress:
            self.failure_lengths.append(len(self.current_path)); self.failure_lengths = self.failure_lengths[-64:]
            if self.active_success is not None:
                self.active_success["failures"] = self.active_success.get("failures", 0) + 1
            if self.active_frontier is not None:
                self.active_frontier["failures"] = self.active_frontier.get("failures", 0) + 1
                self.active_frontier["hazard"] = True
            for row in self.frontiers.values():
                if row["path"] == self.current_path: row["hazard"] = True
        self.last_token = None

    def status(self) -> dict:
        return {"format": self.format, "known_states": len(self.known_states),
                "novelty_frontiers": len(self.frontiers), "frontier_replays": self.frontier_replays,
                "frontier_aborts": self.frontier_aborts, "causal_effects": self.causal_effects,
                "successful_coordinate_procedures": len(self.successful_paths),
                "successful_procedure_replays": self.success_replays,
                "successful_procedure_aborts": self.success_aborts,
                "hazardous_frontiers": sum(bool(row.get("hazard")) for row in self.frontiers.values()),
                "median_failed_episode_length": (float(np.median(self.failure_lengths))
                                                  if self.failure_lengths else 0.0),
                "autonomous_effects": self.autonomous_effects,
                "action_attempts": dict(self.action_attempts),
                "action_hazards": dict(self.action_hazard)}

    def export(self) -> dict:
        return self.status() | {
            "action_effect": dict(self.action_effect), "action_novelty": dict(self.action_novelty),
            "state_action_attempts": {state: dict(rows) for state, rows in self.state_action_attempts.items()},
            "delta_actions": {key: sorted(rows) for key, rows in self.delta_actions.items()},
            "autonomous_deltas": sorted(self.autonomous_deltas), "known_states_data": sorted(self.known_states),
            "frontiers": self.frontiers, "successful_paths": self.successful_paths,
            "failure_lengths": self.failure_lengths}

    def restore(self, value: dict) -> None:
        if value.get("format") != self.format: raise ValueError("unsupported causal discovery format")
        def ints(name): return Counter({int(k): int(v) for k, v in value.get(name, {}).items()})
        self.action_attempts = ints("action_attempts"); self.action_effect = ints("action_effect")
        self.action_novelty = ints("action_novelty"); self.action_hazard = ints("action_hazards")
        self.state_action_attempts = defaultdict(Counter, {
            state: Counter({int(k): int(v) for k, v in rows.items()})
            for state, rows in value.get("state_action_attempts", {}).items()})
        self.delta_actions = defaultdict(set, {key: set(map(int, rows))
                                               for key, rows in value.get("delta_actions", {}).items()})
        self.autonomous_deltas = set(value.get("autonomous_deltas", []))
        self.known_states = set(value.get("known_states_data", []))
        self.frontiers = dict(value.get("frontiers", {}))
        self.successful_paths = list(value.get("successful_paths", []))
        self.failure_lengths = [int(x) for x in value.get("failure_lengths", [])]
        self.frontier_replays = int(value.get("frontier_replays", 0))
        self.frontier_aborts = int(value.get("frontier_aborts", 0))
        self.success_replays = int(value.get("successful_procedure_replays", 0))
        self.success_aborts = int(value.get("successful_procedure_aborts", 0))
        self.causal_effects = int(value.get("causal_effects", 0))
        self.autonomous_effects = int(value.get("autonomous_effects", 0))
