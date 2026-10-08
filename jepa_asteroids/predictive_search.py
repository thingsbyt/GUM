"""Compact action-conditioned world modeling and multi-step visual search.

The planner is deliberately task-agnostic. It receives rendered categorical
pixels, legal action IDs, sparse progress, and termination. It learns a compact
visual transition model online and searches predicted futures for either
previously rewarded feature changes or informative novel states.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import math

import numpy as np


def predictive_features(grid: np.ndarray) -> np.ndarray:
    """Color-aware spatial summary used as the learned model's state."""
    grid = np.asarray(grid, dtype=np.uint8)
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    occupied = grid != background
    height, width = grid.shape
    ys, xs = np.nonzero(occupied)
    if len(xs):
        cx = float(xs.mean()) / max(1, width - 1)
        cy = float(ys.mean()) / max(1, height - 1)
        sx = float(xs.std()) / max(1, width - 1)
        sy = float(ys.std()) / max(1, height - 1)
    else:
        cx = cy = sx = sy = 0.0
    horizontal = float(np.mean(grid[:, 1:] != grid[:, :-1])) if width > 1 else 0.0
    vertical = float(np.mean(grid[1:, :] != grid[:-1, :])) if height > 1 else 0.0
    probabilities = counts.astype(float) / max(1, grid.size)
    entropy = float(-(probabilities * np.log2(probabilities + 1e-12)).sum()) / 4.0
    histogram = np.bincount(grid.ravel(), minlength=16)[:16].astype(float) / max(1, grid.size)
    pooled = []
    for y0, y1 in zip(np.linspace(0, height, 5, dtype=int)[:-1],
                      np.linspace(0, height, 5, dtype=int)[1:]):
        for x0, x1 in zip(np.linspace(0, width, 5, dtype=int)[:-1],
                          np.linspace(0, width, 5, dtype=int)[1:]):
            pooled.append(float(np.mean(occupied[y0:y1, x0:x1])))
    return np.asarray([float(occupied.mean()), horizontal, vertical, entropy,
                       cx, cy, sx, sy] + histogram.tolist() + pooled, dtype=float)


def feature_key(features: np.ndarray) -> str:
    """Noise-tolerant identity for model graph nodes."""
    vector = np.asarray(features, dtype=float)
    # Geometry receives finer bins than color proportions and pooled density.
    quantized = np.rint(vector * 16.0).astype(np.int16).tolist()
    payload = json.dumps(quantized, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


class PredictiveSearchPlanner:
    """Online compact world model with bounded latent-space beam search."""

    format = "wailah-predictive-search-v1"

    def __init__(self):
        self.state_visits = Counter()
        self.action_visits = defaultdict(Counter)
        self.transitions = defaultdict(Counter)
        self.local_deltas = defaultdict(list)
        self.global_deltas = defaultdict(list)
        self.q = defaultdict(dict)
        self.terminal_risk = Counter()
        self.progress_direction = None
        self.progress_examples = 0
        self.episode = []
        self.observations = 0
        self.search_steps = 0
        self.progress_search_steps = 0
        self.frontier_search_steps = 0
        self.graph_plan_steps = 0
        self.last_plan = None

    def reset_episode(self) -> None:
        self.episode = []
        self.last_plan = None

    @staticmethod
    def _edge(state: str, action: int) -> str:
        return f"{state}:{int(action)}"

    def _graph_frontier(self, start: str, actions: list[int]) -> tuple[int, list[int]] | None:
        """Find a known route to a compact state with an under-tested action."""
        queue = deque([start]); previous = {start: None}; via = {}
        target = None
        while queue and len(previous) <= 1200:
            state = queue.popleft()
            tried = self.action_visits[state]
            if any(tried[action] < 2 for action in actions):
                target = state; break
            for action in actions:
                outcomes = self.transitions.get(self._edge(state, action))
                if not outcomes: continue
                nxt = outcomes.most_common(1)[0][0]
                if nxt not in previous:
                    previous[nxt] = state; via[nxt] = action; queue.append(nxt)
        if target is None:
            return None
        if target == start:
            least = min(actions, key=lambda action: (self.action_visits[start][action], action))
            return int(least), [int(least)]
        path = []; cursor = target
        while previous[cursor] is not None:
            path.append(int(via[cursor])); cursor = previous[cursor]
        path.reverse()
        return path[0], path

    def _predicted_delta(self, state: str, action: int):
        local = self.local_deltas.get(self._edge(state, action), [])[-32:]
        rows = local if len(local) >= 2 else self.global_deltas.get(int(action), [])[-64:]
        if len(rows) < 2:
            return None
        matrix = np.asarray(rows, dtype=float)
        return np.median(matrix, axis=0), float(np.mean(np.std(matrix, axis=0)))

    def _beam_plan(self, features: np.ndarray, actions: list[int]):
        start = feature_key(features)
        # Each row is score, predicted features, first action, sequence.
        beam = [(0.0, np.asarray(features, dtype=float), None, [])]
        best = None
        goal = self.progress_direction
        goal_norm = float(np.linalg.norm(goal)) if goal is not None else 0.0
        for depth in range(1, 7):
            expanded = []
            for score, vector, first, sequence in beam:
                state = feature_key(vector)
                for action in actions:
                    prediction = self._predicted_delta(state, action)
                    if prediction is None:
                        continue
                    delta, uncertainty = prediction
                    nxt = np.clip(vector + delta, 0.0, 1.0)
                    next_state = feature_key(nxt)
                    visits = self.action_visits[state][action]
                    novelty = 1.0 / math.sqrt(1.0 + self.state_visits[next_state])
                    information = uncertainty / math.sqrt(1.0 + visits)
                    change = min(1.0, float(np.linalg.norm(delta)) * 4.0)
                    risk = self.terminal_risk[self._edge(state, action)] / max(1, visits)
                    learned_value = self.q[state].get(action, 0.0)
                    progress_score = 0.0
                    if goal_norm > 1e-9:
                        progress_score = float(np.dot(delta, goal) /
                                               (np.linalg.norm(delta) * goal_norm + 1e-9))
                    # Before any outcome, seek information and novel reachable
                    # states. Once sparse progress reveals a direction in the
                    # learned representation, exploitation must outweigh raw
                    # novelty or the planner can prefer moving away merely
                    # because the wrong side of the screen is less familiar.
                    novelty_weight = .8 if self.progress_examples else 2.4
                    information_weight = .6 if self.progress_examples else 1.2
                    progress_weight = 4.0 if self.progress_examples else 0.0
                    step_score = (novelty_weight * novelty + information_weight * information
                                  + .5 * change + progress_weight * progress_score
                                  + .08 * learned_value - 5.0 * risk)
                    total = score + (.86 ** (depth - 1)) * step_score
                    row = (total, nxt, action if first is None else first,
                           sequence + [int(action)])
                    expanded.append(row)
                    if best is None or row[0] > best[0]: best = row
            if not expanded:
                break
            expanded.sort(key=lambda row: (-row[0], len(row[3]), row[3]))
            beam = expanded[:48]
        if best is None:
            return None
        reason = ("predictive-search-progress" if self.progress_examples else
                  "predictive-search-frontier")
        return int(best[2]), reason, best[3], float(best[0]), start

    def recommend(self, grid: np.ndarray, actions: list[int]):
        actions = sorted(map(int, actions))
        if not actions or self.observations < max(12, 2 * len(actions)):
            return None
        features = predictive_features(grid); state = feature_key(features)
        # Rewarded compact states receive first priority through learned Q.
        if self.q.get(state):
            ranked = [(value, -self.terminal_risk[self._edge(state, action)], -action, action)
                      for action, value in self.q[state].items() if action in actions]
            # Tiny positive values are intrinsic novelty bonuses, not evidence
            # of task progress. Only let the reward cache short-circuit search
            # after a value large enough to require an observed outcome.
            if ranked and max(row[0] for row in ranked) > 1.0:
                action = int(max(ranked)[3]); self.search_steps += 1
                self.progress_search_steps += 1
                self.last_plan = {"reason": "predictive-search-rewarded-state",
                                  "sequence": [action], "state": state}
                return action, "predictive-search-rewarded-state"
        rollout = self._beam_plan(features, actions)
        graph = self._graph_frontier(state, actions)
        if rollout is not None:
            action, reason, sequence, score, _ = rollout
            self.search_steps += 1
            if self.progress_examples: self.progress_search_steps += 1
            else: self.frontier_search_steps += 1
            self.last_plan = {"reason": reason, "sequence": sequence,
                              "score": score, "state": state}
            return action, reason
        if graph is not None:
            action, sequence = graph; self.search_steps += 1; self.graph_plan_steps += 1
            self.last_plan = {"reason": "predictive-graph-frontier",
                              "sequence": sequence, "state": state}
            return int(action), "predictive-graph-frontier"
        return None

    def observe(self, before: np.ndarray, action: int, after: np.ndarray,
                progress: bool, done: bool) -> None:
        before_features = predictive_features(before); after_features = predictive_features(after)
        state, nxt = feature_key(before_features), feature_key(after_features)
        action = int(action); edge = self._edge(state, action)
        delta = after_features - before_features
        self.state_visits[state] += 1; self.state_visits[nxt] += 1
        self.action_visits[state][action] += 1
        self.transitions[edge][nxt] += 1
        self.local_deltas[edge].append(delta.tolist()); self.local_deltas[edge] = self.local_deltas[edge][-64:]
        self.global_deltas[action].append(delta.tolist()); self.global_deltas[action] = self.global_deltas[action][-128:]
        self.observations += 1
        if done and not progress:
            self.terminal_risk[edge] += 1
        future = max(self.q[nxt].values(), default=0.0) if not done else 0.0
        reward = 20.0 if progress else (-4.0 if done else 0.02 / math.sqrt(self.state_visits[nxt]))
        old = self.q[state].get(action, 0.0)
        self.q[state][action] = old + .3 * (reward + .96 * future - old)
        self.episode.append((state, action))
        if progress:
            if self.progress_direction is None:
                self.progress_direction = delta.copy()
            else:
                rate = 1.0 / (self.progress_examples + 1)
                self.progress_direction = (1.0 - rate) * self.progress_direction + rate * delta
            self.progress_examples += 1
            value = 20.0
            for prior_state, prior_action in reversed(self.episode):
                self.q[prior_state][prior_action] = max(self.q[prior_state].get(prior_action, 0.0), value)
                value *= .96
            self.episode = []
        elif done:
            value = -4.0
            for prior_state, prior_action in reversed(self.episode[-20:]):
                self.q[prior_state][prior_action] = min(self.q[prior_state].get(prior_action, 0.0), value)
                value *= .9
            self.episode = []

    def status(self) -> dict:
        return {"format": self.format, "observations": self.observations,
                "compact_states": len(self.state_visits), "modeled_edges": len(self.transitions),
                "search_steps": self.search_steps,
                "progress_search_steps": self.progress_search_steps,
                "frontier_search_steps": self.frontier_search_steps,
                "graph_plan_steps": self.graph_plan_steps,
                "progress_examples": self.progress_examples,
                "last_plan": self.last_plan}

    def export(self) -> dict:
        return self.status() | {
            "state_visits": dict(self.state_visits),
            "action_visits": {state: dict(rows) for state, rows in self.action_visits.items()},
            "transitions": {edge: dict(rows) for edge, rows in self.transitions.items()},
            "local_deltas": dict(self.local_deltas),
            "global_deltas": {str(action): rows for action, rows in self.global_deltas.items()},
            "q": {state: {str(action): value for action, value in rows.items()}
                  for state, rows in self.q.items()},
            "terminal_risk": dict(self.terminal_risk),
            "progress_direction": (None if self.progress_direction is None else
                                   self.progress_direction.tolist())}

    def restore(self, value: dict) -> None:
        if value.get("format") != self.format:
            raise ValueError("unsupported predictive search format")
        self.state_visits = Counter({str(k): int(v) for k, v in value.get("state_visits", {}).items()})
        self.action_visits = defaultdict(Counter, {str(state): Counter({int(a): int(n)
            for a, n in rows.items()}) for state, rows in value.get("action_visits", {}).items()})
        self.transitions = defaultdict(Counter, {str(edge): Counter(rows)
                                                   for edge, rows in value.get("transitions", {}).items()})
        self.local_deltas = defaultdict(list, {str(edge): [[float(x) for x in row] for row in rows]
                                                for edge, rows in value.get("local_deltas", {}).items()})
        self.global_deltas = defaultdict(list, {int(action): [[float(x) for x in row] for row in rows]
                                                 for action, rows in value.get("global_deltas", {}).items()})
        self.q = defaultdict(dict, {str(state): {int(action): float(score)
                                                  for action, score in rows.items()}
                                    for state, rows in value.get("q", {}).items()})
        self.terminal_risk = Counter({str(k): int(v) for k, v in value.get("terminal_risk", {}).items()})
        direction = value.get("progress_direction")
        self.progress_direction = None if direction is None else np.asarray(direction, dtype=float)
        self.observations = int(value.get("observations", 0))
        self.search_steps = int(value.get("search_steps", 0))
        self.progress_search_steps = int(value.get("progress_search_steps", 0))
        self.frontier_search_steps = int(value.get("frontier_search_steps", 0))
        self.graph_plan_steps = int(value.get("graph_plan_steps", 0))
        self.progress_examples = int(value.get("progress_examples", 0))
        self.last_plan = value.get("last_plan")
