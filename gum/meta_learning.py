"""A small strategy-evolving learner for testing learning-to-learn.

Exact task solutions are never retained.  Across tasks, only a three-weight
best-first acquisition policy is selected and mutated from measured efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


FAMILIES = ("toggle", "counter", "rotate", "permutation")


def _hash_frame(frame): return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()[:20]


class LearningPuzzle:
    """Pixel-only goal puzzle with unique dynamics, controls, and presentation."""

    def __init__(self, seed: int, family: str):
        if family not in FAMILIES: raise ValueError(family)
        self.seed = int(seed); self.family = family; rng = np.random.default_rng(seed)
        self.n = int(rng.integers(4, 7)); self.palette = rng.integers(55, 246, (self.n, 3), dtype=np.uint8)
        self.slots = rng.permutation(self.n); self.background = rng.integers(8, 42, 3, dtype=np.uint8)
        if family == "toggle":
            self.base = np.zeros(self.n, dtype=np.int16); operations = []
            for index in range(self.n):
                mask = np.zeros(self.n, dtype=np.int16); mask[index] = 1
                if index % 2 and self.n > 4: mask[(index + 2) % self.n] = 1
                operations.append(("toggle", mask))
        elif family == "counter":
            self.base = np.zeros(self.n, dtype=np.int16); operations = []
            for index in range(self.n):
                delta = np.zeros(self.n, dtype=np.int16); delta[index] = 1 + index % 2
                operations.append(("counter", delta))
        elif family == "rotate":
            self.base = np.arange(self.n, dtype=np.int16); operations = [("rotate", amount) for amount in (-2, -1, 1, 2)]
        else:
            self.base = np.arange(self.n, dtype=np.int16); pairs = []
            for index in range(self.n): pairs.append((index, (index + 1) % self.n))
            if self.n >= 5: pairs.extend(((0, 2), (1, 3)))
            operations = [("swap", pair) for pair in pairs]
        order = rng.permutation(len(operations)); self.operations = [operations[int(i)] for i in order]
        self.action_count = len(self.operations)
        # The goal is generated after controls and presentation are fixed, but is exposed only as pixels.
        self.goal = self.base.copy(); secret = rng.integers(0, self.action_count, int(rng.integers(4, 8)))
        for action in secret: self.goal = self._apply_to(self.goal, int(action))
        if np.array_equal(self.goal, self.base): self.goal = self._apply_to(self.goal, 0)
        self.task_id = hashlib.sha256(f"{seed}:{family}:{self.n}".encode()).hexdigest()[:16]
        self.reset()

    def _apply_to(self, state, action):
        state = state.copy(); kind, value = self.operations[int(action)]
        if kind == "toggle": state ^= value
        elif kind == "counter": state = (state + value) % 3
        elif kind == "rotate": state = np.roll(state, int(value))
        else:
            left, right = value; state[left], state[right] = state[right], state[left]
        return state

    def reset(self): self.state = self.base.copy(); return self.render()

    def step(self, action):
        self.state = self._apply_to(self.state, int(action)); success = bool(np.array_equal(self.state, self.goal))
        return self.render(), (1.0 if success else 0.0), success, {"success": success}

    @property
    def goal_frame(self):
        current = self.state.copy(); self.state = self.goal.copy(); frame = self.render(); self.state = current; return frame

    def render(self):
        frame = np.zeros((3, 32, 32), dtype=np.uint8); frame[:] = self.background[:, None, None]
        visible = self.state[self.slots]
        for position, value in enumerate(visible):
            row, col = divmod(position, 3); y, x = 3 + row * 13, 3 + col * 10
            color = self.palette[int(value) % len(self.palette)]
            frame[:, y:y + 7, x:x + 7] = color[:, None, None]
        return frame

    def audit(self):
        return {"task_id": self.task_id, "family": self.family, "seed": self.seed, "variables": self.n,
                "actions": self.action_count, "goal_state": self.goal.tolist(),
                "operations": [(kind, value.tolist() if hasattr(value, "tolist") else list(value) if isinstance(value, tuple) else value)
                               for kind, value in self.operations]}


@dataclass
class StrategyGenome:
    genome_id: str
    weights: np.ndarray
    total_cost: float = 0.0
    evaluations: int = 0
    parent: str | None = None

    @property
    def mean_cost(self): return self.total_cost / self.evaluations if self.evaluations else float("inf")

    def to_json(self):
        return {"genome_id": self.genome_id, "weights": self.weights.tolist(), "total_cost": self.total_cost,
                "evaluations": self.evaluations, "mean_cost": self.mean_cost, "parent": self.parent}


def solve_with_strategy(world: LearningPuzzle, weights, budget=1800):
    """Generic best-first state discovery using only pixels, actions, and goal pixels."""
    weights = np.asarray(weights, dtype=float); start = world.reset(); goal = world.goal_frame
    nodes = [{"frame": start, "path": (), "tried": set(), "order": 0}]
    known = {_hash_frame(start)}; interactions = expansions = 0; order = 1
    while interactions < budget:
        candidates = []
        for index, node in enumerate(nodes):
            if len(node["tried"]) >= world.action_count: continue
            distance = float(np.mean(np.abs(node["frame"].astype(float) - goal.astype(float))) / 255.0)
            depth = len(node["path"]) / 12.0; exhausted = len(node["tried"]) / world.action_count
            score = weights[0] * distance + weights[1] * depth + weights[2] * exhausted
            candidates.append((score, node["order"], index))
        if not candidates: break
        _, _, index = min(candidates); node = nodes[index]
        available = [action for action in range(world.action_count) if action not in node["tried"]]
        action = available[0]; node["tried"].add(action); observation = world.reset()
        aborted = False
        for replay in node["path"]:
            observation, reward, done, _ = world.step(replay); interactions += 1
            if done: return {"success": True, "interactions": interactions, "expansions": expansions,
                             "solution_length": len(node["path"]), "solution": list(node["path"])}
            if interactions >= budget: aborted = True; break
        if aborted: break
        observation, reward, done, _ = world.step(action); interactions += 1; expansions += 1
        path = node["path"] + (action,)
        if done: return {"success": True, "interactions": interactions, "expansions": expansions,
                         "solution_length": len(path), "solution": list(path)}
        key = _hash_frame(observation)
        if key not in known:
            known.add(key); nodes.append({"frame": observation, "path": path, "tried": set(), "order": order}); order += 1
    return {"success": False, "interactions": interactions, "expansions": expansions,
            "solution_length": None, "solution": None}


class MetaLearningMind:
    """Evolves its acquisition strategy while deliberately forgetting task solutions."""

    format = "gum-meta-learning-mind-v1"

    def __init__(self, seed=991, population=18):
        self.seed = int(seed); self.rng = np.random.default_rng(seed); self.population_size = int(population)
        self.initial_weights = np.zeros(3, dtype=float)
        self.population = [self._make(self.initial_weights, None)]
        for _ in range(self.population_size - 1): self.population.append(self._make(self.rng.normal(0, 1.2, 3), None))
        self.tasks_seen = 0; self.learning_curve = []; self.evolution_events = []; self.solutions_retained = 0

    def _make(self, weights, parent):
        rounded = np.round(np.asarray(weights, dtype=float), 8)
        genome_id = "strategy-" + hashlib.sha256(rounded.tobytes()).hexdigest()[:12]
        return StrategyGenome(genome_id, rounded, parent=parent)

    def champion(self):
        evaluated = [row for row in self.population if row.evaluations]
        return min(evaluated, key=lambda row: (row.mean_cost, row.genome_id)) if evaluated else self.population[0]

    def learn_task(self, seed: int, family: str, budget=1800):
        before = self.champion(); pre = solve_with_strategy(LearningPuzzle(seed, family), before.weights, budget)
        initial = solve_with_strategy(LearningPuzzle(seed, family), self.initial_weights, budget)
        evaluations = []
        for genome in self.population:
            result = solve_with_strategy(LearningPuzzle(seed, family), genome.weights, budget)
            cost = result["interactions"] if result["success"] else budget * 1.5
            genome.total_cost += cost; genome.evaluations += 1
            evaluations.append({"genome_id": genome.genome_id, "cost": cost, "success": result["success"]})
        self.tasks_seen += 1
        if self.tasks_seen % 3 == 0: self._evolve()
        after = self.champion()
        row = {"task_index": self.tasks_seen, "task_id": LearningPuzzle(seed, family).task_id,
               "family_audit_only": family, "pre_update_strategy": before.genome_id,
               "pre_update_success": pre["success"], "pre_update_interactions": pre["interactions"],
               "initial_strategy_interactions": initial["interactions"],
               "relative_efficiency": initial["interactions"] / max(1, pre["interactions"]),
               "post_update_champion": after.genome_id, "solutions_retained": self.solutions_retained,
               "population_successes": sum(item["success"] for item in evaluations)}
        self.learning_curve.append(row); return row

    def _evolve(self):
        ranked = sorted(self.population, key=lambda row: (row.mean_cost, row.genome_id)); elites = ranked[:4]
        next_population = [StrategyGenome(row.genome_id, row.weights.copy(), row.total_cost, row.evaluations, row.parent)
                           for row in elites]
        while len(next_population) < self.population_size:
            parent = elites[(len(next_population) - len(elites)) % len(elites)]
            scale = max(0.08, 0.55 * (0.88 ** (self.tasks_seen // 3)))
            child = self._make(parent.weights + self.rng.normal(0, scale, 3), parent.genome_id)
            # Prior evidence is discounted rather than inherited as a task solution.
            child.total_cost = parent.total_cost * 0.25; child.evaluations = max(1, parent.evaluations // 4)
            next_population.append(child)
        previous = self.champion().genome_id; self.population = next_population
        self.evolution_events.append({"after_tasks": self.tasks_seen, "previous_champion": previous,
                                      "new_population": [row.genome_id for row in self.population]})

    def evaluate(self, specs, budget=1800):
        champion = self.champion(); rows = []
        for seed, family in specs:
            evolved = solve_with_strategy(LearningPuzzle(seed, family), champion.weights, budget)
            initial = solve_with_strategy(LearningPuzzle(seed, family), self.initial_weights, budget)
            shuffled = solve_with_strategy(LearningPuzzle(seed, family), champion.weights[::-1], budget)
            rows.append({"task_id": LearningPuzzle(seed, family).task_id, "family_audit_only": family,
                         "evolved": evolved, "initial": initial, "shuffled": shuffled})
        return {"champion": champion.to_json(), "rows": rows}

    def status(self):
        return {"format": self.format, "tasks_seen": self.tasks_seen, "solutions_retained": self.solutions_retained,
                "champion": self.champion().to_json(), "population": [row.to_json() for row in self.population],
                "learning_curve": self.learning_curve, "evolution_events": self.evolution_events}

    def save(self, path: Path):
        value = self.status() | {"seed": self.seed, "population_size": self.population_size,
                                 "initial_weights": self.initial_weights.tolist(),
                                 "rng_state": self.rng.bit_generator.state}
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        value = json.loads(Path(path).read_text(encoding="utf-8")); result = cls(value["seed"], value["population_size"])
        result.initial_weights = np.asarray(value["initial_weights"], dtype=float); result.tasks_seen = int(value["tasks_seen"])
        result.solutions_retained = int(value["solutions_retained"]); result.learning_curve = list(value["learning_curve"])
        result.evolution_events = list(value["evolution_events"]); result.population = []
        for row in value["population"]:
            result.population.append(StrategyGenome(row["genome_id"], np.asarray(row["weights"], dtype=float),
                                                     float(row["total_cost"]), int(row["evaluations"]), row["parent"]))
        result.rng.bit_generator.state = value["rng_state"]; return result
