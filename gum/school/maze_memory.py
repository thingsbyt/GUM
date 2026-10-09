"""Reward-gated episodic spatial memory for the Changing Maze school.

The visual parser, map store, path search, and anonymous-control hypothesis
space are fixed learner machinery.  Nothing supplies a hidden map, position,
goal coordinate, or action meaning.  During an episode the learner derives
all of those beliefs from pixel changes.  Across training episodes a tiny
two-armed controller learns, from scalar return only, whether to engage that
memory machinery.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Any

import numpy as np

from gum.protocol import PublicWorldSpec, Transition
from gum.storage import atomic_write_json

from .worlds import CHANGING_MAZE_ADAPTER


class MazeMemoryError(ValueError):
    pass


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NO_MOVEMENT = (0, 0)


def _shortest_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    open_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = parent[cell]
            return list(reversed(path))
        for dr, dc in _DIRECTIONS:
            nxt = cell[0] + dr, cell[1] + dc
            if nxt in open_cells and nxt not in parent:
                parent[nxt] = cell
                queue.append(nxt)
    return None


class MazeMemoryLearner:
    """A scalar-reward-gated visual map builder with online motor grounding."""

    format = "gum-school-maze-memory-v1"
    modes = ("reactive-exploration", "episodic-map")

    size = 11
    scale = 8
    unseen = (5, 8, 12)
    wall = (83, 92, 101)
    goal_color = (58, 211, 126)
    agent = (69, 181, 255)

    def __init__(self, seed: int, *, alpha: float = 0.5):
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.rng = np.random.default_rng(self.seed)
        self.mode_values = np.zeros(len(self.modes), dtype=np.float64)
        self.mode_visits = np.zeros(len(self.modes), dtype=np.int64)
        self.training_episodes = 0
        self.training_interactions = 0
        self.evaluation_episodes = 0
        self.evaluation_interactions = 0
        self.boundary_violations: list[str] = []
        self.spec: PublicWorldSpec | None = None
        self._active_mode = 0
        self._last_confidence = 0.0
        self._episode_return = 0.0
        self._episode_steps = 0
        self._last_action: int | None = None
        self._last_position: tuple[int, int] | None = None

    @staticmethod
    def _validate_observation(spec: PublicWorldSpec, observation: Any) -> np.ndarray:
        array = np.asarray(observation)
        if array.dtype != np.uint8 or array.shape != spec.observation_shape:
            raise MazeMemoryError("observation differs from the public RGB contract")
        return array

    def _cell_color(
        self, observation: np.ndarray, row: int, col: int
    ) -> tuple[int, int, int]:
        return tuple(
            int(value)
            for value in observation[
                row * self.scale + self.scale // 2,
                col * self.scale + self.scale // 2,
            ]
        )

    def _agent_position(self, observation: np.ndarray) -> tuple[int, int]:
        mask = np.all(observation == np.asarray(self.agent, dtype=np.uint8), axis=2)
        ys, xs = np.where(mask)
        if not len(xs):
            raise MazeMemoryError("could not locate the agent in public pixels")
        return int(np.mean(ys)) // self.scale, int(np.mean(xs)) // self.scale

    def _update_map(self, observation: np.ndarray) -> None:
        self.position = self._agent_position(observation)
        for row in range(self.size):
            for col in range(self.size):
                cell = row, col
                color = self._cell_color(observation, row, col)
                if color == self.unseen:
                    continue
                if color == self.wall:
                    self.walls.add(cell)
                    self.open_cells.discard(cell)
                else:
                    self.open_cells.add(cell)
                    self.walls.discard(cell)
                    if color == self.goal_color:
                        self.goal = cell

    def _reset_episode_memory(self, observation: np.ndarray, action_count: int) -> None:
        self.open_cells: set[tuple[int, int]] = set()
        self.walls: set[tuple[int, int]] = set()
        self.goal: tuple[int, int] | None = None
        labels = set(_DIRECTIONS) | {_NO_MOVEMENT}
        self.action_hypotheses = {
            action: set(labels) for action in range(action_count)
        }
        self.probes: set[tuple[tuple[int, int], int]] = set()
        self.position = (0, 0)
        self._update_map(observation)

    def _select_mode(self, training: bool) -> int:
        if training:
            untried = np.flatnonzero(self.mode_visits == 0)
            if len(untried):
                return int(untried[0])
            if self.rng.random() < 0.10:
                return int(self.rng.integers(len(self.modes)))
        finalists = np.flatnonzero(self.mode_values == np.max(self.mode_values))
        return int(finalists[0])

    def begin(self, spec: PublicWorldSpec, observation: Any, *, training: bool) -> None:
        if (
            spec.adapter != CHANGING_MAZE_ADAPTER
            or spec.observation_kind != "rgb"
            or spec.action_kind != "discrete-anonymous"
            or spec.action_count < 4
        ):
            raise MazeMemoryError("unsupported school perception or action contract")
        array = self._validate_observation(spec, observation)
        self.spec = spec
        self._active_mode = self._select_mode(training)
        self._last_confidence = 0.0
        self._episode_return = 0.0
        self._episode_steps = 0
        self._last_action = None
        self._last_position = None
        self._reset_episode_memory(array, spec.action_count)

    def _propagate_hypotheses(self) -> None:
        changed = True
        while changed:
            changed = False
            assigned = {
                next(iter(options))
                for options in self.action_hypotheses.values()
                if len(options) == 1 and next(iter(options)) != _NO_MOVEMENT
            }
            for options in self.action_hypotheses.values():
                if len(options) == 1:
                    continue
                reduced = options - assigned
                if reduced and reduced != options:
                    options.intersection_update(reduced)
                    changed = True
            for direction in _DIRECTIONS:
                possible = [
                    action
                    for action, options in self.action_hypotheses.items()
                    if direction in options
                ]
                if len(possible) == 1 and self.action_hypotheses[possible[0]] != {direction}:
                    self.action_hypotheses[possible[0]] = {direction}
                    changed = True
            if len(assigned) == len(_DIRECTIONS):
                for options in self.action_hypotheses.values():
                    if len(options) > 1 and options != {_NO_MOVEMENT}:
                        options.intersection_update({_NO_MOVEMENT})
                        changed = True

    def _desired_direction(self) -> tuple[int, int]:
        if self.goal is not None:
            path = _shortest_path(self.position, self.goal, self.open_cells)
            if path and len(path) > 1:
                return (
                    path[1][0] - self.position[0],
                    path[1][1] - self.position[1],
                )

        reachable = []
        for cell in self.open_cells:
            path = _shortest_path(self.position, cell, self.open_cells)
            if path is None:
                continue
            has_unknown_neighbor = any(
                (cell[0] + dr, cell[1] + dc) not in self.open_cells
                and (cell[0] + dr, cell[1] + dc) not in self.walls
                for dr, dc in _DIRECTIONS
            )
            if has_unknown_neighbor:
                reachable.append((len(path), cell, path))
        if reachable:
            _, _, path = min(reachable, key=lambda row: (row[0], row[1]))
            if len(path) > 1:
                return (
                    path[1][0] - self.position[0],
                    path[1][1] - self.position[1],
                )

        safe = [
            direction
            for direction in _DIRECTIONS
            if (self.position[0] + direction[0], self.position[1] + direction[1])
            in self.open_cells
        ]
        return safe[0] if safe else _DIRECTIONS[0]

    def _memory_action(self) -> int:
        assert self.spec is not None
        desired = self._desired_direction()
        exact = [
            action
            for action, options in self.action_hypotheses.items()
            if options == {desired}
        ]
        if exact:
            self._last_confidence = 1.0
            return int(exact[0])

        candidates = [
            action
            for action, options in self.action_hypotheses.items()
            if desired in options and (self.position, action) not in self.probes
        ]
        if not candidates:
            candidates = [
                action
                for action, options in self.action_hypotheses.items()
                if desired in options
            ]
        if not candidates:
            # Contradictory evidence should not make the public interface fail.
            self.action_hypotheses = {
                action: set(_DIRECTIONS) | {_NO_MOVEMENT}
                for action in range(self.spec.action_count)
            }
            candidates = list(range(self.spec.action_count))
        action = min(candidates, key=lambda item: (len(self.action_hypotheses[item]), item))
        self.probes.add((self.position, action))
        self._last_confidence = 0.0
        return int(action)

    def act(self, observation: Any, *, training: bool) -> int:
        if self.spec is None:
            raise MazeMemoryError("begin must be called before act")
        array = self._validate_observation(self.spec, observation)
        self._update_map(array)
        self._last_position = self.position
        if self._active_mode == 1:
            action = self._memory_action()
        else:
            action = int(self.rng.integers(self.spec.action_count))
            self._last_confidence = 0.0
        self._last_action = action
        return action

    def observe(self, action: Any, transition: Transition, *, training: bool) -> None:
        if self.spec is None or self._last_position is None:
            raise MazeMemoryError("begin and act must precede observe")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise MazeMemoryError("action must be an integer slot")
        action = int(action)
        if not 0 <= action < self.spec.action_count:
            raise MazeMemoryError("action is outside the public contract")
        if not isinstance(transition.reward, (int, float)):
            raise MazeMemoryError("reward must be numeric")
        array = self._validate_observation(self.spec, transition.observation)
        previous = self._last_position
        self._update_map(array)
        delta = self.position[0] - previous[0], self.position[1] - previous[1]
        if self._active_mode == 1:
            if delta in _DIRECTIONS:
                self.action_hypotheses[action] = {delta}
            elif delta == _NO_MOVEMENT:
                demonstrably_open = {
                    direction
                    for direction in _DIRECTIONS
                    if (previous[0] + direction[0], previous[1] + direction[1])
                    in self.open_cells
                }
                remaining = self.action_hypotheses[action] - demonstrably_open
                self.action_hypotheses[action] = remaining or {_NO_MOVEMENT}
            self._propagate_hypotheses()
        self._episode_return += float(transition.reward)
        self._episode_steps += 1
        if training:
            self.training_interactions += 1
        else:
            self.evaluation_interactions += 1

    def finish_episode(self, *, training: bool) -> None:
        if training:
            value = self.mode_values[self._active_mode]
            self.mode_values[self._active_mode] += self.alpha * (
                self._episode_return - value
            )
            self.mode_visits[self._active_mode] += 1
            self.training_episodes += 1
        else:
            self.evaluation_episodes += 1

    def confidence(self) -> float:
        return float(self._last_confidence)

    def status(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "seed": self.seed,
            "mode_values": self.mode_values.tolist(),
            "mode_visits": self.mode_visits.tolist(),
            "selected_mode": self.modes[int(np.argmax(self.mode_values))],
            "training_episodes": self.training_episodes,
            "training_interactions": self.training_interactions,
            "evaluation_episodes": self.evaluation_episodes,
            "evaluation_interactions": self.evaluation_interactions,
            "boundary_violations": list(self.boundary_violations),
            "information_boundary": [
                "rgb-pixels",
                "anonymous-action-slot",
                "scalar-reward",
                "termination",
                "episodic-memory",
            ],
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "seed": self.seed,
            "alpha": self.alpha,
            "rng_state": self.rng.bit_generator.state,
            "mode_values": self.mode_values.tolist(),
            "mode_visits": self.mode_visits.tolist(),
            "training_episodes": self.training_episodes,
            "training_interactions": self.training_interactions,
            "evaluation_episodes": self.evaluation_episodes,
            "evaluation_interactions": self.evaluation_interactions,
            "boundary_violations": list(self.boundary_violations),
        }

    def save(self, path: Path) -> None:
        atomic_write_json(Path(path), self.to_json(), backup=False, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> "MazeMemoryLearner":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MazeMemoryError(f"maze learner state is unreadable: {error}") from error
        required = {
            "format", "seed", "alpha", "rng_state", "mode_values", "mode_visits",
            "training_episodes", "training_interactions", "evaluation_episodes",
            "evaluation_interactions", "boundary_violations",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise MazeMemoryError("maze learner fields differ from the reviewed format")
        if value["format"] != cls.format:
            raise MazeMemoryError("unsupported maze learner format")
        result = cls(value["seed"], alpha=value["alpha"])
        result.rng.bit_generator.state = value["rng_state"]
        result.mode_values = np.asarray(value["mode_values"], dtype=np.float64)
        result.mode_visits = np.asarray(value["mode_visits"], dtype=np.int64)
        result.training_episodes = int(value["training_episodes"])
        result.training_interactions = int(value["training_interactions"])
        result.evaluation_episodes = int(value["evaluation_episodes"])
        result.evaluation_interactions = int(value["evaluation_interactions"])
        result.boundary_violations = [str(item) for item in value["boundary_violations"]]
        if result.mode_values.shape != (2,) or result.mode_visits.shape != (2,):
            raise MazeMemoryError("maze controller dimensions differ from the reviewed format")
        return result
