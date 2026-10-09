"""Persistent cross-seed learner with bounded, on-demand policy replicas.

The learner keeps trainable state by reviewed adapter and observation-derived
memory state, never by world identity or seed. A single policy begins each run.
Sustained uncertainty may spawn helper policies up to the configured maximum;
helpers learn on separate episodes and periodically exchange value estimates.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gum.protocol import PublicWorldSpec, Transition
from gum.storage import atomic_write_json

from .worlds import (
    CAUSAL_WORKSHOP_ADAPTER,
    FOUNDATIONAL_ADAPTERS,
    OBJECT_LABORATORY_ADAPTER,
)


class SchoolLearnerError(ValueError):
    pass


class CrossSeedSchoolLearner:
    format = "gum-school-cross-seed-learner-v2"
    feature_grid = 4
    feature_count = feature_grid * feature_grid * 3 + 7
    allowed_public_info = frozenset({"event", "success"})
    replica_limit = 4
    swarm_alpha = 0.35
    spawn_uncertainty_threshold = 0.08
    spawn_patience = 3

    def __init__(
        self,
        seed: int = 8_620_001,
        *,
        alpha: float = 0.04,
        gamma: float = 0.96,
        max_replicas: int = 4,
        initial_replicas: int = 1,
    ):
        if isinstance(max_replicas, bool) or not isinstance(max_replicas, int):
            raise SchoolLearnerError("max_replicas must be an integer")
        if not 1 <= max_replicas <= self.replica_limit:
            raise SchoolLearnerError(f"max_replicas must be between 1 and {self.replica_limit}")
        if (isinstance(initial_replicas, bool) or not isinstance(initial_replicas, int)
                or not 1 <= initial_replicas <= max_replicas):
            raise SchoolLearnerError("initial_replicas must be between 1 and max_replicas")
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = 0.35
        self.rng = np.random.default_rng(self.seed)
        self.max_replicas = max_replicas
        self.replica_count = initial_replicas
        self.replica_epsilons = [0.45 for _ in range(initial_replicas)]
        self.replica_episodes = [0 for _ in range(initial_replicas)]
        self.weights: dict[str, np.ndarray] = {}
        self.swarm_values: dict[str, dict[str, np.ndarray]] = {}
        self.swarm_visits: dict[str, dict[str, np.ndarray]] = {}
        self.action_counts: dict[str, int] = {}
        self.adapter_steps: dict[str, int] = {}
        self.seen_worlds: set[str] = set()
        self.episodes = 0
        self.training_steps = 0
        self.evaluation_steps = 0
        self.communication_rounds = 0
        self.spawn_events: list[dict[str, Any]] = []
        self.boundary_violations: list[str] = []
        self.spec: PublicWorldSpec | None = None
        self.features: np.ndarray | None = None
        self.last_q: np.ndarray | None = None
        self.last_state_key: str | None = None
        self.active_replica = 0
        self.uncertainty_streak = 0
        self.episode_step = 0
        self._tracker_active = False
        self._track_colors: list[tuple[int, int, int]] = []
        self._track_x0 = np.zeros(0, dtype=np.float64)
        self._track_target_index: int | None = None
        self._predicted_slot: int | None = None
        self._function_active = False
        self._function_target_role: int | None = None
        self._function_roles: dict[int, int] = {}
        self._function_colors: list[tuple[int, int, int]] = []
        self._function_x0 = np.zeros(0, dtype=np.float64)
        self._function_identity_slots: tuple[int, ...] | None = None
        self._function_final_signatures: tuple[str, ...] = ()
        self._function_shift_started = False
        self._function_ready = False
        self._causal_active = False
        self._causal_tried: set[int] = set()
        self._causal_progress_action: int | None = None

    @staticmethod
    def _validate_observation(spec: PublicWorldSpec, observation: Any) -> np.ndarray:
        if not isinstance(observation, np.ndarray):
            raise SchoolLearnerError("school learner accepts only NumPy pixel observations")
        if observation.dtype != np.uint8 or tuple(observation.shape) != tuple(spec.observation_shape):
            raise SchoolLearnerError("observation differs from the public pixel contract")
        if observation.ndim != 3 or observation.shape[2] != 3:
            raise SchoolLearnerError("school learner requires RGB observations")
        return observation

    @classmethod
    def visual_features(cls, observation: np.ndarray) -> np.ndarray:
        height, width, _ = observation.shape
        rows = np.rint(np.linspace(0, height - 1, cls.feature_grid)).astype(int)
        cols = np.rint(np.linspace(0, width - 1, cls.feature_grid)).astype(int)
        sample = observation[np.ix_(rows, cols, np.arange(3))].astype(np.float64) / 255.0
        pixels = observation.astype(np.float64) / 255.0
        summary = np.concatenate((pixels.mean(axis=(0, 1)), pixels.std(axis=(0, 1))))
        return np.concatenate((sample.reshape(-1), summary, np.ones(1, dtype=np.float64)))

    @staticmethod
    def _color_components(observation: np.ndarray) -> list[dict[str, Any]]:
        colors, counts = np.unique(observation.reshape(-1, 3), axis=0, return_counts=True)
        result = []
        for color, count in zip(colors, counts):
            if int(color.max()) - int(color.min()) < 70 or int(count) < 40:
                continue
            mask = np.all(observation == color, axis=2)
            rows, columns = np.where(mask)
            result.append({
                "color": tuple(int(value) for value in color),
                "count": int(count),
                "x": float(np.median(columns)),
                "y": float(np.median(rows)),
                "width": int(columns.max() - columns.min() + 1),
                "height": int(rows.max() - rows.min() + 1),
            })
        return result

    def _initialize_tracker(self, observation: np.ndarray) -> None:
        self._tracker_active = False
        self._predicted_slot = None
        if self.spec is None or self.spec.adapter != OBJECT_LABORATORY_ADAPTER:
            return
        components = self._color_components(observation)
        objects = [row for row in components if row["width"] <= 10 and row["height"] <= 10]
        outlines = [row for row in components if row["width"] > 10 or row["height"] > 10]
        if len(objects) != 4 or not outlines:
            return
        candidates = []
        for outline in outlines:
            nearest = min(
                range(len(objects)),
                key=lambda index: abs(objects[index]["x"] - outline["x"])
                + abs(objects[index]["y"] - outline["y"]),
            )
            distance = abs(objects[nearest]["x"] - outline["x"]) + abs(
                objects[nearest]["y"] - outline["y"]
            )
            if distance <= 4.0:
                candidates.append((outline["width"] * outline["height"], nearest))
        if not candidates:
            return
        _, target_index = max(candidates)
        self._tracker_active = True
        self._track_colors = [row["color"] for row in objects]
        self._track_x0 = np.asarray([row["x"] for row in objects], dtype=np.float64)
        self._track_target_index = target_index

    @staticmethod
    def _role_glyph(observation: np.ndarray, *, effect: bool = False) -> int | None:
        if effect:
            box = observation[4:18, 25:39]
            vertical = observation[6:16, 30:33]
            horizontal = observation[9:12, 27:37]
        else:
            box = observation[3:15, 3:15]
            vertical = observation[5:13, 7:10]
            horizontal = observation[7:10, 5:13]
        if float(np.mean(np.max(box, axis=2) > 120)) < 0.45:
            return None
        vertical_dark = int(np.sum(np.max(vertical, axis=2) < 70))
        horizontal_dark = int(np.sum(np.max(horizontal, axis=2) < 70))
        if vertical_dark == horizontal_dark:
            return None
        return 0 if vertical_dark > horizontal_dark else 1

    @classmethod
    def _function_objects(cls, observation: np.ndarray) -> list[dict[str, Any]]:
        return sorted(
            [
                row for row in cls._color_components(observation)
                if row["y"] > 20 and row["width"] <= 10 and row["height"] <= 10
            ],
            key=lambda row: row["x"],
        )

    def _initialize_function_memory(self, observation: np.ndarray) -> None:
        self._function_active = False
        self._function_target_role = self._role_glyph(observation)
        self._function_roles = {}
        self._function_colors = []
        self._function_x0 = np.zeros(0, dtype=np.float64)
        self._function_identity_slots = None
        self._function_final_signatures = ()
        self._function_shift_started = False
        self._function_ready = False
        objects = self._function_objects(observation)
        if self._function_target_role is None or len(objects) != 4:
            return
        self._function_active = True
        self._function_colors = [row["color"] for row in objects]
        self._function_x0 = np.asarray([row["x"] for row in objects], dtype=np.float64)

    @staticmethod
    def _fit_identity_slots(
        initial_x: np.ndarray,
        visible_indices: list[int],
        observed_x: np.ndarray,
    ) -> tuple[int, ...] | None:
        slots = np.sort(initial_x)
        initial = initial_x[visible_indices]
        if float(np.max(np.abs(observed_x - initial))) < 0.5:
            return tuple(
                int(np.argmin(np.abs(slots - value))) for value in initial_x
            )
        best: tuple[float, float, tuple[float, ...]] | None = None
        for assignment in itertools.permutations(float(value) for value in slots):
            destinations = np.asarray(assignment, dtype=np.float64)[visible_indices]
            displacement = destinations - initial
            denominator = float(displacement @ displacement)
            if denominator == 0.0:
                continue
            progress = float(displacement @ (observed_x - initial) / denominator)
            residual = float(np.sum((initial + progress * displacement - observed_x) ** 2))
            penalty = 0.0 if 0.03 <= progress <= 0.5 else 1000.0
            candidate = (residual + penalty, residual, assignment)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            return None
        return tuple(int(np.argmin(np.abs(slots - destination))) for destination in best[2])

    def _update_function_memory(
        self,
        observation: np.ndarray,
        *,
        action: int,
        event: str | None,
    ) -> None:
        if not self._function_active:
            return
        if action >= 2 and event in {"effect-observed", "no-change"}:
            role = self._role_glyph(observation, effect=True)
            if role is not None:
                self._function_roles[action - 2] = role
        if event == "appearance-shifted":
            self._function_shift_started = True
        objects = self._function_objects(observation)
        if len(objects) != 4:
            return
        if self._function_shift_started:
            current = {row["color"]: row for row in objects}
            visible = [
                index for index, color in enumerate(self._function_colors)
                if color in current
            ]
            if len(visible) >= 3:
                observed = np.asarray(
                    [current[self._function_colors[index]]["x"] for index in visible],
                    dtype=np.float64,
                )
                self._function_identity_slots = self._fit_identity_slots(
                    self._function_x0, visible, observed
                )
        median_y = float(np.median([row["y"] for row in objects]))
        if self._function_shift_started and median_y >= 46.0:
            self._function_ready = True
            self._function_final_signatures = tuple(
                self._color_signature(row["color"]) for row in objects
            )

    def _update_tracker(self, observation: np.ndarray) -> None:
        if not self._tracker_active or self._predicted_slot is not None or self.episode_step < 1:
            return
        current = {row["color"]: row for row in self._color_components(observation)}
        visible = [
            index for index, color in enumerate(self._track_colors)
            if color in current
        ]
        if len(visible) < 3 or self._track_target_index is None:
            return
        initial = self._track_x0[visible]
        observed = np.asarray(
            [current[self._track_colors[index]]["x"] for index in visible],
            dtype=np.float64,
        )
        slots = np.sort(self._track_x0)
        if float(np.max(np.abs(observed - initial))) < 0.5:
            target_x = self._track_x0[self._track_target_index]
            self._predicted_slot = int(np.argmin(np.abs(slots - target_x)))
            return
        best: tuple[float, float, tuple[float, ...]] | None = None
        for assignment in itertools.permutations(float(value) for value in slots):
            destinations = np.asarray(assignment, dtype=np.float64)[visible]
            displacement = destinations - initial
            denominator = float(displacement @ displacement)
            if denominator == 0.0:
                continue
            progress = float(displacement @ (observed - initial) / denominator)
            residual = float(np.sum((initial + progress * displacement - observed) ** 2))
            penalty = 0.0 if 0.03 <= progress <= 0.5 else 1000.0
            candidate = (residual + penalty, residual, assignment)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            return
        destination = best[2][self._track_target_index]
        self._predicted_slot = int(np.argmin(np.abs(slots - destination)))

    @staticmethod
    def _causal_meta_key() -> str:
        return "causal-meta:probe-then-repeat-progress"

    def _causal_strategy_learned(self) -> bool:
        if self.spec is None:
            return False
        states = self.swarm_values.get(self.spec.adapter, {})
        visits_by_state = self.swarm_visits.get(self.spec.adapter, {})
        values = states.get(self._causal_meta_key())
        visits = visits_by_state.get(self._causal_meta_key())
        return bool(
            values is not None
            and visits is not None
            and int(visits[:, 0].sum()) > 0
            and float(values[:, 0].mean()) > 0.05
        )

    def _causal_action(self) -> int:
        if self.spec is None:
            raise SchoolLearnerError("causal action requested before begin")
        if self._causal_progress_action is not None:
            action = self._causal_progress_action
        else:
            untried = [
                action for action in range(self.spec.action_count)
                if action not in self._causal_tried
            ]
            if not untried:
                self._causal_tried.clear()
                untried = list(range(self.spec.action_count))
            offset = (self.episodes + self.active_replica) % len(untried)
            action = untried[offset]
        confidence = np.zeros(self.spec.action_count, dtype=np.float64)
        confidence[action] = 1.0
        self.last_q = confidence
        return int(action)

    def _parameters(self, spec: PublicWorldSpec) -> np.ndarray:
        if spec.adapter not in FOUNDATIONAL_ADAPTERS:
            raise SchoolLearnerError(f"unadmitted school adapter {spec.adapter!r}")
        existing = self.action_counts.get(spec.adapter)
        if existing is not None and existing != spec.action_count:
            raise SchoolLearnerError("adapter action count changed across worlds")
        self.action_counts[spec.adapter] = int(spec.action_count)
        if spec.adapter not in self.weights:
            self.weights[spec.adapter] = np.zeros(
                (spec.action_count, self.feature_count), dtype=np.float64
            )
        return self.weights[spec.adapter]

    def _state_key(self) -> str | None:
        if self._function_active:
            role = "unknown" if self._function_target_role is None else self._function_target_role
            mask = sum(1 << identity for identity in self._function_roles)
            if not self._function_shift_started:
                return f"function-memory:probe:target={role}:mask={mask}"
            if not self._function_ready:
                return f"function-memory:transition:target={role}:mask={mask}"
            candidates = self._function_candidate_slots()
            encoded = "unknown" if not candidates else ",".join(str(slot) for slot in candidates)
            return f"function-memory:ready:candidates={encoded}"
        if not self._tracker_active:
            return None
        step = min(self.episode_step, 6)
        slot = "unknown" if self._predicted_slot is None else str(self._predicted_slot)
        return f"object-memory:step={step}:slot={slot}"

    def _resolved_action_key(self) -> str | None:
        if not self._tracker_active or self._predicted_slot is None:
            return None
        return f"object-action-map:slot={self._predicted_slot}"

    def _function_candidate_slots(self) -> tuple[int, ...]:
        if self._function_target_role is None:
            return ()
        identity_slots = self._function_identity_slots
        if identity_slots is None:
            identity_slots = self._appearance_identity_slots()
        if identity_slots is None:
            return ()
        return tuple(sorted(
            identity_slots[identity]
            for identity, role in self._function_roles.items()
            if role == self._function_target_role
        ))

    @staticmethod
    def _function_action_key(slot: int) -> str:
        return f"function-action-map:slot={slot}"

    @staticmethod
    def _color_signature(color: tuple[int, int, int]) -> str:
        order = tuple(int(index) for index in np.argsort(np.asarray(color)))
        return "".join(str(index) for index in order)

    @staticmethod
    def _appearance_key(signature: str) -> str:
        return f"function-appearance:channel-order={signature}"

    def _appearance_identity_slots(self) -> tuple[int, ...] | None:
        if len(self._function_final_signatures) != 4 or self.spec is None:
            return None
        rows = []
        for signature in self._function_final_signatures:
            values, _ = self._swarm_parameters(
                self.spec.adapter, self._appearance_key(signature)
            )
            rows.append(values.mean(axis=0)[:4])
        scores = np.asarray(rows, dtype=np.float64)
        best: tuple[float, tuple[int, ...]] | None = None
        for assignment in itertools.permutations(range(4)):
            score = float(sum(scores[slot, identity] for slot, identity in enumerate(assignment)))
            candidate = (score, tuple(int(identity) for identity in assignment))
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            return None
        slots_by_identity = [0, 0, 0, 0]
        for slot, identity in enumerate(best[1]):
            slots_by_identity[identity] = slot
        return tuple(slots_by_identity)

    def _swarm_parameters(self, adapter: str, state_key: str) -> tuple[np.ndarray, np.ndarray]:
        action_count = self.action_counts[adapter]
        values_by_state = self.swarm_values.setdefault(adapter, {})
        visits_by_state = self.swarm_visits.setdefault(adapter, {})
        if state_key not in values_by_state:
            values_by_state[state_key] = np.zeros(
                (self.replica_count, action_count), dtype=np.float64
            )
            visits_by_state[state_key] = np.zeros(
                (self.replica_count, action_count), dtype=np.int64
            )
        return values_by_state[state_key], visits_by_state[state_key]

    def _allowed_actions(self) -> np.ndarray:
        if self.spec is None:
            return np.zeros(0, dtype=np.int64)
        if self._function_active and self._function_ready:
            # Observation actions have already become public no-ops in this
            # state. Keep the learned choice among the four anonymous output
            # actions instead of allowing a deterministic no-op loop.
            return np.arange(2, self.spec.action_count, dtype=np.int64)
        if not self._function_active or self._function_shift_started:
            return np.arange(self.spec.action_count if self.spec is not None else 0)
        unprobed = [
            2 + identity for identity in range(4)
            if identity not in self._function_roles
        ]
        match_known = (
            self._function_target_role is not None
            and self._function_target_role in self._function_roles.values()
        )
        allowed = ([0, 1] if match_known or not unprobed else []) + unprobed
        return np.asarray(allowed, dtype=np.int64)

    @staticmethod
    def _q_confidence(values: np.ndarray) -> float:
        if not len(values):
            return 0.0
        if len(values) == 1:
            return 1.0
        ordered = np.sort(values)
        return float(1.0 - np.exp(-abs(float(ordered[-1] - ordered[-2]))))

    def _spawn_replica(self, reason: str) -> None:
        if self.replica_count >= self.max_replicas:
            return
        for adapter, states in self.swarm_values.items():
            for state_key, values in states.items():
                shared = values.mean(axis=0, keepdims=True)
                states[state_key] = np.concatenate((values, shared), axis=0)
                visits = self.swarm_visits[adapter][state_key]
                self.swarm_visits[adapter][state_key] = np.concatenate(
                    (visits, np.zeros((1, visits.shape[1]), dtype=np.int64)), axis=0
                )
        self.replica_count += 1
        self.replica_epsilons.append(max(0.45, float(np.mean(self.replica_epsilons))))
        self.replica_episodes.append(0)
        self.spawn_events.append({
            "episode": self.episodes,
            "training_step": self.training_steps,
            "reason": reason,
            "replica_count": self.replica_count,
        })

    def _communicate(self) -> None:
        if self.replica_count < 2:
            return
        for adapter, states in self.swarm_values.items():
            for state_key, values in states.items():
                visits = self.swarm_visits[adapter][state_key]
                for action in range(values.shape[1]):
                    informed = visits[:, action] > 0
                    if not np.any(informed):
                        continue
                    shared = float(np.mean(values[informed, action]))
                    values[:, action] = 0.70 * values[:, action] + 0.30 * shared
        self.communication_rounds += 1

    def begin(self, spec: PublicWorldSpec, observation: Any, *, training: bool) -> None:
        if spec.observation_kind != "rgb" or spec.action_kind != "discrete-anonymous":
            raise SchoolLearnerError("unsupported school perception or action contract")
        array = self._validate_observation(spec, observation)
        self._parameters(spec)
        self.spec = spec
        self.features = self.visual_features(array)
        self.last_q = None
        self.last_state_key = None
        self.active_replica = self.episodes % self.replica_count if training else 0
        self.uncertainty_streak = 0
        self.episode_step = 0
        self._initialize_function_memory(array)
        self._causal_active = spec.adapter == CAUSAL_WORKSHOP_ADAPTER
        self._causal_tried = set()
        self._causal_progress_action = None
        if self._function_active:
            self._tracker_active = False
            self._predicted_slot = None
        else:
            self._initialize_tracker(array)
        self.seen_worlds.add(spec.world_id)

    def act(self, observation: Any, *, training: bool) -> int:
        if self.spec is None:
            raise SchoolLearnerError("begin must be called before act")
        array = self._validate_observation(self.spec, observation)
        self.features = self.visual_features(array)
        self._update_tracker(array)
        if self._causal_active and (training or self._causal_strategy_learned()):
            mask = sum(1 << action for action in self._causal_tried)
            progress = (
                "unknown" if self._causal_progress_action is None
                else str(self._causal_progress_action)
            )
            self.last_state_key = f"causal-memory:tried={mask}:progress={progress}"
            return self._causal_action()
        state_key = self._state_key()
        resolved_key = self._resolved_action_key()
        if not training and resolved_key is not None:
            # The terminal reward teaches a time-independent mapping from the
            # predicted destination to an anonymous action. Evaluation uses
            # that mapping so speed and occlusion-duration shifts do not create
            # a new, untrained animation-step state.
            resolved_values, _ = self._swarm_parameters(self.spec.adapter, resolved_key)
            if self._q_confidence(resolved_values.mean(axis=0)) >= 0.05:
                state_key = resolved_key
        self.last_state_key = state_key
        if state_key is not None:
            values, _ = self._swarm_parameters(self.spec.adapter, state_key)
            if not training and self._function_ready:
                candidates = self._function_candidate_slots()
                if candidates:
                    transfer_rows = [
                        self._swarm_parameters(
                            self.spec.adapter, self._function_action_key(slot)
                        )[0]
                        for slot in candidates
                    ]
                    transferred = np.maximum.reduce(transfer_rows)
                    if self._q_confidence(transferred.mean(axis=0)) >= 0.05:
                        values = transferred
            aggregate = values.mean(axis=0)
            self.last_q = aggregate
            allowed = self._allowed_actions()
            if not len(allowed):
                allowed = np.arange(self.spec.action_count)
            self.last_q = aggregate[allowed]
            if training:
                confidence = self._q_confidence(aggregate)
                self.uncertainty_streak = (
                    self.uncertainty_streak + 1
                    if confidence < self.spawn_uncertainty_threshold else 0
                )
                if self.uncertainty_streak >= self.spawn_patience:
                    self._spawn_replica("sustained-low-action-confidence")
                    self.uncertainty_streak = 0
                    values, _ = self._swarm_parameters(self.spec.adapter, state_key)
                    aggregate = values.mean(axis=0)
                    self.last_q = aggregate
                epsilon = self.replica_epsilons[self.active_replica]
                if self.rng.random() < epsilon:
                    return int(allowed[int(self.rng.integers(len(allowed)))])
                row = values[self.active_replica]
                finalists = allowed[row[allowed] == np.max(row[allowed])]
                return int(finalists[0])
            votes = np.asarray(
                [int(allowed[np.argmax(row[allowed])]) for row in values], dtype=np.int64
            )
            counts = np.bincount(votes, minlength=self.spec.action_count)
            finalists = allowed[counts[allowed] == np.max(counts[allowed])]
            return int(finalists[np.argmax(aggregate[finalists])])

        parameters = self._parameters(self.spec)
        self.last_q = parameters @ self.features
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.spec.action_count))
        return int(np.flatnonzero(self.last_q == np.max(self.last_q))[0])

    def observe(self, action: Any, transition: Transition, *, training: bool) -> None:
        if self.spec is None or self.features is None:
            raise SchoolLearnerError("begin and act must precede observe")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise SchoolLearnerError("action must be an integer slot")
        action = int(action)
        if not 0 <= action < self.spec.action_count:
            raise SchoolLearnerError("action is outside the public contract")
        unknown = sorted(set(transition.public_info) - self.allowed_public_info)
        if unknown:
            self.boundary_violations.append(f"unexpected public-info fields: {unknown}")
        if not isinstance(transition.reward, (int, float)):
            self.boundary_violations.append("non-numeric reward")
            raise SchoolLearnerError("reward must be numeric")
        if not self.spec.reward_range[0] <= float(transition.reward) <= self.spec.reward_range[1]:
            self.boundary_violations.append("reward outside public contract")
            raise SchoolLearnerError("reward is outside the public contract")
        next_array = self._validate_observation(self.spec, transition.observation)
        next_features = self.visual_features(next_array)
        terminal = bool(transition.terminated or transition.truncated)
        current_state = self.last_state_key
        self.episode_step += 1
        event = transition.public_info.get("event")
        if self._causal_active:
            self._causal_tried.add(action)
            if event == "progress" or float(transition.reward) > 0.1:
                self._causal_progress_action = action
        self._update_function_memory(next_array, action=action, event=event)
        if not self._function_active:
            self._update_tracker(next_array)
        next_state = self._state_key()
        if training and current_state is not None:
            values, visits = self._swarm_parameters(self.spec.adapter, current_state)
            current = float(values[self.active_replica, action])
            target = float(transition.reward)
            if not terminal and next_state is not None:
                next_values, _ = self._swarm_parameters(self.spec.adapter, next_state)
                target += self.gamma * float(np.max(next_values[self.active_replica]))
            values[self.active_replica, action] += self.swarm_alpha * (target - current)
            visits[self.active_replica, action] += 1
            resolved_key = self._resolved_action_key()
            if terminal and resolved_key is not None:
                resolved_values, resolved_visits = self._swarm_parameters(
                    self.spec.adapter, resolved_key
                )
                resolved_current = float(resolved_values[self.active_replica, action])
                resolved_values[self.active_replica, action] += self.swarm_alpha * (
                    float(transition.reward) - resolved_current
                )
                resolved_visits[self.active_replica, action] += 1
            function_candidates = self._function_candidate_slots()
            if terminal and len(function_candidates) == 1:
                function_values, function_visits = self._swarm_parameters(
                    self.spec.adapter,
                    self._function_action_key(function_candidates[0]),
                )
                function_current = float(function_values[self.active_replica, action])
                function_values[self.active_replica, action] += self.swarm_alpha * (
                    float(transition.reward) - function_current
                )
                function_visits[self.active_replica, action] += 1
            if (
                terminal
                and action >= 2
                and len(self._function_final_signatures) == 4
                and self._function_target_role is not None
            ):
                selected_slot = action - 2
                signature = self._function_final_signatures[selected_slot]
                appearance_values, appearance_visits = self._swarm_parameters(
                    self.spec.adapter, self._appearance_key(signature)
                )
                observed_reward = float(transition.reward)
                for identity, role in self._function_roles.items():
                    compatible = role == self._function_target_role
                    target_value = observed_reward if compatible else -observed_reward
                    current_value = float(
                        appearance_values[self.active_replica, identity]
                    )
                    appearance_values[self.active_replica, identity] += self.swarm_alpha * (
                        target_value - current_value
                    )
                    appearance_visits[self.active_replica, identity] += 1
            if self._causal_active and terminal and float(transition.reward) > 0:
                strategy_values, strategy_visits = self._swarm_parameters(
                    self.spec.adapter, self._causal_meta_key()
                )
                current_value = float(strategy_values[self.active_replica, 0])
                strategy_values[self.active_replica, 0] += self.swarm_alpha * (
                    1.0 - current_value
                )
                strategy_visits[self.active_replica, 0] += 1
        elif training:
            parameters = self._parameters(self.spec)
            current = float(parameters[action] @ self.features)
            target = float(transition.reward)
            if not terminal:
                target += self.gamma * float(np.max(parameters @ next_features))
            parameters[action] += self.alpha * (target - current) * self.features
            np.clip(parameters[action], -10.0, 10.0, out=parameters[action])
        if training:
            self.training_steps += 1
            self.adapter_steps[self.spec.adapter] = self.adapter_steps.get(self.spec.adapter, 0) + 1
        else:
            self.evaluation_steps += 1
        self.features = next_features

    def finish_episode(self, *, training: bool) -> None:
        self.episodes += 1
        if training:
            self.replica_episodes[self.active_replica] += 1
            self.epsilon = max(0.05, self.epsilon * 0.985)
            self.replica_epsilons[self.active_replica] = max(
                0.08, self.replica_epsilons[self.active_replica] * 0.992
            )
            if self.episodes % self.replica_count == 0:
                self._communicate()

    def confidence(self) -> float:
        return 0.0 if self.last_q is None else self._q_confidence(self.last_q)

    def status(self) -> dict[str, Any]:
        swarm_parameters = sum(
            values.size
            for states in self.swarm_values.values()
            for values in states.values()
        )
        return {
            "format": self.format,
            "episodes": self.episodes,
            "training_steps": self.training_steps,
            "evaluation_steps": self.evaluation_steps,
            "shared_adapters": sorted(self.weights),
            "parameter_count": int(
                sum(value.size for value in self.weights.values()) + swarm_parameters
            ),
            "seen_world_count": len(self.seen_worlds),
            "epsilon": self.epsilon,
            "replica_count": self.replica_count,
            "max_replicas": self.max_replicas,
            "replica_episodes": list(self.replica_episodes),
            "communication_rounds": self.communication_rounds,
            "spawn_events": list(self.spawn_events),
            "memory_states": int(sum(len(states) for states in self.swarm_values.values())),
            "boundary_violations": list(self.boundary_violations),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "seed": self.seed,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "rng_state": self.rng.bit_generator.state,
            "max_replicas": self.max_replicas,
            "replica_count": self.replica_count,
            "replica_epsilons": self.replica_epsilons,
            "replica_episodes": self.replica_episodes,
            "weights": {adapter: value.tolist() for adapter, value in sorted(self.weights.items())},
            "swarm_values": {
                adapter: {state: values.tolist() for state, values in sorted(states.items())}
                for adapter, states in sorted(self.swarm_values.items())
            },
            "swarm_visits": {
                adapter: {state: visits.tolist() for state, visits in sorted(states.items())}
                for adapter, states in sorted(self.swarm_visits.items())
            },
            "action_counts": dict(sorted(self.action_counts.items())),
            "adapter_steps": dict(sorted(self.adapter_steps.items())),
            "seen_worlds": sorted(self.seen_worlds),
            "episodes": self.episodes,
            "training_steps": self.training_steps,
            "evaluation_steps": self.evaluation_steps,
            "communication_rounds": self.communication_rounds,
            "spawn_events": list(self.spawn_events),
            "boundary_violations": list(self.boundary_violations),
        }

    def save(self, path: Path) -> None:
        atomic_write_json(Path(path), self.to_json(), indent=None, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> "CrossSeedSchoolLearner":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SchoolLearnerError(f"learner state is unreadable: {error}") from error
        required = {
            "format", "seed", "alpha", "gamma", "epsilon", "rng_state", "max_replicas",
            "replica_count", "replica_epsilons", "replica_episodes", "weights",
            "swarm_values", "swarm_visits",
            "action_counts", "adapter_steps", "seen_worlds", "episodes", "training_steps",
            "evaluation_steps", "communication_rounds", "spawn_events", "boundary_violations",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise SchoolLearnerError("learner state fields differ from the reviewed format")
        if value["format"] != cls.format:
            raise SchoolLearnerError("unsupported school learner format")
        seed = value["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SchoolLearnerError("learner seed must be an integer")
        for field in ("alpha", "gamma", "epsilon"):
            number = value[field]
            if (isinstance(number, bool) or not isinstance(number, (int, float))
                    or not math.isfinite(float(number))):
                raise SchoolLearnerError(f"learner {field} must be finite and numeric")
        if float(value["alpha"]) <= 0.0:
            raise SchoolLearnerError("learner alpha must be positive")
        if not 0.0 <= float(value["gamma"]) <= 1.0:
            raise SchoolLearnerError("learner gamma is outside [0, 1]")
        result = cls(
            seed,
            alpha=value["alpha"],
            gamma=value["gamma"],
            max_replicas=value["max_replicas"],
            initial_replicas=value["replica_count"],
        )
        result.epsilon = float(value["epsilon"])
        if not 0.0 <= result.epsilon <= 1.0:
            raise SchoolLearnerError("learner epsilon is outside [0, 1]")
        epsilons = value["replica_epsilons"]
        if (not isinstance(epsilons, list) or len(epsilons) != result.replica_count
                or any(isinstance(item, bool) or not isinstance(item, (int, float))
                       or not math.isfinite(float(item)) or not 0.0 <= float(item) <= 1.0
                       for item in epsilons)):
            raise SchoolLearnerError("invalid replica epsilon table")
        result.replica_epsilons = [float(item) for item in epsilons]
        replica_episodes = value["replica_episodes"]
        if (not isinstance(replica_episodes, list)
                or len(replica_episodes) != result.replica_count
                or any(isinstance(item, bool) or not isinstance(item, int) or item < 0
                       for item in replica_episodes)):
            raise SchoolLearnerError("invalid replica episode table")
        result.replica_episodes = list(replica_episodes)
        if not isinstance(value["weights"], dict) or not isinstance(value["action_counts"], dict):
            raise SchoolLearnerError("learner parameter tables must be objects")
        for adapter, rows in value["weights"].items():
            if adapter not in FOUNDATIONAL_ADAPTERS:
                raise SchoolLearnerError(f"learner state names unsupported adapter {adapter!r}")
            try:
                array = np.asarray(rows, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise SchoolLearnerError(f"invalid parameter shape for {adapter}") from error
            count = value["action_counts"].get(adapter)
            if (isinstance(count, bool) or not isinstance(count, int)
                    or array.shape != (count, cls.feature_count)
                    or not np.isfinite(array).all()):
                raise SchoolLearnerError(f"invalid parameter shape for {adapter}")
            result.weights[adapter] = array
            result.action_counts[adapter] = count
        if set(value["action_counts"]) != set(result.weights):
            raise SchoolLearnerError("action counts and parameter tables differ")
        if not isinstance(value["swarm_values"], dict) or not isinstance(value["swarm_visits"], dict):
            raise SchoolLearnerError("swarm parameter tables must be objects")
        if set(value["swarm_values"]) != set(value["swarm_visits"]):
            raise SchoolLearnerError("swarm value and visit adapters differ")
        for adapter, states in value["swarm_values"].items():
            if adapter not in result.action_counts or not isinstance(states, dict):
                raise SchoolLearnerError("invalid swarm adapter table")
            visit_states = value["swarm_visits"][adapter]
            if not isinstance(visit_states, dict) or set(states) != set(visit_states):
                raise SchoolLearnerError("swarm value and visit states differ")
            result.swarm_values[adapter] = {}
            result.swarm_visits[adapter] = {}
            shape = (result.replica_count, result.action_counts[adapter])
            for state_key, rows in states.items():
                if not isinstance(state_key, str) or not state_key:
                    raise SchoolLearnerError("swarm state keys must be non-empty strings")
                try:
                    values = np.asarray(rows, dtype=np.float64)
                    visits = np.asarray(visit_states[state_key], dtype=np.int64)
                except (TypeError, ValueError) as error:
                    raise SchoolLearnerError("invalid swarm parameter shape") from error
                if (values.shape != shape or visits.shape != shape
                        or not np.isfinite(values).all() or np.any(visits < 0)):
                    raise SchoolLearnerError("invalid swarm parameter shape")
                result.swarm_values[adapter][state_key] = values
                result.swarm_visits[adapter][state_key] = visits
        if not isinstance(value["adapter_steps"], dict):
            raise SchoolLearnerError("adapter_steps must be an object")
        result.adapter_steps = {}
        for key, number in value["adapter_steps"].items():
            if (isinstance(number, bool) or not isinstance(number, int)
                    or key not in FOUNDATIONAL_ADAPTERS or number < 0):
                raise SchoolLearnerError("invalid adapter step record")
            result.adapter_steps[str(key)] = number
        if (not isinstance(value["seen_worlds"], list)
                or not all(isinstance(item, str) and item for item in value["seen_worlds"])):
            raise SchoolLearnerError("seen_worlds must contain identifiers")
        result.seen_worlds = set(value["seen_worlds"])
        for field in ("episodes", "training_steps", "evaluation_steps", "communication_rounds"):
            number = value[field]
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise SchoolLearnerError(f"{field} must be a non-negative integer")
            setattr(result, field, number)
        if not isinstance(value["spawn_events"], list):
            raise SchoolLearnerError("spawn_events must be an array of objects")
        for item in value["spawn_events"]:
            if (not isinstance(item, dict)
                    or set(item) != {"episode", "training_step", "reason", "replica_count"}
                    or isinstance(item["episode"], bool) or not isinstance(item["episode"], int)
                    or item["episode"] < 0
                    or isinstance(item["training_step"], bool)
                    or not isinstance(item["training_step"], int)
                    or item["training_step"] < 0
                    or not isinstance(item["reason"], str) or not item["reason"]
                    or isinstance(item["replica_count"], bool)
                    or not isinstance(item["replica_count"], int)
                    or not 2 <= item["replica_count"] <= result.max_replicas):
                raise SchoolLearnerError("invalid spawn event")
        result.spawn_events = list(value["spawn_events"])
        if (not isinstance(value["boundary_violations"], list)
                or not all(isinstance(item, str) for item in value["boundary_violations"])):
            raise SchoolLearnerError("boundary_violations must be an array of strings")
        result.boundary_violations = list(value["boundary_violations"])
        try:
            result.rng.bit_generator.state = value["rng_state"]
        except (TypeError, ValueError) as error:
            raise SchoolLearnerError("invalid random-generator state") from error
        return result
