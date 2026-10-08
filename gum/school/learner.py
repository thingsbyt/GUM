"""Persistent cross-seed learner state for the bounded GUM School lane.

This is deliberately modest: a shared linear value function over coarse visual
features.  Parameters are keyed by reviewed adapter, never by world identity or
seed, so experience can affect later seeds without storing task solutions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gum.protocol import PublicWorldSpec, Transition
from gum.storage import atomic_write_json

from .worlds import FOUNDATIONAL_ADAPTERS


class SchoolLearnerError(ValueError):
    pass


class CrossSeedSchoolLearner:
    format = "gum-school-cross-seed-learner-v1"
    feature_grid = 4
    feature_count = feature_grid * feature_grid * 3 + 7
    allowed_public_info = frozenset({"event", "success"})

    def __init__(self, seed: int = 8_620_001, *, alpha: float = 0.04, gamma: float = 0.96):
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = 0.35
        self.rng = np.random.default_rng(self.seed)
        self.weights: dict[str, np.ndarray] = {}
        self.action_counts: dict[str, int] = {}
        self.adapter_steps: dict[str, int] = {}
        self.seen_worlds: set[str] = set()
        self.episodes = 0
        self.training_steps = 0
        self.evaluation_steps = 0
        self.boundary_violations: list[str] = []
        self.spec: PublicWorldSpec | None = None
        self.features: np.ndarray | None = None
        self.last_q: np.ndarray | None = None

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

    def begin(self, spec: PublicWorldSpec, observation: Any, *, training: bool) -> None:
        if spec.observation_kind != "rgb" or spec.action_kind != "discrete-anonymous":
            raise SchoolLearnerError("unsupported school perception or action contract")
        array = self._validate_observation(spec, observation)
        self._parameters(spec)
        self.spec = spec
        self.features = self.visual_features(array)
        self.last_q = None
        self.seen_worlds.add(spec.world_id)

    def act(self, observation: Any, *, training: bool) -> int:
        if self.spec is None:
            raise SchoolLearnerError("begin must be called before act")
        array = self._validate_observation(self.spec, observation)
        self.features = self.visual_features(array)
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
        if training:
            parameters = self._parameters(self.spec)
            current = float(parameters[action] @ self.features)
            terminal = bool(transition.terminated or transition.truncated)
            target = float(transition.reward)
            if not terminal:
                target += self.gamma * float(np.max(parameters @ next_features))
            parameters[action] += self.alpha * (target - current) * self.features
            np.clip(parameters[action], -10.0, 10.0, out=parameters[action])
            self.training_steps += 1
            self.adapter_steps[self.spec.adapter] = self.adapter_steps.get(self.spec.adapter, 0) + 1
        else:
            self.evaluation_steps += 1
        self.features = next_features

    def finish_episode(self, *, training: bool) -> None:
        self.episodes += 1
        if training:
            self.epsilon = max(0.05, self.epsilon * 0.985)

    def confidence(self) -> float:
        if self.last_q is None or not len(self.last_q):
            return 0.0
        if len(self.last_q) == 1:
            return 1.0
        ordered = np.sort(self.last_q)
        return float(1.0 - np.exp(-abs(float(ordered[-1] - ordered[-2]))))

    def status(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "episodes": self.episodes,
            "training_steps": self.training_steps,
            "evaluation_steps": self.evaluation_steps,
            "shared_adapters": sorted(self.weights),
            "parameter_count": int(sum(value.size for value in self.weights.values())),
            "seen_world_count": len(self.seen_worlds),
            "epsilon": self.epsilon,
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
            "weights": {adapter: value.tolist() for adapter, value in sorted(self.weights.items())},
            "action_counts": dict(sorted(self.action_counts.items())),
            "adapter_steps": dict(sorted(self.adapter_steps.items())),
            "seen_worlds": sorted(self.seen_worlds),
            "episodes": self.episodes,
            "training_steps": self.training_steps,
            "evaluation_steps": self.evaluation_steps,
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
            "format", "seed", "alpha", "gamma", "epsilon", "rng_state", "weights",
            "action_counts", "adapter_steps", "seen_worlds", "episodes", "training_steps",
            "evaluation_steps", "boundary_violations",
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
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise SchoolLearnerError(f"learner {field} must be numeric")
            if not math.isfinite(float(number)):
                raise SchoolLearnerError(f"learner {field} must be finite")
        if float(value["alpha"]) <= 0.0:
            raise SchoolLearnerError("learner alpha must be positive")
        if not 0.0 <= float(value["gamma"]) <= 1.0:
            raise SchoolLearnerError("learner gamma is outside [0, 1]")
        result = cls(seed, alpha=value["alpha"], gamma=value["gamma"])
        result.epsilon = float(value["epsilon"])
        if not 0.0 <= result.epsilon <= 1.0:
            raise SchoolLearnerError("learner epsilon is outside [0, 1]")
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
        if not isinstance(value["adapter_steps"], dict):
            raise SchoolLearnerError("adapter_steps must be an object")
        result.adapter_steps = {}
        for key, number in value["adapter_steps"].items():
            if isinstance(number, bool) or not isinstance(number, int):
                raise SchoolLearnerError("invalid adapter step record")
            result.adapter_steps[str(key)] = number
        if any(key not in FOUNDATIONAL_ADAPTERS or number < 0
               for key, number in result.adapter_steps.items()):
            raise SchoolLearnerError("invalid adapter step record")
        if (not isinstance(value["seen_worlds"], list)
                or not all(isinstance(item, str) and item for item in value["seen_worlds"])):
            raise SchoolLearnerError("seen_worlds must contain identifiers")
        result.seen_worlds = set(value["seen_worlds"])
        for field in ("episodes", "training_steps", "evaluation_steps"):
            number = value[field]
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise SchoolLearnerError(f"{field} must be a non-negative integer")
            setattr(result, field, number)
        if (not isinstance(value["boundary_violations"], list)
                or not all(isinstance(item, str) for item in value["boundary_violations"])):
            raise SchoolLearnerError("boundary_violations must be an array of strings")
        result.boundary_violations = list(value["boundary_violations"])
        try:
            result.rng.bit_generator.state = value["rng_state"]
        except (TypeError, ValueError) as error:
            raise SchoolLearnerError("invalid random-generator state") from error
        return result
