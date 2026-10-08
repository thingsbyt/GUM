"""A persistent pixel-only practice mind for validating the GUM protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .protocol import PublicWorldSpec, Transition


def observation_id(observation) -> str:
    """Stable fingerprint for one image or a nested multi-agent observation."""
    digest = hashlib.sha256()

    def add(value):
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(b"array\0"); digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode()); digest.update(array.tobytes())
        elif isinstance(value, (list, tuple)):
            digest.update(b"sequence\0"); digest.update(str(len(value)).encode())
            for item in value: add(item)
        elif isinstance(value, dict):
            digest.update(b"mapping\0")
            for key in sorted(value): digest.update(str(key).encode()); add(value[key])
        else:
            digest.update(repr(value).encode("utf-8"))

    add(observation)
    return digest.hexdigest()[:24]


class PixelQLearner:
    """Tabular control over pixel hashes; old world skills remain intact."""
    format = "gum-pixel-q-mind-v1"

    def __init__(self, seed=8_100_001, alpha=.35, gamma=.96):
        self.seed = int(seed); self.rng = np.random.default_rng(seed)
        self.alpha = float(alpha); self.gamma = float(gamma); self.epsilon = 1.0
        self.q: dict[str, np.ndarray] = {}; self.world_states = {}; self.episodes = 0
        self.training_steps = 0; self.evaluation_steps = 0; self.spec = None; self.state_key = None

    def _key(self, observation): return f"{self.spec.world_id}:{observation_id(observation)}"

    def _values(self, key):
        if key not in self.q: self.q[key] = np.zeros(self.spec.action_count, dtype=np.float32)
        return self.q[key]

    def begin(self, spec: PublicWorldSpec, observation: np.ndarray, *, training: bool):
        self.spec = spec; self.state_key = self._key(observation)
        if training: self._values(self.state_key)

    def act(self, observation: np.ndarray, *, training: bool):
        self.state_key = self._key(observation); values = self._values(self.state_key) if training else self.q.get(self.state_key)
        if training and self.rng.random() < self.epsilon: return int(self.rng.integers(self.spec.action_count))
        if values is None: return int(self.rng.integers(self.spec.action_count))
        best = np.flatnonzero(values == values.max()); return int(best[int(self.rng.integers(len(best)))])

    def observe(self, action: int, transition: Transition, *, training: bool):
        next_key = self._key(transition.observation)
        if training:
            current = self._values(self.state_key); next_values = self._values(next_key)
            terminal = transition.terminated or transition.truncated
            target = float(transition.reward) + (0.0 if terminal else self.gamma * float(next_values.max()))
            current[int(action)] += self.alpha * (target - float(current[int(action)]))
            self.training_steps += 1
            states = self.world_states.setdefault(self.spec.world_id, set()); states.update((self.state_key, next_key))
        else: self.evaluation_steps += 1
        self.state_key = next_key

    def finish_episode(self): self.episodes += 1

    def status(self):
        return {"format": self.format, "episodes": self.episodes, "training_steps": self.training_steps,
            "evaluation_steps": self.evaluation_steps, "learned_states": len(self.q),
            "worlds_retained": len(self.world_states),
            "states_by_world": {world: len(states) for world, states in self.world_states.items()},
            "epsilon": self.epsilon}

    def save(self, path: Path):
        payload = {"format": self.format, "seed": self.seed, "alpha": self.alpha, "gamma": self.gamma,
            "epsilon": self.epsilon, "episodes": self.episodes, "training_steps": self.training_steps,
            "evaluation_steps": self.evaluation_steps, "q": {key: value.tolist() for key, value in self.q.items()},
            "world_states": {world: sorted(states) for world, states in self.world_states.items()}}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format") != cls.format: raise ValueError("unsupported mind format")
        model = cls(value["seed"], value["alpha"], value["gamma"]); model.epsilon = float(value["epsilon"])
        model.episodes = int(value["episodes"]); model.training_steps = int(value["training_steps"])
        model.evaluation_steps = int(value["evaluation_steps"])
        model.q = {key: np.asarray(row, dtype=np.float32) for key, row in value["q"].items()}
        model.world_states = {world: set(rows) for world, rows in value["world_states"].items()}; return model
