"""First-class GUM adapters for the project's established visual worlds."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .protocol import PublicWorldSpec, Transition


ASTEROIDS_ADAPTER = "gum-two-rocket-asteroids-v1"
IMMUNE_ADAPTER = "gum-cooperative-immune-v1"


def _private(folder: Path) -> dict:
    return json.loads((Path(folder) / "genome.private.json").read_text(encoding="utf-8"))


class TwoRocketAsteroidsAdapter:
    def __init__(self, world_id: str, *, seed: int, difficulty=2, max_steps=360):
        from jepa_asteroids.cooperative_asteroids import DuoAsteroids, DuoConfig
        self.world_id = str(world_id); self.seed = int(seed)
        self.config = DuoConfig(difficulty=int(difficulty), max_steps=int(max_steps))
        self.game = DuoAsteroids(self.config, self.seed)

    def public_spec(self):
        return PublicWorldSpec(self.world_id, "two-rocket-asteroids", ASTEROIDS_ADAPTER, 2,
            "two-local-rgb-views", (2, self.config.image_height, self.config.image_width, 3),
            "two-anonymous-discrete-actions", 5, self.config.max_steps, (-40.0, 2.0))

    def reset(self, seed=None): return self.game.reset(self.seed if seed is None else int(seed))

    def step(self, actions):
        frames, reward, terminated, truncated, info = self.game.step(actions)
        public = {key: info[key] for key in
                  ("hits", "hits_by_agent", "friendly_fire", "alive", "deaths", "wave", "lives", "return")}
        public["success"] = bool(truncated and all(info["alive"]) and info["hits"] > 0)
        return Transition(frames, reward, terminated, truncated, public)

    def human_frame(self): return np.concatenate(self.game.observe(), axis=1)
    def audit_state(self): return self.game.audit_state()


class CooperativeImmuneAdapter:
    def __init__(self, world_id: str, *, seed: int, spread_interval=19, max_steps=900):
        from jepa_asteroids.cooperative_immune_savior import CooperativeImmuneSaviorGame
        self.world_id = str(world_id); self.seed = int(seed); self.max_steps = int(max_steps)
        self.spread_interval = int(spread_interval)
        self.game = CooperativeImmuneSaviorGame(self.seed, spread_interval=self.spread_interval,
                                                max_steps=self.max_steps)
        self.game.reset()

    def public_spec(self):
        frame = self.game.observe()[0]
        return PublicWorldSpec(self.world_id, "cooperative-immune-savior", IMMUNE_ADAPTER, 2,
            "two-local-rgb-views-channel-first", (2, *frame.shape),
            "two-anonymous-discrete-actions", 5, self.max_steps, (-1.0, 1.0))

    def reset(self, seed=None): return self.game.reset(seed)

    def step(self, actions):
        frames, reward, done, info = self.game.step(actions)
        public = {key: info[key] for key in
                  ("success", "failure", "healthy_preserved", "healthy_damaged", "disease_remaining")}
        return Transition(frames, reward, done, False, public)

    def human_frame(self):
        return np.concatenate([np.moveaxis(frame, 0, -1) for frame in self.game.observe()], axis=1)

    def audit_state(self): return self.game.audit_state()


def load_asteroids(folder: Path):
    value = _private(folder)
    return TwoRocketAsteroidsAdapter(value["world_id"], seed=value["seed"],
        difficulty=value.get("difficulty", 2), max_steps=value.get("max_steps", 360))


def load_immune(folder: Path):
    value = _private(folder)
    return CooperativeImmuneAdapter(value["world_id"], seed=value["seed"],
        spread_interval=value.get("spread_interval", 19), max_steps=value.get("max_steps", 900))


def create_builtin_world(root: Path, family: str, *, seed: int, **options) -> Path:
    """Create a split public/private package without exposing hidden world state."""
    family = str(family)
    if family == "asteroids":
        adapter, defaults = ASTEROIDS_ADAPTER, {"difficulty": 2, "max_steps": 360}
    elif family in ("immune", "immune-savior"):
        adapter, defaults = IMMUNE_ADAPTER, {"spread_interval": 19, "max_steps": 900}
        family = "immune-savior"
    else: raise ValueError(f"unknown built-in family {family!r}")
    config = defaults | {key: int(value) for key, value in options.items()}
    identity = hashlib.sha256(json.dumps({"family": family, "seed": int(seed), **config},
                                         sort_keys=True).encode()).hexdigest()[:16]
    world_id = f"{family}-{identity}"; folder = Path(root) / world_id; folder.mkdir(parents=True, exist_ok=True)
    private = {"format": "gum-private-world-v1", "world_id": world_id, "family": family,
               "seed": int(seed), **config}
    (folder / "genome.private.json").write_text(json.dumps(private, indent=2), encoding="utf-8")
    world = load_asteroids(folder) if family == "asteroids" else load_immune(folder)
    public = world.public_spec().to_json() | {"created_by": "GUM World Creator v1",
        "private_state_disclosed_to_mind": False}
    (folder / "world.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
    return folder
