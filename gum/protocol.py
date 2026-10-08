"""Small protocol shared by minds, worlds, and the GUM harness."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class PublicWorldSpec:
    world_id: str
    family: str
    adapter: str
    agents: int
    observation_kind: str
    observation_shape: tuple[int, ...]
    action_kind: str
    action_count: int
    horizon: int
    reward_range: tuple[float, float]
    protocol_version: int = 1

    def to_json(self): return asdict(self)


@dataclass(frozen=True)
class Transition:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    public_info: dict[str, Any]


class WorldAdapter(Protocol):
    """Only these methods are visible to a mind."""
    def public_spec(self) -> PublicWorldSpec: ...
    def reset(self, seed: int | None = None) -> Any: ...
    def step(self, action: Any) -> Transition: ...


class AuditableWorld(WorldAdapter, Protocol):
    """Harness-only methods. Their results must never be passed to a mind."""
    def human_frame(self) -> np.ndarray: ...
    def audit_state(self) -> dict[str, Any]: ...


class Mind(Protocol):
    def begin(self, spec: PublicWorldSpec, observation: Any, *, training: bool) -> None: ...
    def act(self, observation: Any, *, training: bool) -> Any: ...
    def observe(self, action: Any, transition: Transition, *, training: bool) -> None: ...
    def status(self) -> dict[str, Any]: ...
    def save(self, path: Path) -> None: ...
