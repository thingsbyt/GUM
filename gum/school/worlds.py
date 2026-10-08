"""Source-reviewed foundational worlds for GUM School.

The public contract exposes pixels, anonymous action slots, scalar rewards,
termination, and a deliberately small public-info dictionary.  Generators and
hidden state stay behind explicit loaders; world packages are strict JSON data,
never executable plugins.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from gum.protocol import PublicWorldSpec, Transition


OBJECT_LABORATORY_ADAPTER = "gum-object-laboratory-v1"
CAUSAL_WORKSHOP_ADAPTER = "gum-causal-workshop-v1"
CHANGING_MAZE_ADAPTER = "gum-changing-maze-v1"

FOUNDATIONAL_ADAPTERS = (
    OBJECT_LABORATORY_ADAPTER,
    CAUSAL_WORKSHOP_ADAPTER,
    CHANGING_MAZE_ADAPTER,
)

_SCHOOL_BY_ADAPTER = {
    OBJECT_LABORATORY_ADAPTER: "object-laboratory",
    CAUSAL_WORKSHOP_ADAPTER: "causal-workshop",
    CHANGING_MAZE_ADAPTER: "changing-maze",
}
_ALLOWED_MECHANISMS = {
    OBJECT_LABORATORY_ADAPTER: {"occlusion", "functional-category", "mixed"},
    CAUSAL_WORKSHOP_ADAPTER: {"controls", "composition", "mixed"},
    CHANGING_MAZE_ADAPTER: {"memory", "revision", "mixed"},
}


class FoundationalWorldError(ValueError):
    pass


def _identity(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _strict_json(path: Path, required: set[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FoundationalWorldError(f"invalid world data {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise FoundationalWorldError(f"{path.name} must contain an object")
    unknown = set(value) - required
    missing = required - set(value)
    if unknown or missing:
        raise FoundationalWorldError(
            f"{path.name} fields differ from the reviewed format; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return value


def _action(value: Any, count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise FoundationalWorldError("action must be an integer slot")
    result = int(value)
    if not 0 <= result < count:
        raise FoundationalWorldError(f"action must be in [0, {count})")
    return result


class _EpisodeWorld:
    adapter: str
    family: str
    action_count: int
    horizon: int
    shape: tuple[int, int, int]

    def __init__(self, world_id: str, *, seed: int, mechanism: str):
        self.world_id = str(world_id)
        self.seed = int(seed)
        self.mechanism = str(mechanism)
        self.steps = 0
        self._done = False
        self._last_observation = np.zeros(self.shape, dtype=np.uint8)

    def public_spec(self) -> PublicWorldSpec:
        return PublicWorldSpec(
            self.world_id,
            self.family,
            self.adapter,
            1,
            "rgb",
            self.shape,
            "discrete-anonymous",
            self.action_count,
            self.horizon,
            (-1.0, 1.0),
        )

    def _begin_reset(self, seed: int | None) -> np.random.Generator:
        episode_seed = self.seed if seed is None else int(seed)
        self._episode_seed = episode_seed
        self.steps = 0
        self._done = False
        return np.random.default_rng(episode_seed)

    def _ensure_live(self) -> None:
        if self._done:
            raise FoundationalWorldError("episode is over; reset before stepping again")

    def _finish(self, observation: np.ndarray, reward: float, terminated: bool, event: str) -> Transition:
        self.steps += 1
        truncated = self.steps >= self.horizon and not terminated
        self._done = bool(terminated or truncated)
        self._last_observation = np.asarray(observation, dtype=np.uint8).copy()
        public = {"success": bool(terminated and reward > 0), "event": str(event)}
        return Transition(self._last_observation.copy(), float(reward), bool(terminated), truncated, public)

    def human_frame(self) -> np.ndarray:
        return self._last_observation.copy()


class ObjectLaboratoryWorld(_EpisodeWorld):
    """Track a marked identity through motion, occlusion, and surface change."""

    adapter = OBJECT_LABORATORY_ADAPTER
    family = "object-laboratory"
    action_count = 6
    horizon = 160
    shape = (64, 64, 3)
    _slots = np.asarray([9.0, 24.0, 39.0, 54.0])
    _palette = ((76, 170, 255), (242, 103, 97), (108, 220, 145), (190, 126, 255))
    _changed = ((243, 154, 75), (83, 210, 211), (232, 99, 184), (164, 206, 75))
    _role_colors = ((240, 181, 64), (75, 210, 224))

    def reset(self, seed: int | None = None) -> np.ndarray:
        rng = self._begin_reset(seed)
        self.active_mechanism = self.mechanism
        if self.mechanism == "mixed":
            self.active_mechanism = (
                "functional-category" if self._episode_seed % 2 == 0 else "occlusion"
            )
        self.target = int(rng.integers(4))
        self.final_order = tuple(int(value) for value in rng.permutation(4))
        self.phase = 0
        self.functional_roles = tuple(int(value) for value in rng.permutation((0, 0, 1, 1)))
        self.target_role = int(rng.integers(2))
        self.probed: dict[int, int] = {}
        self.last_effect: int | None = None
        self._last_observation = self._render()
        return self._last_observation.copy()

    def _positions(self) -> np.ndarray:
        progress = min(self.phase, 6) / 6.0
        destinations = np.empty(4, dtype=float)
        for final_slot, identity in enumerate(self.final_order):
            destinations[identity] = self._slots[final_slot]
        return self._slots + (destinations - self._slots) * progress

    def _render(self) -> np.ndarray:
        if self.active_mechanism == "functional-category":
            return self._render_functional()
        image = Image.new("RGB", (64, 64), (10, 17, 25))
        draw = ImageDraw.Draw(image)
        for slot in self._slots:
            draw.line((int(slot), 3, int(slot), 60), fill=(25, 38, 49), width=1)
        positions = self._positions()
        y = 8 + self.phase * 8
        palette = self._palette if self.phase < 4 else self._changed
        for identity, x in enumerate(positions):
            x = int(round(float(x)))
            color = palette[identity]
            if identity % 2:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            else:
                draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill=color)
            if self.phase == 0 and identity == self.target:
                draw.rectangle((x - 6, y - 6, x + 6, y + 6), outline=(250, 212, 60), width=2)
        draw.rectangle((0, 23, 63, 41), fill=(67, 72, 83))
        draw.line((0, 22, 63, 22), fill=(125, 134, 145), width=1)
        draw.line((0, 42, 63, 42), fill=(125, 134, 145), width=1)
        return np.asarray(image, dtype=np.uint8)

    def _render_functional(self) -> np.ndarray:
        image = Image.new("RGB", (64, 64), (10, 17, 25))
        draw = ImageDraw.Draw(image)
        role_color = self._role_colors[self.target_role]
        draw.rectangle((3, 3, 14, 14), fill=role_color, outline=(225, 232, 238))
        if self.target_role == 0:
            draw.line((8, 5, 8, 12), fill=(18, 25, 32), width=2)
        else:
            draw.line((5, 8, 12, 8), fill=(18, 25, 32), width=2)
        y = 30 if self.phase == 0 else 48
        palette = self._palette if self.phase == 0 else self._changed
        order = tuple(range(4)) if self.phase == 0 else self.final_order
        for slot, identity in enumerate(order):
            x = int(self._slots[slot])
            color = palette[identity]
            if identity % 2:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            else:
                draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill=color)
        if self.last_effect is not None:
            effect = self._role_colors[self.last_effect]
            draw.rectangle((25, 4, 38, 17), fill=effect, outline=(225, 232, 238))
            if self.last_effect == 0:
                draw.line((31, 6, 31, 15), fill=(18, 25, 32), width=2)
            else:
                draw.line((27, 10, 36, 10), fill=(18, 25, 32), width=2)
        return np.asarray(image, dtype=np.uint8)

    def step(self, action: Any) -> Transition:
        self._ensure_live()
        slot = _action(action, self.action_count)
        if self.active_mechanism == "functional-category":
            return self._step_functional(slot)
        event = "advanced"
        reward = -0.01 if slot == 0 else -0.02
        terminated = False
        if self.phase < 6:
            self.phase += 1
            event = "observed" if slot == 1 else "advanced"
        elif slot >= 2:
            chosen_slot = slot - 2
            target_slot = self.final_order.index(self.target)
            terminated = True
            reward = 1.0 if chosen_slot == target_slot else -1.0
            event = "accepted" if reward > 0 else "rejected"
        else:
            event = "observed"
        return self._finish(self._render(), reward, terminated, event)

    def _step_functional(self, slot: int) -> Transition:
        reward = -0.01
        terminated = False
        if self.phase == 0 and slot >= 2:
            identity = slot - 2
            self.last_effect = self.functional_roles[identity]
            self.probed[identity] = self.last_effect
            reward = 0.02
            event = "effect-observed"
        elif self.phase == 0:
            self.phase = 6
            self.last_effect = None
            event = "appearance-shifted"
        elif slot >= 2:
            identity = self.final_order[slot - 2]
            reward = 1.0 if self.functional_roles[identity] == self.target_role else -1.0
            terminated = True
            event = "accepted" if reward > 0 else "rejected"
        else:
            event = "observed"
        return self._finish(self._render(), reward, terminated, event)

    def audit_state(self) -> dict[str, Any]:
        return {
            "episode_seed": self._episode_seed,
            "target_identity": self.target,
            "active_mechanism": self.active_mechanism,
            "final_order": list(self.final_order),
            "functional_roles": list(self.functional_roles),
            "target_role": self.target_role,
            "probed": dict(self.probed),
            "phase": self.phase,
            "steps": self.steps,
        }


class CausalWorkshopWorld(_EpisodeWorld):
    """Discover shuffled controls whose useful effects have ordered prerequisites."""

    adapter = CAUSAL_WORKSHOP_ADAPTER
    family = "causal-workshop"
    action_count = 8
    horizon = 200
    shape = (72, 72, 3)

    def reset(self, seed: int | None = None) -> np.ndarray:
        rng = self._begin_reset(seed)
        permutation = [int(value) for value in rng.permutation(self.action_count)]
        self.active_mechanism = self.mechanism
        if self.mechanism == "mixed":
            self.active_mechanism = "composition" if self._episode_seed % 2 == 0 else "controls"
        self.advance_actions = (
            tuple(permutation[:3]) if self.active_mechanism == "composition"
            else (permutation[0], permutation[0], permutation[0])
        )
        self.reset_action = permutation[3]
        self.back_action = permutation[4]
        self.stage = 0
        self.last_action: int | None = None
        self.last_effect = "none"
        self._last_observation = self._render()
        return self._last_observation.copy()

    def _render(self) -> np.ndarray:
        image = Image.new("RGB", (72, 72), (11, 18, 25))
        draw = ImageDraw.Draw(image)
        for index, x in enumerate((18, 36, 54)):
            color = (61, 220, 132) if index < self.stage else (63, 72, 82)
            draw.ellipse((x - 6, 11, x + 6, 23), fill=color, outline=(145, 157, 168))
            if index:
                draw.line((x - 18 + 7, 17, x - 7, 17), fill=(103, 117, 129), width=2)
        for action in range(8):
            x0 = 4 + (action % 4) * 17
            y0 = 40 + (action // 4) * 16
            fill = (76, 126, 171) if action == self.last_action else (36, 52, 65)
            draw.rectangle((x0, y0, x0 + 13, y0 + 10), fill=fill, outline=(98, 120, 137))
        return np.asarray(image, dtype=np.uint8)

    def step(self, action: Any) -> Transition:
        self._ensure_live()
        slot = _action(action, self.action_count)
        self.last_action = slot
        old_stage = self.stage
        if self.stage < 3 and slot == self.advance_actions[self.stage]:
            self.stage += 1
            self.last_effect = "progress"
        elif slot == self.reset_action:
            self.stage = 0
            self.last_effect = "regression"
        elif slot == self.back_action and self.stage:
            self.stage -= 1
            self.last_effect = "regression"
        else:
            self.last_effect = "no-change"
        terminated = self.stage == 3
        if terminated:
            reward = 1.0
            event = "completed"
        elif self.stage > old_stage:
            reward = 0.2
            event = "progress"
        elif self.stage < old_stage:
            reward = -0.1
            event = "regression"
        else:
            reward = -0.01
            event = "no-change"
        return self._finish(self._render(), reward, terminated, event)

    def audit_state(self) -> dict[str, Any]:
        return {
            "episode_seed": self._episode_seed,
            "active_mechanism": self.active_mechanism,
            "advance_actions": list(self.advance_actions),
            "reset_action": self.reset_action,
            "back_action": self.back_action,
            "stage": self.stage,
            "steps": self.steps,
        }


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _maze_path(size: int, walls: set[tuple[int, int]], start: tuple[int, int], goal: tuple[int, int]):
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
            if (0 <= nxt[0] < size and 0 <= nxt[1] < size
                    and nxt not in walls and nxt not in parent):
                parent[nxt] = cell
                queue.append(nxt)
    return None


class ChangingMazeWorld(_EpisodeWorld):
    """Local visual navigation with shuffled controls and an observable revision."""

    adapter = CHANGING_MAZE_ADAPTER
    family = "changing-maze"
    action_count = 6
    horizon = 240
    size = 11
    scale = 8
    shape = (size * scale, size * scale, 3)

    def _make_layout(self, rng: np.random.Generator):
        start, goal = (1, 1), (self.size - 2, self.size - 2)
        boundary = {
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if row in {0, self.size - 1} or col in {0, self.size - 1}
        }
        interior = [(row, col) for row in range(1, self.size - 1)
                    for col in range(1, self.size - 1) if (row, col) not in {start, goal}]
        for _ in range(100):
            walls = set(boundary)
            for cell in interior:
                if rng.random() < 0.20:
                    walls.add(cell)
            path = _maze_path(self.size, walls, start, goal)
            if path is None or len(path) < 15:
                continue
            candidates = []
            for cell in path[4:-4]:
                changed = walls | {cell}
                if _maze_path(self.size, changed, start, goal) is not None:
                    candidates.append(cell)
            if candidates:
                return walls, start, goal, candidates[int(rng.integers(len(candidates)))]
        raise FoundationalWorldError("could not generate a solvable changing maze")

    def reset(self, seed: int | None = None) -> np.ndarray:
        rng = self._begin_reset(seed)
        self.walls, self.position, self.goal, self.change_cell = self._make_layout(rng)
        self.initial_walls = set(self.walls)
        self.action_map = tuple(int(value) for value in rng.permutation(4))
        self.initial_action_map = self.action_map
        self.revision_enabled = self.mechanism == "revision" or (
            self.mechanism == "mixed" and self._episode_seed % 2 == 0
        )
        self.change_step = 8
        self.changed = False
        self._last_observation = self._render()
        return self._last_observation.copy()

    def _visible(self, cell: tuple[int, int]) -> bool:
        return max(abs(cell[0] - self.position[0]), abs(cell[1] - self.position[1])) <= 2

    def _render(self) -> np.ndarray:
        image = Image.new("RGB", (self.shape[1], self.shape[0]), (5, 8, 12))
        draw = ImageDraw.Draw(image)
        for row in range(self.size):
            for col in range(self.size):
                cell = (row, col)
                if not self._visible(cell):
                    continue
                x0, y0 = col * self.scale, row * self.scale
                if cell in self.walls:
                    fill = (83, 92, 101)
                elif cell == self.goal:
                    fill = (58, 211, 126)
                else:
                    fill = (22, 31, 40)
                draw.rectangle((x0, y0, x0 + self.scale - 1, y0 + self.scale - 1), fill=fill)
                draw.rectangle((x0, y0, x0 + self.scale - 1, y0 + self.scale - 1), outline=(31, 45, 57))
        row, col = self.position
        x0, y0 = col * self.scale, row * self.scale
        draw.rectangle((x0 + 2, y0 + 2, x0 + 5, y0 + 5), fill=(69, 181, 255))
        if self.changed and self.steps == self.change_step:
            draw.rectangle((0, 0, self.shape[1] - 1, self.shape[0] - 1), outline=(246, 159, 65), width=2)
        return np.asarray(image, dtype=np.uint8)

    def step(self, action: Any) -> Transition:
        self._ensure_live()
        slot = _action(action, self.action_count)
        event = "observed"
        reward = -0.01
        if slot < 4:
            semantic = self.action_map[slot]
            dr, dc = _DIRECTIONS[semantic]
            nxt = self.position[0] + dr, self.position[1] + dc
            if nxt in self.walls:
                reward = -0.04
                event = "blocked"
            else:
                self.position = nxt
                event = "moved"
        if self.revision_enabled and not self.changed and self.steps + 1 == self.change_step:
            self.walls.add(self.change_cell)
            self.action_map = tuple((direction + 1) % 4 for direction in self.action_map)
            self.changed = True
            event = "world-changed"
        terminated = self.position == self.goal
        if terminated:
            reward = 1.0
            event = "goal-reached"
        return self._finish(self._render(), reward, terminated, event)

    def audit_state(self) -> dict[str, Any]:
        return {
            "episode_seed": self._episode_seed,
            "position": list(self.position),
            "goal": list(self.goal),
            "walls": [list(cell) for cell in sorted(self.walls)],
            "initial_action_map": list(self.initial_action_map),
            "action_map": list(self.action_map),
            "change_cell": list(self.change_cell),
            "revision_enabled": self.revision_enabled,
            "changed": self.changed,
            "steps": self.steps,
        }


_WORLD_CLASSES = {
    OBJECT_LABORATORY_ADAPTER: ObjectLaboratoryWorld,
    CAUSAL_WORKSHOP_ADAPTER: CausalWorkshopWorld,
    CHANGING_MAZE_ADAPTER: ChangingMazeWorld,
}


def create_foundational_world(
    root: Path,
    adapter: str,
    *,
    seed: int,
    mechanism: str = "mixed",
) -> Path:
    """Create a strict split public/private package for one reviewed adapter."""
    adapter = str(adapter)
    if adapter not in _WORLD_CLASSES:
        raise FoundationalWorldError(f"unsupported foundational adapter {adapter!r}")
    if mechanism not in _ALLOWED_MECHANISMS[adapter]:
        raise FoundationalWorldError(f"unsupported mechanism {mechanism!r} for {adapter}")
    private = {
        "format": "gum-school-world-data-v1",
        "adapter": adapter,
        "school": _SCHOOL_BY_ADAPTER[adapter],
        "seed": int(seed),
        "mechanism": mechanism,
    }
    world_id = f"{private['school']}-{_identity(private)}"
    folder = Path(root) / world_id
    folder.mkdir(parents=True, exist_ok=False)
    private["world_id"] = world_id
    (folder / "genome.private.json").write_text(
        json.dumps(private, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    world = _WORLD_CLASSES[adapter](world_id, seed=int(seed), mechanism=mechanism)
    public = world.public_spec().to_json() | {
        "created_by": "GUM School reviewed generator v1",
        "private_state_disclosed_to_mind": False,
    }
    (folder / "world.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return folder


def _load_foundational(folder: Path, expected_adapter: str):
    folder = Path(folder)
    private = _strict_json(
        folder / "genome.private.json",
        {"format", "adapter", "school", "seed", "mechanism", "world_id"},
    )
    if private["format"] != "gum-school-world-data-v1":
        raise FoundationalWorldError("unsupported foundational world data format")
    if private["adapter"] != expected_adapter:
        raise FoundationalWorldError("world data names a different adapter")
    if private["school"] != _SCHOOL_BY_ADAPTER[expected_adapter]:
        raise FoundationalWorldError("world data names a different school")
    if private["mechanism"] not in _ALLOWED_MECHANISMS[expected_adapter]:
        raise FoundationalWorldError("world data requests an unreviewed mechanism")
    if isinstance(private["seed"], bool) or not isinstance(private["seed"], int):
        raise FoundationalWorldError("world seed must be an integer")
    expected_id = f"{private['school']}-{_identity({key: private[key] for key in ('format', 'adapter', 'school', 'seed', 'mechanism')})}"
    if private["world_id"] != expected_id or folder.name != expected_id:
        raise FoundationalWorldError("world identity does not match its reviewed data")
    public = _strict_json(
        folder / "world.json",
        set(asdict(_WORLD_CLASSES[expected_adapter](expected_id, seed=private["seed"], mechanism=private["mechanism"]).public_spec()))
        | {"created_by", "private_state_disclosed_to_mind"},
    )
    world = _WORLD_CLASSES[expected_adapter](
        expected_id, seed=private["seed"], mechanism=private["mechanism"]
    )
    expected_public = world.public_spec().to_json() | {
        "created_by": "GUM School reviewed generator v1",
        "private_state_disclosed_to_mind": False,
    }
    expected_public = json.loads(json.dumps(expected_public))
    if public != expected_public:
        raise FoundationalWorldError("public world record differs from the reviewed adapter contract")
    return world


def load_object_laboratory(folder: Path) -> ObjectLaboratoryWorld:
    return _load_foundational(folder, OBJECT_LABORATORY_ADAPTER)


def load_causal_workshop(folder: Path) -> CausalWorkshopWorld:
    return _load_foundational(folder, CAUSAL_WORKSHOP_ADAPTER)


def load_changing_maze(folder: Path) -> ChangingMazeWorld:
    return _load_foundational(folder, CHANGING_MAZE_ADAPTER)


FOUNDATIONAL_LOADERS = {
    OBJECT_LABORATORY_ADAPTER: load_object_laboratory,
    CAUSAL_WORKSHOP_ADAPTER: load_causal_workshop,
    CHANGING_MAZE_ADAPTER: load_changing_maze,
}

GENERATOR_MECHANISMS = {
    "authored-object-occlusion-v1": (OBJECT_LABORATORY_ADAPTER, "occlusion"),
    "authored-object-function-v1": (OBJECT_LABORATORY_ADAPTER, "functional-category"),
    "authored-causal-controls-v1": (CAUSAL_WORKSHOP_ADAPTER, "controls"),
    "authored-causal-composition-v1": (CAUSAL_WORKSHOP_ADAPTER, "composition"),
    "authored-maze-memory-v1": (CHANGING_MAZE_ADAPTER, "memory"),
    "authored-maze-revision-v1": (CHANGING_MAZE_ADAPTER, "revision"),
}


def create_lesson_world(root: Path, lesson: dict[str, Any], *, seed: int) -> Path:
    """Resolve one reviewed curriculum generator to its explicit adapter."""
    if not isinstance(lesson, dict):
        raise FoundationalWorldError("lesson must be a validated object")
    training = lesson.get("training")
    if not isinstance(training, dict):
        raise FoundationalWorldError("lesson has no training generator")
    generator = training.get("generator")
    resolved = GENERATOR_MECHANISMS.get(generator)
    if resolved is None:
        raise FoundationalWorldError(f"unregistered lesson generator {generator!r}")
    adapter, mechanism = resolved
    if lesson.get("adapter") != adapter:
        raise FoundationalWorldError("lesson generator and adapter do not match")
    return create_foundational_world(root, adapter, seed=int(seed), mechanism=mechanism)
