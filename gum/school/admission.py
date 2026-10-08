"""Deterministic admission audit for the three foundational school worlds."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from gum.lineage import canonical, file_sha256

from .validation import DEFAULT_CURRICULUM, PROHIBITED_SOLUTION_KEYS, load_json
from .worlds import (
    CAUSAL_WORKSHOP_ADAPTER,
    CHANGING_MAZE_ADAPTER,
    FOUNDATIONAL_LOADERS,
    OBJECT_LABORATORY_ADAPTER,
    create_foundational_world,
)


class WorldAdmissionError(RuntimeError):
    pass


_SAFE_ACTION = {
    OBJECT_LABORATORY_ADAPTER: 0,
    CAUSAL_WORKSHOP_ADAPTER: 0,
    CHANGING_MAZE_ADAPTER: 5,
}


def _array_hash(value: Any) -> str:
    array = np.asarray(value)
    payload = canonical({"dtype": str(array.dtype), "shape": list(array.shape)}) + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _transition_row(transition) -> dict[str, Any]:
    return {
        "observation_sha256": _array_hash(transition.observation),
        "reward": transition.reward,
        "terminated": transition.terminated,
        "truncated": transition.truncated,
        "public_info": transition.public_info,
    }


def _episode_trace(world, seed: int, actions: list[int]) -> dict[str, Any]:
    observation = world.reset(seed)
    rows = []
    for action in actions:
        transition = world.step(action)
        rows.append({"action": action, **_transition_row(transition)})
        if transition.terminated or transition.truncated:
            break
    return {"initial_observation_sha256": _array_hash(observation), "transitions": rows}


class _ObjectPolicy:
    def reset(self, observation: np.ndarray) -> None:
        yellow = np.all(observation == np.asarray((250, 212, 60), dtype=np.uint8), axis=2)
        xs = np.where(yellow)[1]
        self.mode = "occlusion" if len(xs) else "functional-category"
        self.steps = 0
        if self.mode == "occlusion":
            self.start_x = float(np.mean(xs))
            center_x = int(round(self.start_x))
            self.target_color = tuple(int(value) for value in observation[8, center_x])
            self.second_x: float | None = None
        else:
            role_colors = ((240, 181, 64), (75, 210, 224))
            target_color = tuple(int(value) for value in observation[5, 5])
            self.target_role = role_colors.index(target_color)
            self.probe_index = 0
            self.roles: dict[int, int] = {}
            self.shifted = False

    def act(self, observation: np.ndarray) -> int:
        if self.mode == "functional-category":
            if self.probe_index < 4:
                return 2 + self.probe_index
            if not self.shifted:
                return 0
            changed = ((243, 154, 75), (83, 210, 211), (232, 99, 184), (164, 206, 75))
            for slot, x in enumerate((9, 24, 39, 54)):
                color = tuple(int(value) for value in observation[48, x])
                if color in changed:
                    identity = changed.index(color)
                    if self.roles[identity] == self.target_role:
                        return 2 + slot
            raise WorldAdmissionError("object baseline could not resolve a public function category")
        if self.steps < 6:
            return 0
        velocity = 0.0 if self.second_x is None else self.second_x - self.start_x
        predicted = self.start_x + 6.0 * velocity
        slots = np.asarray([9.0, 24.0, 39.0, 54.0])
        return 2 + int(np.argmin(np.abs(slots - predicted)))

    def observe(self, _action: int, transition) -> None:
        self.steps += 1
        if self.mode == "functional-category":
            if self.probe_index < 4:
                effect = tuple(int(value) for value in transition.observation[5, 26])
                role_colors = ((240, 181, 64), (75, 210, 224))
                self.roles[self.probe_index] = role_colors.index(effect)
                self.probe_index += 1
            elif not self.shifted:
                self.shifted = True
            return
        if self.steps == 1:
            color = np.asarray(self.target_color, dtype=np.uint8)
            mask = np.all(transition.observation == color, axis=2)
            xs = np.where(mask)[1]
            if len(xs):
                self.second_x = float(np.mean(xs))


def _causal_stage(observation: np.ndarray) -> int:
    green = np.asarray((61, 220, 132), dtype=np.uint8)
    return sum(bool(np.array_equal(observation[17, x], green)) for x in (18, 36, 54))


class _CausalPolicy:
    def reset(self, observation: np.ndarray) -> None:
        self.stage = _causal_stage(observation)
        self.known: dict[int, int] = {}
        self.tried = {stage: set() for stage in range(3)}

    def act(self, observation: np.ndarray) -> int:
        self.stage = _causal_stage(observation)
        if self.stage in self.known:
            return self.known[self.stage]
        for action in range(8):
            if action not in self.tried[self.stage]:
                return action
        self.tried[self.stage].clear()
        return 0

    def observe(self, action: int, transition) -> None:
        old_stage = self.stage
        new_stage = _causal_stage(transition.observation)
        self.tried[old_stage].add(action)
        if new_stage == old_stage + 1:
            self.known[old_stage] = action
        self.stage = new_stage


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _shortest_path(
    start: tuple[int, int],
    target: tuple[int, int],
    open_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == target:
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


class _MazePolicy:
    size = 11
    scale = 8
    unseen = (5, 8, 12)
    wall = (83, 92, 101)
    goal_color = (58, 211, 126)
    agent = (69, 181, 255)

    def reset(self, observation: np.ndarray) -> None:
        self.open: set[tuple[int, int]] = set()
        self.walls: set[tuple[int, int]] = set()
        self.goal: tuple[int, int] | None = None
        self.action_map: dict[int, tuple[int, int]] = {}
        self.calibration_tried: set[int] = set()
        self.last_position: tuple[int, int] | None = None
        self.position = (0, 0)
        self._update_map(observation)

    def _cell_color(self, observation: np.ndarray, row: int, col: int) -> tuple[int, int, int]:
        return tuple(int(value) for value in observation[
            row * self.scale + self.scale // 2,
            col * self.scale + self.scale // 2,
        ])

    def _agent_position(self, observation: np.ndarray) -> tuple[int, int]:
        mask = np.all(observation == np.asarray(self.agent, dtype=np.uint8), axis=2)
        ys, xs = np.where(mask)
        if not len(xs):
            raise WorldAdmissionError("maze baseline could not locate the public agent glyph")
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
                    self.open.discard(cell)
                else:
                    self.open.add(cell)
                    self.walls.discard(cell)
                    if color == self.goal_color:
                        self.goal = cell

    def _desired_direction(self) -> tuple[int, int]:
        if self.goal is not None:
            path = _shortest_path(self.position, self.goal, self.open)
            if path and len(path) > 1:
                return path[1][0] - self.position[0], path[1][1] - self.position[1]

        reachable = []
        for cell in self.open:
            path = _shortest_path(self.position, cell, self.open)
            if path is None:
                continue
            unknown = []
            for direction in _DIRECTIONS:
                nxt = cell[0] + direction[0], cell[1] + direction[1]
                if nxt not in self.open and nxt not in self.walls:
                    unknown.append(direction)
            if unknown:
                reachable.append((len(path), cell, path, unknown))
        if reachable:
            _, cell, path, unknown = min(reachable, key=lambda row: (row[0], row[1]))
            if cell == self.position:
                return unknown[0]
            return path[1][0] - self.position[0], path[1][1] - self.position[1]
        return _DIRECTIONS[0]

    def act(self, _observation: np.ndarray) -> int:
        desired = self._desired_direction()
        for action, direction in self.action_map.items():
            if direction == desired:
                return action
        for action in range(4):
            if action not in self.action_map and action not in self.calibration_tried:
                self.calibration_tried.add(action)
                return action
        if len(self.action_map) < 4:
            self.calibration_tried.clear()
            for action in range(4):
                if action not in self.action_map:
                    self.calibration_tried.add(action)
                    return action
        self.action_map.clear()
        self.calibration_tried.clear()
        return 0

    def observe(self, action: int, transition) -> None:
        previous = self.position
        self._update_map(transition.observation)
        delta = self.position[0] - previous[0], self.position[1] - previous[1]
        if transition.public_info.get("event") == "world-changed":
            self.action_map.clear()
            self.calibration_tried.clear()
            return
        if delta in _DIRECTIONS:
            expected = self.action_map.get(action)
            if expected is not None and expected != delta:
                self.action_map.clear()
            for other, direction in list(self.action_map.items()):
                if direction == delta and other != action:
                    del self.action_map[other]
            self.action_map[action] = delta
            self.calibration_tried.discard(action)


_POLICIES = {
    OBJECT_LABORATORY_ADAPTER: _ObjectPolicy,
    CAUSAL_WORKSHOP_ADAPTER: _CausalPolicy,
    CHANGING_MAZE_ADAPTER: _MazePolicy,
}


def _run_policy(world, seed: int, policy=None, *, random_seed: int | None = None) -> dict[str, Any]:
    observation = world.reset(seed)
    if policy is not None:
        policy.reset(observation)
    rng = np.random.default_rng(random_seed)
    total = 0.0
    for step in range(world.public_spec().horizon):
        action = (int(rng.integers(world.public_spec().action_count))
                  if policy is None else policy.act(observation))
        transition = world.step(action)
        if policy is not None:
            policy.observe(action, transition)
        observation = transition.observation
        total += transition.reward
        if transition.terminated or transition.truncated:
            return {
                "success": bool(transition.public_info["success"]),
                "steps": step + 1,
                "return": total,
            }
    raise WorldAdmissionError("world exceeded its declared horizon")


def _contains_prohibited_key(value: Any) -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in PROHIBITED_SOLUTION_KEYS:
                hits.append(str(key))
            hits.extend(_contains_prohibited_key(child))
    elif isinstance(value, list):
        for child in value:
            hits.extend(_contains_prohibited_key(child))
    return hits


def _check_one(adapter: str, admission: dict[str, Any], root: Path, trials: int) -> dict[str, Any]:
    package = create_foundational_world(root, adapter, seed=admission["seed_partitions"]["training"][0])
    loader = FOUNDATIONAL_LOADERS[adapter]
    world = loader(package)
    declaration = admission["declaration"]
    spec = world.public_spec()
    declared = {
        "observation_kind": spec.observation_kind,
        "action_kind": spec.action_kind,
        "action_count": spec.action_count,
        "horizon": spec.horizon,
        "reward_minimum": spec.reward_range[0],
        "reward_maximum": spec.reward_range[1],
    }
    declaration_matches = declared == declaration

    seed = admission["seed_partitions"]["development"][0]
    actions = [_SAFE_ACTION[adapter]] * 16
    deterministic_reset = _episode_trace(loader(package), seed, actions) == _episode_trace(
        loader(package), seed, actions
    )

    horizon_world = loader(package)
    horizon_world.reset(seed)
    final = None
    for _ in range(spec.horizon):
        final = horizon_world.step(_SAFE_ACTION[adapter])
        if final.terminated or final.truncated:
            break
    hard_horizon = bool(final and final.truncated and horizon_world.steps == spec.horizon)

    inspection_world = loader(package)
    observation = inspection_world.reset(seed)
    audit_before = inspection_world.audit_state()
    expected_audit = deepcopy(audit_before)
    frame = inspection_world.human_frame()
    observation[:] = 255
    frame[:] = 255
    audit_before["tampered"] = True
    safe_inspection = (
        inspection_world.audit_state() == expected_audit
        and inspection_world.human_frame().dtype == np.uint8
        and inspection_world.human_frame().shape == spec.observation_shape
    )

    replay_rng = np.random.default_rng(seed + 81_337)
    replay_actions = [int(value) for value in replay_rng.integers(0, spec.action_count, size=40)]
    replay = _episode_trace(loader(package), seed, replay_actions) == _episode_trace(
        loader(package), seed, replay_actions
    )

    public_package = json.loads((package / "world.json").read_text(encoding="utf-8"))
    leakage_world = loader(package)
    initial = leakage_world.reset(seed)
    transition = leakage_world.step(_SAFE_ACTION[adapter])
    hidden_state_leakage = (
        not _contains_prohibited_key(public_package)
        and initial.dtype == np.uint8
        and initial.shape == spec.observation_shape
        and set(transition.public_info) == {"success", "event"}
        and not _contains_prohibited_key(transition.public_info)
    )
    data_only = sorted(path.suffix for path in package.iterdir()) == [".json", ".json"]

    base_seeds = admission["seed_partitions"]["development"]
    trial_seeds = [base_seeds[index % len(base_seeds)] + 10_000 * (index // len(base_seeds))
                   for index in range(trials)]
    random_rows = [
        _run_policy(loader(package), trial_seed, random_seed=trial_seed + 700_001)
        for trial_seed in trial_seeds
    ]
    scripted_rows = [
        _run_policy(loader(package), trial_seed, _POLICIES[adapter]())
        for trial_seed in trial_seeds
    ]

    def summarize(rows):
        return {
            "trials": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_rate": sum(row["success"] for row in rows) / len(rows),
            "mean_steps": sum(row["steps"] for row in rows) / len(rows),
            "mean_return": sum(row["return"] for row in rows) / len(rows),
        }

    random_summary = summarize(random_rows)
    scripted_summary = summarize(scripted_rows)
    random_nontrivial = (
        random_summary["success_rate"] < 0.90
        or random_summary["mean_steps"] >= scripted_summary["mean_steps"] * 1.50
    )
    scripted_nontrivial = (
        scripted_summary["success_rate"] >= 0.75
        and (
            scripted_summary["success_rate"] - random_summary["success_rate"] >= 0.20
            or scripted_summary["mean_steps"] <= random_summary["mean_steps"] * 0.50
        )
    )
    checks = {
        "public_contract": declaration_matches,
        "deterministic_reset": deterministic_reset,
        "hard_horizon": hard_horizon,
        "safe_inspection": safe_inspection,
        "replay": replay,
        "hidden_state_leakage": hidden_state_leakage,
        "world_packages_are_data": data_only,
        "random_baseline": random_nontrivial,
        "scripted_baseline": scripted_nontrivial,
    }
    return {
        "admission_id": admission["admission_id"],
        "adapter": adapter,
        "school": admission["school"],
        "checks": checks,
        "passed": all(checks.values()),
        "baselines": {"random": random_summary, "scripted-public-observation": scripted_summary},
        "audit_boundary": {
            "policy_inputs": ["rgb-pixels", "anonymous-action-slots", "scalar-reward", "termination-signal", "public-info"],
            "policy_received_audit_state": False,
            "public_info_fields": ["event", "success"],
            "world_package_files": ["genome.private.json", "world.json"],
        },
        "replay_digest": hashlib.sha256(canonical(_episode_trace(loader(package), seed, replay_actions))).hexdigest(),
    }


def run_foundational_admission(
    output_path: Path | None = None,
    *,
    curriculum_path: Path = DEFAULT_CURRICULUM,
    working_directory: Path | None = None,
    trials: int = 24,
) -> dict[str, Any]:
    """Audit all foundational adapters and optionally save canonical evidence."""
    if trials < 4:
        raise WorldAdmissionError("at least four baseline trials are required")
    curriculum = load_json(curriculum_path)
    admissions = {row["adapter"]: row for row in curriculum["world_admissions"]}
    repository_root = Path(__file__).resolve().parents[2]

    def execute(root: Path):
        rows = []
        for adapter in (OBJECT_LABORATORY_ADAPTER, CAUSAL_WORKSHOP_ADAPTER, CHANGING_MAZE_ADAPTER):
            rows.append(_check_one(adapter, admissions[adapter], root, trials))
        source_paths = [
            repository_root / "gum" / "protocol.py",
            repository_root / "gum" / "school" / "worlds.py",
            repository_root / "gum" / "school" / "admission.py",
        ]
        report = {
            "format": "gum-school-world-admission-evidence-v1",
            "phase": 2,
            "scope": "foundational-adapters",
            "training_performed": False,
            "sealed_worlds_generated": False,
            "baseline_trials_per_adapter": trials,
            "source_hashes": {
                path.relative_to(repository_root).as_posix(): f"sha256:{file_sha256(path)}"
                for path in source_paths
            },
            "adapters": rows,
            "passed": all(row["passed"] for row in rows),
        }
        if not report["passed"]:
            failures = {
                row["adapter"]: [name for name, passed in row["checks"].items() if not passed]
                for row in rows if not row["passed"]
            }
            raise WorldAdmissionError(f"foundational admission failed: {failures}")
        return report

    if working_directory is None:
        temporary_parent = Path(output_path).parent if output_path is not None else Path.cwd()
        temporary_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="gum-school-admission-", dir=temporary_parent
        ) as temporary:
            report = execute(Path(temporary))
    else:
        root = Path(working_directory)
        root.mkdir(parents=True, exist_ok=True)
        report = execute(root)
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical(report) + b"\n")
    return report
