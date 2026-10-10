"""Authoritative four-body Room A for the cooperative escape experiment.

This module owns physics and pixels, never policy.  Each active body supplies
one anonymous action per joint tick.  The shared action map is randomized at
reset and is available only through the audit interface used by environment
tests.  Learner views contain pixels, not coordinates, tile labels, routes, or
the plate/gate dependency graph.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gum.protocol import PublicWorldSpec, Transition


TEAM_SIZE = 4
ACTION_COUNT = 5
ROOM_A_ADAPTER = "gum-cooperative-escape-room-a-v1"
CONTRACT_VERSION = 1
WAIT, NORTH, EAST, SOUTH, WEST = "wait", "north", "east", "south", "west"
SEMANTIC_ACTIONS = (WAIT, NORTH, EAST, SOUTH, WEST)
DELTAS = {
    WAIT: (0, 0),
    NORTH: (0, -1),
    EAST: (1, 0),
    SOUTH: (0, 1),
    WEST: (-1, 0),
}
MEMBER_NAMES = ("Fortis-1", "Fortis-2", "Fortis-3", "Fortis-4")
MEMBER_COLORS = ((235, 174, 55), (50, 198, 222), (157, 105, 239), (157, 217, 69))
MEMBER_SHAPES = ("hexagon", "circle", "rounded-square", "triangle")


class EscapeChamberError(ValueError):
    pass


@dataclass(frozen=True)
class RewardTable:
    treatment: str = "team"
    escape_bonus: float = 1.0
    team_completion_bonus: float = 1.0
    active_tick_cost: float = 0.001
    mixed_team_weight: float = 0.5
    mixed_individual_weight: float = 0.5

    def validate(self) -> None:
        if self.treatment not in {"team", "individual", "mixed"}:
            raise EscapeChamberError("unknown incentive treatment")
        if min(self.escape_bonus, self.team_completion_bonus, self.active_tick_cost) < 0:
            raise EscapeChamberError("reward magnitudes cannot be negative")
        if self.mixed_team_weight < 0 or self.mixed_individual_weight < 0:
            raise EscapeChamberError("mixed reward weights cannot be negative")
        if not np.isclose(self.mixed_team_weight + self.mixed_individual_weight, 1.0):
            raise EscapeChamberError("mixed reward weights must sum to one")

    def to_json(self) -> dict[str, Any]:
        return {
            "treatment": self.treatment,
            "escape_bonus": self.escape_bonus,
            "team_completion_bonus": self.team_completion_bonus,
            "active_tick_cost": self.active_tick_cost,
            "mixed_team_weight": self.mixed_team_weight,
            "mixed_individual_weight": self.mixed_individual_weight,
        }


@dataclass(frozen=True)
class JointStep:
    observations: tuple[np.ndarray, ...]
    rewards: tuple[float, ...]
    terminated: bool
    truncated: bool
    executed_actions: tuple[int | None, ...]
    active_before: tuple[bool, ...]
    active_after: tuple[bool, ...]
    escaped_this_tick: tuple[int, ...]
    resolution: dict[str, Any]

    def transition_for(self, member: int) -> Transition:
        if not 0 <= member < TEAM_SIZE:
            raise EscapeChamberError("member index is outside the team")
        return Transition(
            self.observations[member].copy(),
            self.rewards[member],
            self.terminated,
            self.truncated,
            {
                "contract_version": CONTRACT_VERSION,
                "episode_boundary": bool(self.terminated or self.truncated),
            },
        )


class EscapeChamberRoomA:
    """Deterministic simultaneous-action chamber with a maximum of three escapes.

    Authoritative joint-tick order:

    1. Decode each active body's anonymous action with the episode map.
    2. Hold gate/exit crossing proposals at their source and resolve all other
       movement simultaneously. Duplicate destinations and two-body swaps make
       every involved body wait. Moves into a body that is not leaving are then
       cancelled to a fixed point.
    3. Compute plate occupancy from those resolved ordinary movements. The gate
       is open for this tick only if an active, non-escaped body is now on it.
    4. Resolve the single-file gate crossing, then the exit-boundary crossing.
       A holder that left in step 2 therefore closes the gate before step 4.
    5. Apply rewards, render consequence pixels, and end at three escapes or
       the precommitted horizon.

    Rendering interpolation is deliberately outside this class and cannot
    affect these transitions.
    """

    width = 13
    height = 9
    plate = (2, 4)
    gate_inside = (7, 4)
    gate_outside = (8, 4)
    exit_inside = (11, 4)
    spawn_positions = ((3, 2), (4, 2), (3, 6), (4, 6))
    policy_shape = (96, 160, 3)
    spectator_shape = (720, 1280, 3)

    def __init__(
        self,
        *,
        seed: int,
        horizon: int = 240,
        reward_table: RewardTable | None = None,
    ):
        if horizon < 1:
            raise EscapeChamberError("horizon must be positive")
        self.seed = int(seed)
        self.horizon = int(horizon)
        self.reward_table = reward_table or RewardTable()
        self.reward_table.validate()
        self._floor = {
            *((x, y) for x in range(1, 8) for y in range(1, 8)),
            *((x, y) for x in range(8, 12) for y in range(3, 6)),
        }
        self._reset_done = False

    def public_spec(self) -> PublicWorldSpec:
        return PublicWorldSpec(
            world_id=f"escape-room-a-{self.seed}",
            family="cooperative-escape-chamber",
            adapter=ROOM_A_ADAPTER,
            agents=TEAM_SIZE,
            observation_kind="rgb-member-perspective-full-room",
            observation_shape=self.policy_shape,
            action_kind="discrete-anonymous-shared-map",
            action_count=ACTION_COUNT,
            horizon=self.horizon,
            reward_range=(-self.reward_table.active_tick_cost, 6.0),
            protocol_version=CONTRACT_VERSION,
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, ...]:
        self.episode_seed = self.seed if seed is None else int(seed)
        self._rng = np.random.default_rng(self.episode_seed)
        self._action_map = tuple(
            SEMANTIC_ACTIONS[index] for index in self._rng.permutation(ACTION_COUNT)
        )
        self.positions: list[tuple[int, int] | None] = list(self.spawn_positions)
        self.escaped = [False] * TEAM_SIZE
        self.escape_order: list[int] = []
        self.steps = 0
        self._done = False
        self.gate_open = False
        self.last_actions: list[int | None] = [None] * TEAM_SIZE
        self.last_rewards = [0.0] * TEAM_SIZE
        self.last_resolution: dict[str, Any] = {
            "ordinary_cancelled": [],
            "gate_crossers": [],
            "gate_open": False,
            "escaped": [],
        }
        self._reset_done = True
        return self.observations()

    def _ensure_live(self) -> None:
        if not self._reset_done:
            raise EscapeChamberError("reset must precede step")
        if self._done:
            raise EscapeChamberError("episode is over; reset before stepping again")

    @staticmethod
    def _validate_slot(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise EscapeChamberError("active member actions must be integer slots")
        slot = int(value)
        if not 0 <= slot < ACTION_COUNT:
            raise EscapeChamberError(f"action must be in [0, {ACTION_COUNT})")
        return slot

    def _decode_actions(
        self, actions: Sequence[int | None]
    ) -> tuple[list[int | None], list[str | None]]:
        if len(actions) != TEAM_SIZE:
            raise EscapeChamberError("a joint action must contain four member actions")
        slots: list[int | None] = []
        semantic: list[str | None] = []
        for member, value in enumerate(actions):
            if self.escaped[member]:
                if value is not None:
                    raise EscapeChamberError("escaped members must emit no action")
                slots.append(None)
                semantic.append(None)
            else:
                slot = self._validate_slot(value)
                slots.append(slot)
                semantic.append(self._action_map[slot])
        return slots, semantic

    def _classify_intent(
        self, position: tuple[int, int], semantic: str
    ) -> tuple[str, tuple[int, int] | None]:
        if semantic == WAIT:
            return "ordinary", position
        dx, dy = DELTAS[semantic]
        target = (position[0] + dx, position[1] + dy)
        if position == self.gate_inside and semantic == EAST:
            return "gate", self.gate_outside
        if position == self.exit_inside and semantic == EAST:
            return "exit", None
        if position[0] <= 7 and target[0] >= 8:
            return "ordinary", position
        if position[0] >= self.gate_outside[0] and target[0] < self.gate_outside[0]:
            return "ordinary", position
        if target not in self._floor:
            return "ordinary", position
        return "ordinary", target

    @staticmethod
    def _resolve_ordinary(
        positions: Sequence[tuple[int, int] | None],
        destinations: dict[int, tuple[int, int]],
    ) -> tuple[dict[int, tuple[int, int]], set[int]]:
        moving = {
            member for member, destination in destinations.items()
            if destination != positions[member]
        }
        cancelled: set[int] = set()
        by_destination: dict[tuple[int, int], list[int]] = {}
        for member in moving:
            by_destination.setdefault(destinations[member], []).append(member)
        for members in by_destination.values():
            if len(members) > 1:
                cancelled.update(members)
        moving.difference_update(cancelled)

        for first in tuple(moving):
            for second in tuple(moving):
                if first >= second:
                    continue
                if (destinations[first] == positions[second]
                        and destinations[second] == positions[first]):
                    cancelled.update((first, second))
        moving.difference_update(cancelled)

        changed = True
        while changed:
            changed = False
            stationary_cells = {
                positions[member]
                for member in destinations
                if member not in moving and positions[member] is not None
            }
            blocked = {
                member for member in moving
                if destinations[member] in stationary_cells
            }
            if blocked:
                moving.difference_update(blocked)
                cancelled.update(blocked)
                changed = True

        resolved = {
            member: destinations[member] if member in moving else positions[member]
            for member in destinations
        }
        return resolved, cancelled

    def step(self, actions: Sequence[int | None]) -> JointStep:
        self._ensure_live()
        slots, semantic = self._decode_actions(actions)
        active_before = tuple(not value for value in self.escaped)
        positions_before = tuple(self.positions)
        ordinary: dict[int, tuple[int, int]] = {}
        gate_crossers: list[int] = []
        exit_crossers: list[int] = []
        for member in range(TEAM_SIZE):
            if self.escaped[member]:
                continue
            position = self.positions[member]
            assert position is not None and semantic[member] is not None
            kind, destination = self._classify_intent(position, semantic[member])
            if kind == "gate":
                gate_crossers.append(member)
                ordinary[member] = position
            elif kind == "exit":
                exit_crossers.append(member)
                ordinary[member] = position
            else:
                assert destination is not None
                ordinary[member] = destination

        resolved, cancelled = self._resolve_ordinary(self.positions, ordinary)
        for member, position in resolved.items():
            self.positions[member] = position

        self.gate_open = any(
            not self.escaped[member] and self.positions[member] == self.plate
            for member in range(TEAM_SIZE)
        )
        gate_moved = []
        occupied = {position for position in self.positions if position is not None}
        for member in gate_crossers:
            if self.gate_open and self.gate_outside not in occupied:
                occupied.discard(self.positions[member])
                self.positions[member] = self.gate_outside
                occupied.add(self.gate_outside)
                gate_moved.append(member)

        escaped_this_tick = []
        for member in exit_crossers:
            if self.positions[member] == self.exit_inside:
                self.positions[member] = None
                self.escaped[member] = True
                self.escape_order.append(member)
                escaped_this_tick.append(member)

        rewards = np.zeros(TEAM_SIZE, dtype=np.float64)
        for member, active in enumerate(active_before):
            if active:
                rewards[member] -= self.reward_table.active_tick_cost
        count = len(escaped_this_tick)
        completed = len(self.escape_order) == TEAM_SIZE - 1
        if self.reward_table.treatment == "team":
            rewards += count * self.reward_table.escape_bonus
            if completed:
                rewards += self.reward_table.team_completion_bonus
        elif self.reward_table.treatment == "individual":
            for member in escaped_this_tick:
                rewards[member] += self.reward_table.escape_bonus
        else:
            team = self.reward_table.mixed_team_weight
            individual = self.reward_table.mixed_individual_weight
            rewards += team * count * self.reward_table.escape_bonus
            for member in escaped_this_tick:
                rewards[member] += individual * self.reward_table.escape_bonus
            if completed:
                rewards += team * self.reward_table.team_completion_bonus

        self.steps += 1
        terminated = completed
        truncated = self.steps >= self.horizon and not terminated
        self._done = bool(terminated or truncated)
        self.last_actions = slots
        self.last_rewards = rewards.tolist()
        self.last_resolution = {
            "tick": self.steps,
            "ordinary_cancelled": sorted(cancelled),
            "gate_proposals": list(gate_crossers),
            "gate_crossers": gate_moved,
            "gate_open": self.gate_open,
            "exit_proposals": list(exit_crossers),
            "escaped": escaped_this_tick,
            "positions_before": [None if p is None else list(p) for p in positions_before],
            "positions_after": [None if p is None else list(p) for p in self.positions],
        }
        observations = self.observations()
        return JointStep(
            observations=observations,
            rewards=tuple(float(value) for value in rewards),
            terminated=terminated,
            truncated=truncated,
            executed_actions=tuple(slots),
            active_before=active_before,
            active_after=tuple(not value for value in self.escaped),
            escaped_this_tick=tuple(escaped_this_tick),
            resolution=json.loads(json.dumps(self.last_resolution)),
        )

    def observations(self) -> tuple[np.ndarray, ...]:
        if not self._reset_done:
            raise EscapeChamberError("reset must precede observation")
        return tuple(self._render_policy(member) for member in range(TEAM_SIZE))

    @staticmethod
    def _polygon(center: tuple[float, float], radius: float, sides: int, offset: float = 0):
        return [
            (
                center[0] + np.cos(offset + 2 * np.pi * index / sides) * radius,
                center[1] + np.sin(offset + 2 * np.pi * index / sides) * radius,
            )
            for index in range(sides)
        ]

    @classmethod
    def _draw_body(
        cls,
        draw: ImageDraw.ImageDraw,
        member: int,
        center: tuple[float, float],
        radius: float,
        *,
        self_marker: bool = False,
        label: bool = False,
    ) -> None:
        x, y = center
        color = MEMBER_COLORS[member]
        shadow = (x - radius + 2, y - radius + 3, x + radius + 2, y + radius + 3)
        draw.ellipse(shadow, fill=(20, 24, 29, 150))
        shape = MEMBER_SHAPES[member]
        box = (x - radius, y - radius, x + radius, y + radius)
        if shape == "circle":
            draw.ellipse(box, fill=color, outline=(242, 239, 222), width=max(1, int(radius / 4)))
        elif shape == "rounded-square":
            draw.rounded_rectangle(box, radius=radius / 3, fill=color,
                                   outline=(242, 239, 222), width=max(1, int(radius / 4)))
        elif shape == "triangle":
            draw.polygon(cls._polygon((x, y), radius, 3, -np.pi / 2), fill=color,
                         outline=(242, 239, 222))
        else:
            draw.polygon(cls._polygon((x, y), radius, 6), fill=color,
                         outline=(242, 239, 222))
        draw.line((x, y, x + radius * 0.62, y), fill=(28, 32, 38),
                  width=max(1, int(radius / 4)))
        if self_marker:
            draw.ellipse((x - radius - 3, y - radius - 3, x + radius + 3, y + radius + 3),
                         outline=(255, 255, 255), width=2)
        if label:
            draw.text((x - radius, y + radius + 4), f"F{member + 1}", fill=(239, 235, 217))

    def _draw_room(self, image: Image.Image, *, member_view: int | None) -> None:
        draw = ImageDraw.Draw(image, "RGBA")
        width, height = image.size
        sx, sy = width / self.width, height / self.height
        draw.rectangle((0, 0, width, height), fill=(12, 17, 23, 255))
        for x, y in self._floor:
            pad = min(sx, sy) * 0.045
            rect = (x * sx + pad, y * sy + pad, (x + 1) * sx - pad, (y + 1) * sy - pad)
            outside = x >= 8
            fill = (89, 83, 70, 255) if not outside else (54, 72, 76, 255)
            draw.rounded_rectangle(rect, radius=max(1, int(pad * 1.5)), fill=fill)

        px, py = self.plate
        plate_box = (px * sx + sx * .16, py * sy + sy * .18,
                     (px + 1) * sx - sx * .16, (py + 1) * sy - sy * .18)
        occupied = any(position == self.plate for position in self.positions)
        plate_color = (181, 126, 49, 255) if occupied else (109, 83, 52, 255)
        draw.rounded_rectangle(plate_box, radius=min(sx, sy) * .12, fill=plate_color,
                               outline=(225, 184, 83, 255), width=max(1, int(sx / 40)))
        inset = sy * (.09 if occupied else .04)
        draw.line((plate_box[0] + 3, plate_box[3] - inset,
                   plate_box[2] - 3, plate_box[3] - inset), fill=(55, 45, 35, 220), width=2)

        wall_x = 8 * sx
        gate_y0, gate_y1 = 4 * sy, 5 * sy
        wall_color = (48, 53, 60, 255)
        brass = (177, 132, 52, 255)
        draw.rectangle((wall_x - sx * .09, sy, wall_x + sx * .09, gate_y0), fill=wall_color)
        draw.rectangle((wall_x - sx * .09, gate_y1, wall_x + sx * .09, 8 * sy), fill=wall_color)
        if not self.gate_open:
            draw.rounded_rectangle((wall_x - sx * .16, gate_y0 + sy * .05,
                                    wall_x + sx * .16, gate_y1 - sy * .05),
                                   radius=max(2, int(sx * .06)), fill=brass,
                                   outline=(236, 201, 114, 255), width=max(1, int(sx / 36)))
        else:
            draw.rectangle((wall_x - sx * .17, gate_y0 - sy * .25,
                            wall_x + sx * .17, gate_y0 + sy * .06), fill=brass)

        exit_x = 12 * sx
        draw.rectangle((exit_x - sx * .15, 3.15 * sy, width, 5.85 * sy),
                       fill=(196, 218, 173, 38))
        draw.line((exit_x - sx * .08, 3.2 * sy, exit_x - sx * .08, 5.8 * sy),
                  fill=(198, 222, 178, 255), width=max(2, int(sx / 24)))

        for member, position in enumerate(self.positions):
            if position is None:
                order = self.escape_order.index(member)
                center = (12.35 * sx, (3.55 + order * .85) * sy)
            else:
                center = ((position[0] + .5) * sx, (position[1] + .5) * sy)
            self._draw_body(
                draw, member, center, min(sx, sy) * .29,
                self_marker=member_view == member,
                label=member_view is None,
            )

    def _render_policy(self, member: int) -> np.ndarray:
        image = Image.new("RGB", (self.policy_shape[1], self.policy_shape[0]))
        self._draw_room(image, member_view=member)
        return np.asarray(image, dtype=np.uint8)

    def spectator_frame(self) -> np.ndarray:
        if not self._reset_done:
            raise EscapeChamberError("reset must precede rendering")
        image = Image.new("RGB", (self.spectator_shape[1], self.spectator_shape[0]), (9, 13, 19))
        room = Image.new("RGB", (920, 636))
        self._draw_room(room, member_view=None)
        image.paste(room, (30, 54))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((31, 19), "ROOM A  /  ONE HOLDS, THREE LEAVE", fill=(238, 231, 210), font=font)
        draw.text((970, 54), f"EPISODE SEED  {self.episode_seed}", fill=(147, 158, 170), font=font)
        draw.text((970, 82), f"INCENTIVE  {self.reward_table.treatment.upper()}",
                  fill=(221, 186, 98), font=font)
        draw.text((970, 110), f"JOINT TICK  {self.steps:04d} / {self.horizon}",
                  fill=(147, 158, 170), font=font)
        draw.text((970, 138), f"ESCAPED  {len(self.escape_order)} / 3 LEGAL MAX",
                  fill=(197, 220, 179), font=font)
        draw.text((970, 180), "SPECTATOR ONLY", fill=(223, 113, 94), font=font)
        draw.text((150, 332), "PRESSURE PLATE", fill=(232, 203, 128), font=font)
        draw.text((575, 332), "GATE", fill=(232, 203, 128), font=font)
        draw.text((850, 332), "EXIT", fill=(205, 229, 187), font=font)
        for member in range(TEAM_SIZE):
            y = 224 + member * 94
            status = "ESCAPED" if self.escaped[member] else "ACTIVE"
            action = "-" if self.last_actions[member] is None else f"slot {self.last_actions[member]}"
            draw.rounded_rectangle((968, y, 1250, y + 76), radius=10,
                                   fill=(19, 27, 36), outline=MEMBER_COLORS[member], width=2)
            draw.text((986, y + 13), f"{MEMBER_NAMES[member]}  /  {MEMBER_SHAPES[member]}",
                      fill=MEMBER_COLORS[member], font=font)
            draw.text((986, y + 36), f"{status}    action {action}", fill=(218, 222, 220), font=font)
            draw.text((986, y + 55), f"local reward {self.last_rewards[member]:+.3f}",
                      fill=(157, 168, 176), font=font)
        return np.asarray(image, dtype=np.uint8)

    def audit_state(self) -> dict[str, Any]:
        """Private verifier state. This object must never cross to a learner."""
        value = {
            "format": "gum-cooperative-escape-room-a-audit-v1",
            "contract_version": CONTRACT_VERSION,
            "episode_seed": self.episode_seed,
            "action_map": list(self._action_map),
            "positions": [None if value is None else list(value) for value in self.positions],
            "escaped": list(self.escaped),
            "escape_order": list(self.escape_order),
            "plate": list(self.plate),
            "gate_inside": list(self.gate_inside),
            "gate_outside": list(self.gate_outside),
            "gate_open": self.gate_open,
            "steps": self.steps,
            "done": self._done,
            "reward_table": self.reward_table.to_json(),
            "last_resolution": self.last_resolution,
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        value["state_sha256"] = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
        return value

    def action_slot_for_test(self, semantic: str) -> int:
        """Isolated positive-control hook; experimental learners must not call it."""
        if semantic not in SEMANTIC_ACTIONS:
            raise EscapeChamberError("unknown semantic action")
        return self._action_map.index(semantic)


def single_body_reachable_cells() -> set[tuple[int, int]]:
    """Exhaustively explore Room A positions for a lone body under closed-gate rules."""
    start = EscapeChamberRoomA.spawn_positions[0]
    floor = {
        *((x, y) for x in range(1, 8) for y in range(1, 8)),
        *((x, y) for x in range(8, 12) for y in range(3, 6)),
    }
    reached = {start}
    frontier = [start]
    while frontier:
        position = frontier.pop()
        for semantic in SEMANTIC_ACTIONS:
            dx, dy = DELTAS[semantic]
            target = (position[0] + dx, position[1] + dy)
            if position == EscapeChamberRoomA.gate_inside and semantic == EAST:
                target = position
            elif position[0] <= 7 and target[0] >= 8:
                target = position
            elif target not in floor:
                target = position
            if target not in reached:
                reached.add(target)
                frontier.append(target)
    return reached


def room_a_invariant_report() -> dict[str, Any]:
    """Machine-checkable geometry facts supporting the max-three proof."""
    reachable = single_body_reachable_cells()
    return {
        "format": "gum-cooperative-escape-room-a-invariants-v1",
        "single_body_escape_possible": any(x >= 8 for x, _ in reachable),
        "single_body_reachable_cells": len(reachable),
        "gate_requires_post_ordinary_plate_occupant": True,
        "plate_gate_manhattan_distance": (
            abs(EscapeChamberRoomA.plate[0] - EscapeChamberRoomA.gate_inside[0])
            + abs(EscapeChamberRoomA.plate[1] - EscapeChamberRoomA.gate_inside[1])
        ),
        "single_file_gate": True,
        "episode_terminates_at_escapes": TEAM_SIZE - 1,
        "max_escape_argument": (
            "The fourth body cannot cross the gate: after three bodies have crossed, "
            "it is the only non-escaped body, and it cannot simultaneously occupy the "
            "interior plate after ordinary movement and propose the distant gate crossing."
        ),
    }
