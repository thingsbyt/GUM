"""Two-rocket Asteroids with pixel-only, independently grounded control.

The environment owns physics and a different hidden action permutation for each
rocket.  The learner receives only two RGB views, shared reward and termination.
Its blackboard contains observations derived from pixels, never simulator state.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


SEMANTICS = ("coast", "negative-turn", "positive-turn", "thrust", "fire")
N_ACTIONS = 5
ROCK_CAPACITY = 24
BULLET_CAPACITY = 32
SELF = (74, 224, 211)
TEAMMATE = (245, 158, 72)
ROCK = (63, 73, 92)
ROCK_EDGE = (182, 191, 206)
SELF_BULLET = (255, 224, 115)
TEAM_BULLET = (255, 122, 105)
EXHAUST = (255, 96, 55)
SELF_NOSE = (160, 255, 242)
TEAM_NOSE = (255, 213, 145)
SELF_CENTER = (35, 128, 255)
TEAM_CENTER = (255, 92, 188)


def _wrap_angle(value):
    return (float(value) + math.pi) % (2 * math.pi) - math.pi


@dataclass
class DuoConfig:
    difficulty: int = 2
    max_steps: int = 360
    frames_per_action: int = 4
    width: float = 640.0
    height: float = 360.0
    image_width: int = 320
    image_height: int = 180
    sensor_range: float = 205.0


class DuoAsteroids:
    """A shared asteroid field with two bodies and two unknown controllers."""

    agents = 2

    def __init__(self, config=None, seed=0, *, active_agents=2):
        self.config = config or DuoConfig()
        if self.config.difficulty not in (1, 2, 3):
            raise ValueError("difficulty must be 1, 2, or 3")
        if active_agents not in (1, 2):
            raise ValueError("active_agents must be one or two")
        self.active_agents = active_agents
        self.size = np.asarray((self.config.width, self.config.height), dtype=np.float64)
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        # Public action numbers have unrelated meaning for the two bodies.
        self.semantic_by_action = [tuple(SEMANTICS[i] for i in self.rng.permutation(5)) for _ in range(2)]
        self.ships = np.asarray(((self.size[0] * .34, self.size[1] * .50),
                                 (self.size[0] * .66, self.size[1] * .50)), dtype=np.float64)
        self.spawn_points = self.ships.copy()
        self.velocities = np.zeros((2, 2), dtype=np.float64)
        self.angles = self.rng.uniform(-math.pi, math.pi, size=2)
        self.cooldowns = np.zeros(2, dtype=np.float64)
        self.shields = np.full(2, 2.0, dtype=np.float64)
        self.alive = np.asarray((True, self.active_agents == 2), dtype=bool)
        self.lives = [3, 3 if self.active_agents == 2 else 0]
        self.last_semantics = ["coast", "coast"]
        self.rocks = np.zeros((ROCK_CAPACITY, 5), dtype=np.float64)
        # x, y, vx, vy, ttl, owner+1
        self.bullets = np.zeros((BULLET_CAPACITY, 6), dtype=np.float64)
        self.bullet_birth = np.full(BULLET_CAPACITY, -1, dtype=np.int64)
        self.steps = self.frames = self.wave = self.hits = self.friendly_fire = 0
        self.friendly_events = []; self.death_events = []
        self.hits_by_agent = [0, 0]
        self.deaths = [0, 0]
        self.terminated = self.truncated = False
        self.total_reward = 0.0
        self._spawn_wave()
        return self.observe()

    def _delta(self, a, b):
        return (np.asarray(a) - np.asarray(b) + self.size / 2) % self.size - self.size / 2

    def _spawn_wave(self):
        self.wave += 1
        count = (3, 5, 7)[self.config.difficulty - 1]
        low, high = ((22, 42), (32, 58), (42, 72))[self.config.difficulty - 1]
        for index in range(count):
            position = self.rng.uniform((0, 0), self.size)
            for _ in range(200):
                clear = all(not self.alive[a] or np.linalg.norm(self._delta(position, self.ships[a])) > 145
                            for a in range(2))
                if clear:
                    break
                position = self.rng.uniform((0, 0), self.size)
            heading = float(self.rng.uniform(-math.pi, math.pi)); speed = float(self.rng.uniform(low, high))
            self.rocks[index] = (*position, math.cos(heading) * speed, math.sin(heading) * speed, 28.0)

    def _fire(self, agent):
        if self.cooldowns[agent] > 1e-9 or not self.alive[agent]:
            return
        slots = np.flatnonzero(self.bullets[:, 4] <= 0)
        if not len(slots):
            return
        direction = np.asarray((math.cos(self.angles[agent]), math.sin(self.angles[agent])))
        position = (self.ships[agent] + direction * 14) % self.size
        velocity = self.velocities[agent] + direction * 350
        self.bullets[slots[0]] = (*position, *velocity, 1.45, agent + 1)
        self.bullet_birth[slots[0]] = self.steps
        self.cooldowns[agent] = .17

    def _destroy_rock(self, index, owner):
        rock = self.rocks[index].copy(); self.rocks[index] = 0
        if rock[4] > 20:
            heading = math.atan2(rock[3], rock[2]); speed = max(46.0, np.linalg.norm(rock[2:4]) * 1.3)
            for rotation in (-.68, .68):
                slots = np.flatnonzero(self.rocks[:, 4] <= 0)
                if not len(slots):
                    break
                angle = heading + rotation
                self.rocks[slots[0]] = (*rock[:2], math.cos(angle) * speed,
                                         math.sin(angle) * speed, 14.0)
        self.hits += 1; self.hits_by_agent[owner] += 1

    def step(self, actions):
        if self.terminated or self.truncated:
            raise RuntimeError("episode ended; call reset")
        if len(actions) != 2 or any(not isinstance(a, (int, np.integer)) or not 0 <= int(a) < 5
                                    for a in actions):
            raise ValueError("two integer actions from zero through four required")
        semantics = [self.semantic_by_action[i][int(actions[i])] for i in range(2)]
        self.last_semantics = semantics
        old_hits, old_fire, old_deaths = self.hits, self.friendly_fire, sum(self.deaths)
        dt = 1 / 60
        for _ in range(self.config.frames_per_action):
            self.frames += 1
            for i in range(2):
                if not self.alive[i]:
                    continue
                self.cooldowns[i] = max(0.0, self.cooldowns[i] - dt)
                self.shields[i] = max(0.0, self.shields[i] - dt)
                if semantics[i] == "negative-turn": self.angles[i] = _wrap_angle(self.angles[i] - 3.75 * dt)
                if semantics[i] == "positive-turn": self.angles[i] = _wrap_angle(self.angles[i] + 3.75 * dt)
                if semantics[i] == "thrust":
                    self.velocities[i] += np.asarray((math.cos(self.angles[i]), math.sin(self.angles[i]))) * 175 * dt
                if semantics[i] == "fire": self._fire(i)
                # Mild inertial damping keeps a brief calibration thrust from
                # becoming a permanent hidden trajectory several minutes later.
                self.velocities[i] *= .985
                speed = np.linalg.norm(self.velocities[i])
                if speed > 220: self.velocities[i] *= 220 / speed
                self.ships[i] = (self.ships[i] + self.velocities[i] * dt) % self.size
            active_rocks = self.rocks[:, 4] > 0
            self.rocks[active_rocks, :2] = (self.rocks[active_rocks, :2] + self.rocks[active_rocks, 2:4] * dt) % self.size
            active_bullets = self.bullets[:, 4] > 0
            bullet_indices = np.flatnonzero(active_bullets)
            if len(bullet_indices):
                next_positions = self.bullets[bullet_indices, :2] + self.bullets[bullet_indices, 2:4] * dt
                crossed = np.any((next_positions < 0) | (next_positions >= self.size), axis=1)
                self.bullets[bullet_indices[~crossed], :2] = next_positions[~crossed]
                self.bullets[bullet_indices[crossed]] = 0
            active_bullets = self.bullets[:, 4] > 0
            self.bullets[active_bullets, 4] -= dt
            self.bullets[self.bullets[:, 4] <= 0] = 0

            # Bullets and rocks.
            bi = np.flatnonzero(self.bullets[:, 4] > 0); ri = np.flatnonzero(self.rocks[:, 4] > 0)
            if len(bi) and len(ri):
                distances = self._delta(self.bullets[bi, None, :2], self.rocks[None, ri, :2])
                pairs = np.argwhere((distances * distances).sum(-1) <= (self.rocks[ri, 4][None, :] + 3) ** 2)
                used_b, used_r = set(), set()
                for b_local, r_local in pairs:
                    b, r = int(bi[b_local]), int(ri[r_local])
                    if b in used_b or r in used_r: continue
                    owner = int(self.bullets[b, 5] - 1); self.bullets[b] = 0
                    self._destroy_rock(r, owner); used_b.add(b); used_r.add(r)

            # Friendly fire is real, visible and costly, but does not instantly end the episode.
            bi = np.flatnonzero(self.bullets[:, 4] > 0)
            for b in bi:
                owner = int(self.bullets[b, 5] - 1); other = 1 - owner
                if self.alive[other] and self.shields[other] <= 0:
                    d = self._delta(self.bullets[b, :2], self.ships[other])
                    if float(d @ d) <= 9 ** 2:
                        self.friendly_events.append({"step": self.steps, "frame": self.frames, "owner": owner,
                            "fired_step": int(self.bullet_birth[b]),
                            "bullet": self.bullets[b, :2].round(3).tolist(), "teammate": self.ships[other].round(3).tolist(),
                            "owner_angle": round(float(self.angles[owner]), 4), "owner_action": semantics[owner]})
                        self.bullets[b] = 0; self.friendly_fire += 1
                        self.shields[other] = .35

            # Rock impacts.
            ri = np.flatnonzero(self.rocks[:, 4] > 0)
            for agent in range(2):
                if not self.alive[agent] or self.shields[agent] > 0: continue
                d = self._delta(self.rocks[ri, :2], self.ships[agent]) if len(ri) else np.empty((0, 2))
                if len(d) and np.any((d * d).sum(1) <= (self.rocks[ri, 4] + 8) ** 2):
                    self.death_events.append({"step": self.steps, "frame": self.frames, "agent": agent,
                                              "position": self.ships[agent].round(3).tolist(),
                                              "remaining_lives": self.lives[agent] - 1})
                    self.deaths[agent] += 1; self.lives[agent] -= 1
                    if self.lives[agent] > 0:
                        self.ships[agent] = self.spawn_points[agent]
                        self.velocities[agent] = 0; self.shields[agent] = 1.0
                    else:
                        self.alive[agent] = False
            if not np.any(self.alive):
                self.terminated = True; break
            if not np.any(self.rocks[:, 4] > 0):
                self._spawn_wave(); self.shields[self.alive] = np.maximum(self.shields[self.alive], .45)

        self.steps += 1
        self.truncated = self.steps >= self.config.max_steps and not self.terminated
        reward = ((self.hits - old_hits) - 10 * (self.friendly_fire - old_fire)
                  - 20 * (sum(self.deaths) - old_deaths) + .002 * int(np.sum(self.alive)))
        self.total_reward += reward
        info = {"hits": self.hits, "hits_by_agent": list(self.hits_by_agent), "friendly_fire": self.friendly_fire,
                "alive": self.alive.tolist(), "deaths": list(self.deaths), "wave": self.wave,
                "lives": list(self.lives),
                "seconds": self.frames / 60, "return": self.total_reward}
        return self.observe(), float(reward), self.terminated, self.truncated, info

    def _point(self, point):
        return point[0] * self.config.image_width / self.size[0], point[1] * self.config.image_height / self.size[1]

    def render_for(self, viewer):
        image = Image.new("RGB", (self.config.image_width, self.config.image_height), (8, 12, 20))
        draw = ImageDraw.Draw(image)
        scale = self.config.image_width / self.size[0]
        # Each pilot has a local scanner, not an omniscient global camera.
        center = self.ships[viewer]
        visible_rocks = [row for row in self.rocks[self.rocks[:, 4] > 0]
                         if np.linalg.norm(self._delta(row[:2], center)) <= self.config.sensor_range + row[4]]
        for x, y, _, _, radius in visible_rocks:
            points = []
            for k in range(9):
                angle = 2 * math.pi * k / 9; r = radius * (.88 if k % 2 else 1)
                points.append(self._point((x + r * math.cos(angle), y + r * math.sin(angle))))
            draw.polygon(points, fill=ROCK, outline=ROCK_EDGE)
        visible_bullets = [row for row in self.bullets[self.bullets[:, 4] > 0]
                           if np.linalg.norm(self._delta(row[:2], center)) <= self.config.sensor_range]
        for x, y, _, _, _, owner1 in visible_bullets:
            px, py = self._point((x, y)); color = SELF_BULLET if int(owner1 - 1) == viewer else TEAM_BULLET
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
        # Teammate first so the self marker always remains legible at overlap.
        for agent in (1 - viewer, viewer):
            if not self.alive[agent]: continue
            if agent != viewer and np.linalg.norm(self._delta(self.ships[agent], center)) > self.config.sensor_range:
                continue
            x, y = self.ships[agent]; angle = self.angles[agent]
            points = [self._point((x + 14 * math.cos(angle), y + 14 * math.sin(angle))),
                      self._point((x + 11 * math.cos(angle + 2.45), y + 11 * math.sin(angle + 2.45))),
                      self._point((x + 11 * math.cos(angle - 2.45), y + 11 * math.sin(angle - 2.45)))]
            fill = SELF if agent == viewer else TEAMMATE
            draw.polygon(points, fill=fill, outline=(228, 255, 250) if agent == viewer else (255, 220, 174))
            px, py = self._point((x, y)); nose = self._point((x + 13 * math.cos(angle), y + 13 * math.sin(angle)))
            nose_color = SELF_NOSE if agent == viewer else TEAM_NOSE
            draw.line((px, py, nose[0], nose[1]), fill=nose_color, width=1)
            center_color = SELF_CENTER if agent == viewer else TEAM_CENTER
            draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=center_color)
            if self.last_semantics[agent] == "thrust":
                tail = self._point((x - 11 * math.cos(angle), y - 11 * math.sin(angle)))
                flame = self._point((x - 23 * math.cos(angle), y - 23 * math.sin(angle)))
                draw.line((*tail, *flame), fill=EXHAUST, width=3)
            if self.shields[agent] > 0:
                px, py = self._point((x, y)); radius = 18 * scale
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=(68, 126, 148))
        return np.asarray(image, dtype=np.uint8).copy()

    def observe(self):
        return [self.render_for(0), self.render_for(1)]

    def audit_state(self):
        """State for evaluation only; never passed to the learner."""
        return {"seed": self.seed, "control_maps_audit_only": [list(row) for row in self.semantic_by_action],
                "ships": self.ships.tolist(), "velocities": self.velocities.tolist(), "angles": self.angles.tolist(),
                "alive": self.alive.tolist(), "hits": self.hits, "hits_by_agent": list(self.hits_by_agent),
                "friendly_fire": self.friendly_fire, "friendly_events": list(self.friendly_events),
                "death_events": list(self.death_events), "lives": list(self.lives), "wave": self.wave, "steps": self.steps}


def _components(mask, minimum=2):
    labels, count = ndimage.label(mask)
    rows = []
    for label in range(1, count + 1):
        yy, xx = np.nonzero(labels == label)
        if len(xx) >= minimum:
            rows.append({"center": np.asarray((xx.mean(), yy.mean())), "area": len(xx),
                         "pixels": np.column_stack((xx, yy))})
    return rows


def perceive(frame):
    """Extract anonymous geometry from RGB pixels; no environment object is consulted."""
    rgb = np.asarray(frame, dtype=np.uint8)
    exact = lambda color, tolerance=4: np.max(np.abs(rgb.astype(np.int16) - np.asarray(color)), axis=2) <= tolerance
    own = _components(exact(SELF), 4); teammate = _components(exact(TEAMMATE), 4)
    own_nose = _components(exact(SELF_NOSE), 1); team_nose = _components(exact(TEAM_NOSE), 1)
    own_center = _components(exact(SELF_CENTER), 1); team_center = _components(exact(TEAM_CENTER), 1)
    rocks = _components(exact(ROCK), 8); own_bullets = _components(exact(SELF_BULLET), 2)
    exhaust_parts = _components(exact(EXHAUST), 1)

    def pose(parts, noses, centers):
        if not parts: return None
        part = max(parts, key=lambda row: row["area"])
        center = max(centers, key=lambda row: row["area"])["center"] if centers else part["center"]
        if noses:
            pixels = np.concatenate([row["pixels"] for row in noses], axis=0)
            nose = pixels[int(np.argmax(((pixels - center[None, :]) ** 2).sum(1)))]
        else:
            offsets = part["pixels"] - center[None, :]
            nose = part["pixels"][int(np.argmax((offsets * offsets).sum(1)))]
        return {"center": center, "angle": math.atan2(nose[1] - center[1], nose[0] - center[0])}
    own_pose = pose(own, own_nose, own_center)
    exhaust = bool(own_pose and any(np.linalg.norm(row["center"] - own_pose["center"]) < 16
                                    for row in exhaust_parts))
    return {"self": own_pose, "teammate": pose(teammate, team_nose, team_center),
            "rocks": [row["center"] for row in rocks], "own_bullets": len(own_bullets),
            "exhaust": exhaust, "shape": rgb.shape[:2]}


class ControlGrounder:
    """Learns one anonymous controller by interventions and visible effects."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.probes = [action for action in range(5) for _ in range(2)]
        self.index = 0; self.evidence = {a: {"angle": [], "fire": 0, "thrust": 0} for a in range(5)}
        self.mapping = {}; self.complete = False

    def action(self):
        return self.probes[min(self.index, len(self.probes) - 1)]

    def observe(self, before, action, after):
        earlier, later = perceive(before), perceive(after); row = self.evidence[int(action)]
        if earlier["self"] and later["self"]:
            row["angle"].append(_wrap_angle(later["self"]["angle"] - earlier["self"]["angle"]))
        row["fire"] += int(later["own_bullets"] > earlier["own_bullets"])
        row["thrust"] += int(later["exhaust"])
        self.index += 1
        if self.index >= len(self.probes): self._solve()

    def _solve(self):
        unused = set(range(5))
        thrust = max(unused, key=lambda a: self.evidence[a]["thrust"]); self.mapping["thrust"] = thrust; unused.remove(thrust)
        fire = max(unused, key=lambda a: self.evidence[a]["fire"]); self.mapping["fire"] = fire; unused.remove(fire)
        means = {a: float(np.median(self.evidence[a]["angle"])) if self.evidence[a]["angle"] else 0.0 for a in unused}
        negative = min(unused, key=means.get); self.mapping["negative-turn"] = negative; unused.remove(negative)
        positive = max(unused, key=means.get); self.mapping["positive-turn"] = positive; unused.remove(positive)
        self.mapping["coast"] = unused.pop(); self.complete = True


class SharedVisualBlackboard:
    """Communicates only pixel-grounded tracks and temporary target claims."""

    def __init__(self):
        self.previous = []; self.velocity = []; self.messages = []; self.updates = 0

    def update(self, frame):
        return self.update_many([frame])

    def update_many(self, frames):
        now = []
        for frame in frames:
            for point in perceive(frame)["rocks"]:
                if not any(np.linalg.norm(point - existing) < 6 for existing in now):
                    now.append(point)
        return self.update_points(now)

    def update_points(self, points):
        now = []
        for point in points:
            if not any(np.linalg.norm(point - existing) < 6 for existing in now):
                now.append(point)
        velocity = [np.zeros(2) for _ in now]
        remaining = set(range(len(self.previous)))
        for index, point in enumerate(now):
            if remaining:
                match = min(remaining, key=lambda j: np.linalg.norm(point - self.previous[j]))
                if np.linalg.norm(point - self.previous[match]) < 28:
                    velocity[index] = point - self.previous[match]; remaining.remove(match)
        self.previous = [point.copy() for point in now]; self.velocity = velocity; self.updates += 1
        return [{"id": i, "point": point, "velocity": velocity[i]} for i, point in enumerate(now)]


class TwoRocketTeam:
    """Two visual learners. Coordination can be disabled as a matched ablation."""

    def __init__(self, *, communicate=True):
        self.communicate = bool(communicate); self.grounders = [ControlGrounder(), ControlGrounder()]
        self.board = SharedVisualBlackboard(); self.local_boards = [SharedVisualBlackboard(), SharedVisualBlackboard()]
        self.rounds = 0; self.target_messages = 0; self.prevented_friendly_shots = 0
        self.previous_poses = [None, None]

    def begin(self, frames):
        self.grounders = [ControlGrounder(), ControlGrounder()]
        self.board = SharedVisualBlackboard(); self.local_boards = [SharedVisualBlackboard(), SharedVisualBlackboard()]
        self.rounds = 0; self.target_messages = 0; self.prevented_friendly_shots = 0
        self.previous_poses = [None, None]

    @staticmethod
    def _visual_delta(point, origin, shape):
        size = np.asarray((shape[1], shape[0]), dtype=np.float64)
        return (np.asarray(point) - np.asarray(origin) + size / 2) % size - size / 2

    def _assignment_cost(self, perception, track):
        if perception["self"] is None: return 1e6
        own = perception["self"]["center"]; predicted = track["point"] + track["velocity"] * 3
        delta = self._visual_delta(predicted, own, perception["shape"])
        desired = math.atan2(delta[1], delta[0])
        return abs(_wrap_angle(desired - perception["self"]["angle"])) + .0015 * np.linalg.norm(delta)

    def _joint_assignment(self, perceptions, tracks):
        if not tracks: return [None, None]
        if len(tracks) == 1: return [tracks[0], tracks[0]]
        best = None
        for first in tracks:
            for second in tracks:
                if first["id"] == second["id"]: continue
                cost = self._assignment_cost(perceptions[0], first) + self._assignment_cost(perceptions[1], second)
                if best is None or cost < best[0]: best = (cost, first, second)
        return [best[1], best[2]]

    def _policy(self, agent, view, tracks, claimed, shared_mate=None, perception=None, assigned=None):
        perception = perception or perceive(view); controls = self.grounders[agent].mapping
        if perception["self"] is None or not tracks: return controls["coast"]
        own = perception["self"]["center"]; heading = perception["self"]["angle"]
        visible = [track for track in tracks if any(np.linalg.norm(track["point"] - point) < 7
                                                     for point in perception["rocks"])]
        pool = visible or tracks
        candidates = pool
        # Travel-time prediction is learned from consecutive pixel displacements.
        target = assigned or min(candidates, key=lambda row:
                                 np.linalg.norm(self._visual_delta(row["point"], own, perception["shape"])))
        if self.communicate:
            claimed.add(target["id"]); self.target_messages += 1
            self.board.messages.append({"round": self.rounds, "agent": agent, "target": target["id"]})
        predicted = target["point"] + target["velocity"] * 3.0
        delta = self._visual_delta(predicted, own, perception["shape"])
        desired = math.atan2(delta[1], delta[0]); error = _wrap_angle(desired - heading)

        # A close visual threat temporarily overrides shooting.  This is a
        # generic collision reflex over grounded geometry, not simulator data.
        threat = min(pool, key=lambda row: np.linalg.norm(
            self._visual_delta(row["point"], own, perception["shape"])))
        threat_delta = self._visual_delta(threat["point"] + threat["velocity"] * 2,
                                          own, perception["shape"])
        if np.linalg.norm(threat_delta) < 38:
            escape = _wrap_angle(math.atan2(-threat_delta[1], -threat_delta[0]) - heading)
            if abs(escape) > .18:
                return controls["positive-turn" if escape > 0 else "negative-turn"]
            return controls["thrust"]

        # Do not fire through the teammate when it is nearer than the target.
        mate = perception["teammate"] or shared_mate
        if mate is not None:
            velocity = np.asarray(mate.get("velocity", np.zeros(2)))
            unsafe_lane = False; unsafe_direct = False
            height, width = perception["shape"]; forward = np.asarray((math.cos(heading), math.sin(heading)))
            max_range = .76 * math.hypot(width, height)
            for horizon in range(25):
                base = mate["center"] + velocity * horizon - own
                # Bullets expire at the arena edge, so only the visible image
                # of the teammate can lie on the physical firing segment.
                for sx in (0,):
                    for sy in (0,):
                        candidate = base + np.asarray((sx, sy)); distance = np.linalg.norm(candidate)
                        projection = float(candidate @ forward)
                        cross_track = abs(float(candidate[0] * forward[1] - candidate[1] * forward[0]))
                        unsafe_lane |= 0 < projection < max_range and cross_track < 16.0
                        # The cross-track test is the actual collision corridor;
                        # a broad angular cone would unnecessarily stop most
                        # useful shots in a small toroidal arena.
            if unsafe_lane or unsafe_direct:
                self.prevented_friendly_shots += 1
                return controls["positive-turn" if error >= 0 else "negative-turn"]
        if abs(error) > .12:
            return controls["positive-turn" if error > 0 else "negative-turn"]
        return controls["fire"]

    def act(self, frames):
        actions = []
        for agent in range(2):
            actions.append(self.grounders[agent].action() if not self.grounders[agent].complete else None)
        if all(action is not None for action in actions): return actions
        perceptions = [perceive(frame) for frame in frames]
        if self.communicate:
            tracks = self.board.update_points(perceptions[0]["rocks"] + perceptions[1]["rocks"]); claimed = set()
            poses = [row["self"] for row in perceptions]
            shared_poses = []
            for i, pose in enumerate(poses):
                if pose is None:
                    shared_poses.append(None); continue
                velocity = (np.zeros(2) if self.previous_poses[i] is None else
                            self._visual_delta(pose["center"], self.previous_poses[i], perceptions[i]["shape"]))
                shared_poses.append({"center": pose["center"], "angle": pose["angle"], "velocity": velocity})
                self.previous_poses[i] = pose["center"].copy()
            return [actions[i] if actions[i] is not None
                    else self._policy(i, frames[i], tracks, claimed, shared_poses[1 - i], perceptions[i])
                    for i in range(2)]
        result = []
        for i in range(2):
            tracks = self.local_boards[i].update_points(perceptions[i]["rocks"])
            result.append(actions[i] if actions[i] is not None
                          else self._policy(i, frames[i], tracks, set(), perception=perceptions[i]))
        return result

    def observe(self, before, actions, after, reward, done):
        for i in range(2):
            if not self.grounders[i].complete:
                self.grounders[i].observe(before[i], actions[i], after[i])
        self.rounds += 1

    def status(self):
        return {"controls_grounded": [dict(row.mapping) if row.complete else {} for row in self.grounders],
                "both_controls_grounded": all(row.complete for row in self.grounders),
                "communication": self.communicate, "target_messages": self.target_messages,
                "prevented_friendly_shots": self.prevented_friendly_shots,
                "pixel_track_updates": self.board.updates if self.communicate else sum(b.updates for b in self.local_boards)}


class RandomPair:
    def __init__(self, seed): self.rng = np.random.default_rng(seed)
    def begin(self, frames): pass
    def act(self, frames): return self.rng.integers(0, 5, size=2).tolist()
    def observe(self, *args): pass
    def status(self): return {}


def run_episode(seed, condition="coordinated", config=None, *, capture=False):
    active = 1 if condition == "solo" else 2
    game = DuoAsteroids(config, seed, active_agents=active)
    if condition == "random": policy = RandomPair(seed + 900_001)
    else: policy = TwoRocketTeam(communicate=condition == "coordinated")
    frames = game.reset(seed); policy.begin(frames); movie = [frames[0]] if capture else []
    while not (game.terminated or game.truncated):
        actions = policy.act(frames)
        if condition == "solo": actions[1] = 0
        later, reward, terminated, truncated, info = game.step(actions)
        policy.observe(frames, actions, later, reward, terminated or truncated)
        frames = later
        if capture and (game.steps % 2 == 0 or terminated or truncated): movie.append(frames[0])
    return {"seed": seed, "condition": condition, "hits": game.hits, "hits_by_agent": list(game.hits_by_agent),
            "friendly_fire": game.friendly_fire, "alive": int(np.sum(game.alive)), "both_survive": bool(np.all(game.alive)),
            "lives": list(game.lives), "lives_lost": int(sum(game.deaths)),
            "wave": game.wave, "steps": game.steps, "return": round(game.total_reward, 4),
            "learner": policy.status(), "control_maps_audit_only": [list(row) for row in game.semantic_by_action],
            "friendly_events_audit_only": list(game.friendly_events), "death_events_audit_only": list(game.death_events),
            "movie": movie}


def run_audit(output, *, worlds=12, seed=1_440_001, max_steps=360):
    config = DuoConfig(max_steps=max_steps); rows = []
    for offset in range(worlds):
        world_seed = seed + offset * 1009
        result = {condition: run_episode(world_seed, condition, config)
                  for condition in ("coordinated", "uncommunicative", "solo", "random")}
        for row in result.values(): row.pop("movie", None)
        rows.append({"seed": world_seed, **result})
    aggregate = {}
    for condition in ("coordinated", "uncommunicative", "solo", "random"):
        subset = [row[condition] for row in rows]
        aggregate[condition] = {"mean_hits": round(float(np.mean([x["hits"] for x in subset])), 3),
                                "total_hits": int(sum(x["hits"] for x in subset)),
                                "both_survive": int(sum(x["both_survive"] for x in subset)),
                                "mean_alive": round(float(np.mean([x["alive"] for x in subset])), 3),
                                "friendly_fire": int(sum(x["friendly_fire"] for x in subset)),
                                "lives_lost": int(sum(x["lives_lost"] for x in subset)),
                                "mean_team_return": round(float(np.mean([x["return"] for x in subset])), 3),
                                "mean_wave": round(float(np.mean([x["wave"] for x in subset])), 3)}
    grounded = sum(row["coordinated"]["learner"].get("both_controls_grounded", False) for row in rows)
    report = {"format": "wailah-two-rocket-asteroids-v20-audit-v1", "seed": seed, "worlds": worlds,
              "learner_inputs": ["two RGB frames", "shared scalar reward", "termination"],
              "prohibited_inputs": ["positions", "velocities", "asteroid arrays", "control labels", "simulator state"],
              "aggregate": aggregate, "controls_grounded_worlds": grounded, "per_world": rows}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=12); parser.add_argument("--seed", type=int, default=1_440_001)
    parser.add_argument("--max-steps", type=int, default=360)
    args = parser.parse_args(argv); report = run_audit(args.output, worlds=args.worlds, seed=args.seed, max_steps=args.max_steps)
    print(json.dumps({"aggregate": report["aggregate"], "controls_grounded_worlds": report["controls_grounded_worlds"]}, indent=2))


if __name__ == "__main__": raise SystemExit(main())
