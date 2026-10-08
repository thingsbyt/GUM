"""Asteroids physics retained from the earlier game, now with RGB observations.
The world model never receives snapshot(), reward, positions or velocities.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np

ACTION_NAMES = ('A0', 'A1', 'A2', 'A3', 'A4')
# This private-to-the-environment mapping is not fed to the neural network.
CONTROL_NAMES = ('coast', 'turn left', 'turn right', 'thrust', 'fire')
N_ACTIONS = 5
MAX_ROCKS = 12
MAX_BULLETS = 10


@dataclass
class GameConfig:
    difficulty: int = 1
    max_steps: int = 900
    frames_per_action: int = 6
    width: float = 640.0
    height: float = 320.0
    image_width: int = 512
    image_height: int = 256

class Asteroids:
    def __init__(self, config: GameConfig | None = None, seed: int = 0):
        self.config = config or GameConfig()
        if self.config.difficulty not in (1, 2, 3):
            raise ValueError('difficulty must be 1, 2, or 3')
        if self.config.max_steps < 1 or self.config.frames_per_action < 1:
            raise ValueError('max_steps and frames_per_action must be positive')
        self.size = np.array([self.config.width, self.config.height], dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(seed)
        self.ship = self.size / 2.0
        self.velocity = np.zeros(2, dtype=np.float64)
        self.angle = float(self.rng.uniform(-math.pi, math.pi))
        self.cooldown = 0.0
        self.shield = 0.75
        self.rocks = np.zeros((MAX_ROCKS, 5), dtype=np.float64)  # x,y,vx,vy,r
        self.bullets = np.zeros((MAX_BULLETS, 5), dtype=np.float64)  # x,y,vx,vy,ttl
        self.steps = self.frames = self.hits = self.wave = 0
        self.terminated = self.truncated = False
        self.total_reward = 0.0
        self.last_action = 0
        self.last_hits = 0
        self._spawn_wave()
        return self.observe()

    def _delta(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a - b + self.size / 2.0) % self.size - self.size / 2.0

    def _spawn_wave(self) -> None:
        self.wave += 1
        n = (2, 4, 6)[self.config.difficulty - 1]
        speed_low, speed_high = ((24, 48), (38, 68), (52, 86))[self.config.difficulty - 1]
        for i in range(n):
            pos = self.rng.uniform([0, 0], self.size)
            for _ in range(100):
                if np.linalg.norm(self._delta(pos, self.ship)) > 140:
                    break
                pos = self.rng.uniform([0, 0], self.size)
            theta = self.rng.uniform(-math.pi, math.pi)
            speed = self.rng.uniform(speed_low, speed_high)
            self.rocks[i] = (*pos, math.cos(theta)*speed, math.sin(theta)*speed, 30.0)

    def _fire(self) -> None:
        if self.cooldown > 1e-9:
            return
        slots = np.flatnonzero(self.bullets[:, 4] <= 0)
        if not len(slots):
            return
        direction = np.array([math.cos(self.angle), math.sin(self.angle)])
        pos = (self.ship + direction * 12.0) % self.size
        vel = self.velocity + direction * 360.0
        self.bullets[slots[0]] = (*pos, *vel, 1.5)
        self.cooldown = 0.18

    def _destroy_rock(self, i: int) -> None:
        rock = self.rocks[i].copy()
        self.rocks[i] = 0
        if rock[4] > 20:
            # Deterministic splitting: no random future consulted by the policy.
            theta = math.atan2(rock[3], rock[2])
            speed = max(48.0, float(np.linalg.norm(rock[2:4])) * 1.35)
            for rotation in (-0.65, 0.65):
                slots = np.flatnonzero(self.rocks[:, 4] <= 0)
                if not len(slots):
                    break
                heading = theta + rotation
                self.rocks[slots[0]] = (*rock[:2], math.cos(heading)*speed,
                                         math.sin(heading)*speed, 15.0)
        self.hits += 1
        self.last_hits += 1

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.terminated or self.truncated:
            raise RuntimeError('Episode ended; call reset before step')
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < N_ACTIONS:
            raise ValueError('Action must be an integer from 0 through 4')
        self.last_action = int(action)
        self.last_hits = 0
        dt = 1.0 / 60.0
        frames = 0
        for _ in range(self.config.frames_per_action):
            frames += 1
            self.frames += 1
            self.cooldown = max(0.0, self.cooldown - dt)
            self.shield = max(0.0, self.shield - dt)
            if action in (1, 2):
                self.angle = (self.angle + (-1 if action == 1 else 1)*3.7*dt + math.pi) % (2*math.pi) - math.pi
            if action == 3:
                self.velocity += np.array([math.cos(self.angle), math.sin(self.angle)]) * 185.0 * dt
            self.velocity *= 0.997
            speed = np.linalg.norm(self.velocity)
            if speed > 240:
                self.velocity *= 240.0 / speed
            if action == 4:
                self._fire()
            self.ship = (self.ship + self.velocity*dt) % self.size
            alive = self.rocks[:, 4] > 0
            self.rocks[alive, :2] = (self.rocks[alive, :2] + self.rocks[alive, 2:4]*dt) % self.size
            active = self.bullets[:, 4] > 0
            self.bullets[active, :2] = (self.bullets[active, :2] + self.bullets[active, 2:4]*dt) % self.size
            self.bullets[active, 4] -= dt
            self.bullets[self.bullets[:, 4] <= 0] = 0
            # Each bullet can hit at most one rock; newly split rocks wait a frame.
            bi = np.flatnonzero(self.bullets[:, 4] > 0)
            ri = np.flatnonzero(self.rocks[:, 4] > 0)
            if len(bi) and len(ri):
                d = self._delta(self.bullets[bi, None, :2], self.rocks[None, ri, :2])
                pairs = np.argwhere((d*d).sum(axis=-1) <= (self.rocks[ri, 4][None, :] + 2.0)**2)
                used_b, used_r = set(), set()
                for b, r in pairs:
                    b, r = int(bi[b]), int(ri[r])
                    if b not in used_b and r not in used_r:
                        self.bullets[b] = 0
                        self._destroy_rock(r)
                        used_b.add(b)
                        used_r.add(r)
            ri = np.flatnonzero(self.rocks[:, 4] > 0)
            if self.shield <= 0 and len(ri):
                d = self._delta(self.rocks[ri, :2], self.ship)
                if np.any((d*d).sum(axis=1) <= (self.rocks[ri, 4] + 8.0)**2):
                    self.terminated = True
                    break
            if not len(ri):
                self._spawn_wave()
                self.shield = max(self.shield, 0.5)
        self.steps += 1
        self.truncated = self.steps >= self.config.max_steps and not self.terminated
        survival_reward = 0.002 * frames / self.config.frames_per_action
        reward = float(self.last_hits - 5.0*int(self.terminated) + survival_reward)
        self.total_reward += reward
        return self.observe(), reward, self.terminated, self.truncated, {
            'hits': self.last_hits, 'total_hits': self.hits, 'seconds': self.frames / 60.0,
            'return': self.total_reward, 'wave': self.wave, 'survival_reward': survival_reward,
        }

    def observe(self) -> np.ndarray:
        from .rendering import render_snapshot
        return render_snapshot(self.snapshot(), self.config.image_width, self.config.image_height)

    def snapshot(self) -> dict:
        return {'size': self.size.tolist(), 'ship': self.ship.tolist(),
                'velocity': self.velocity.tolist(), 'angle': self.angle,
                'shield': self.shield, 'rocks': self.rocks[self.rocks[:, 4] > 0].tolist(),
                'bullets': self.bullets[self.bullets[:, 4] > 0].tolist(),
                'hits': self.hits, 'seconds': round(self.frames/60.0, 2), 'wave': self.wave,
                'action': self.last_action, 'dead': self.terminated, 'timeout': self.truncated,
                'return': round(self.total_reward, 4), 'config': asdict(self.config)}
