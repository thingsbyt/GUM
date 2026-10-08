"""Held-out free-form industrial task for the frozen object-centric agent."""
from __future__ import annotations

import numpy as np


WIDTH = 540; HEIGHT = 400; STEP = 11


def _masks():
    rows = []
    for index in range(36):
        rng = np.random.default_rng(1_170_000 + index); mask = np.zeros((9, 9), dtype=bool)
        y = x = 4; mask[y, x] = True
        for _ in range(7 + index % 7):
            dy, dx = ((-1, 0), (0, 1), (1, 0), (0, -1))[int(rng.integers(4))]
            y = int(np.clip(y + dy, 1, 7)); x = int(np.clip(x + dx, 1, 7)); mask[y, x] = True
        mask = np.logical_or(mask, np.roll(mask, -1, 0)); mask = np.logical_or(mask, np.roll(mask, 1, 1))
        rows.append(mask)
    return rows


MASKS = _masks()
PALETTE = np.asarray([(224, 74, 96), (54, 192, 226), (82, 218, 126), (248, 184, 62),
                      (164, 102, 232), (242, 122, 52), (52, 211, 185), (237, 96, 184),
                      (118, 205, 248), (171, 232, 88), (247, 158, 111), (205, 138, 239),
                      (104, 224, 210), (245, 213, 91)], dtype=np.uint8)


class WarehouseSpill:
    action_dim = 5

    def __init__(self, seed=710_001, *, spread_interval=23, max_steps=1600):
        self.seed = int(seed); self.cartridge_names = [f"cartridge{index}" for index in range(6)]
        self.spread_interval = int(spread_interval); self.max_steps = int(max_steps); rng = np.random.default_rng(seed)
        permutation = rng.permutation(5).tolist(); vectors = ((0, -STEP), (STEP, 0), (0, STEP), (-STEP, 0))
        self.control = {permutation[index]: vectors[index] for index in range(4)}
        self.interact_action = permutation[4]; self.start = (278.0, 202.0)
        self.active_pair = frozenset(rng.choice(self.cartridge_names, 2, replace=False).tolist())
        kinds = self.cartridge_names + ["neutralizer", "package", "spill", "forklift"]
        forms = rng.choice(np.arange(len(MASKS)), len(kinds), replace=False)
        colors = rng.choice(len(PALETTE), len(kinds), replace=True)
        self.visual = {kind: (int(form), int(color)) for kind, form, color in zip(kinds, forms, colors)}
        self.visual["forklift"] = (int(forms[-1]), -1)
        self.background = tuple(int(x) for x in rng.integers(4, 31, 3, dtype=np.uint8))
        candidates = [(float(x), float(y)) for y in range(70, 346, STEP * 5)
                      for x in range(58, 499, STEP * 5)
                      if np.hypot(x - self.start[0], y - self.start[1]) > 48]
        rng.shuffle(candidates); names = self.cartridge_names + ["neutralizer"]
        self.fixed = {kind: candidates[index] for index, kind in enumerate(names)}
        cursor = len(names); self.initial_packages = set(candidates[cursor:cursor + 10]); cursor += 10
        self.initial_spill = set(candidates[cursor:cursor + 2]); self.spread_pool = candidates[cursor + 2:]
        self._rng = np.random.default_rng(seed + 9919); self.episode = 0

    def reset(self):
        self.position = self.start; self.inventory = []; self.neutralizer_ready = False
        self.available = set(self.cartridge_names); self.packages = set(self.initial_packages)
        self.spill = set(self.initial_spill); self.packages_damaged = 0; self.failed_pairs = 0
        self.steps = 0; self.done = False; self.events = []; self.episode += 1
        return self.render()

    def _near(self):
        rows = []
        for kind, point in self.fixed.items():
            if kind not in self.cartridge_names or kind in self.available:
                rows.append((np.hypot(point[0] - self.position[0], point[1] - self.position[1]), kind, point))
        rows += [(np.hypot(point[0] - self.position[0], point[1] - self.position[1]), "spill", point)
                 for point in self.spill]
        rows += [(np.hypot(point[0] - self.position[0], point[1] - self.position[1]), "package", point)
                 for point in self.packages]
        return min(rows, default=(999, None, None))

    def _interact(self):
        distance, target, point = self._near(); event = "no-effect"
        if distance > 22: return event
        if target in self.cartridge_names and not self.neutralizer_ready and len(self.inventory) < 2:
            self.inventory.append(target); self.available.remove(target); event = "cartridge-loaded"
        elif target == "neutralizer" and len(self.inventory) == 2 and not self.neutralizer_ready:
            if frozenset(self.inventory) == self.active_pair:
                self.neutralizer_ready = True; event = "neutralizer-activated"
            else:
                self.available.update(self.inventory); self.failed_pairs += 1; event = "mixture-rejected"
            self.inventory = []
        elif target == "spill" and self.neutralizer_ready:
            self.spill.remove(point); event = "spill-neutralized"
        elif target == "package" and self.neutralizer_ready:
            self.packages.remove(point); self.packages_damaged += 1; event = "package-damaged"
        return event

    def _spread(self):
        if self.neutralizer_ready: return None
        occupied = set(self.fixed.values()) | self.packages | self.spill | {self.position}
        choices = [point for point in self.spread_pool if point not in occupied]
        if choices:
            point = choices[int(self._rng.integers(len(choices)))]; self.spill.add(point); return point
        return None

    def step(self, action):
        if self.done: raise RuntimeError("episode ended")
        action = int(action); event = "blocked"
        if action in self.control:
            dx, dy = self.control[action]
            self.position = (float(np.clip(self.position[0] + dx, 15, WIDTH - 15)),
                             float(np.clip(self.position[1] + dy, 15, HEIGHT - 15))); event = "forklift-moved"
        elif action == self.interact_action:
            event = self._interact()
        else:
            raise ValueError(action)
        self.steps += 1
        if self.steps % self.spread_interval == 0: self._spread()
        success = not self.spill and self.packages_damaged == 0
        failure = self.packages_damaged > 0 or self.steps >= self.max_steps
        self.done = success or failure; reward = 1.0 if success else -1.0 if self.packages_damaged else 0.0
        info = {"success": success, "failure": failure and not success,
                "protected_damaged": self.packages_damaged, "protected_remaining": len(self.packages),
                "threat_remaining": len(self.spill), "event_audit_only": event}
        self.events.append({"step": self.steps, "event": event})
        return self.render(), reward, self.done, info

    def _draw(self, frame, kind, point):
        form, color = self.visual[kind]; mask = MASKS[form]
        rgb = np.asarray((252, 252, 252), dtype=np.uint8) if color < 0 else PALETTE[color]
        x, y = int(round(point[0])), int(round(point[1])); patch = frame[y - 4:y + 5, x - 4:x + 5]
        patch[mask] = rgb

    def render(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8); frame[:] = self.background
        for kind, point in self.fixed.items():
            if kind not in self.cartridge_names or kind in self.available: self._draw(frame, kind, point)
        for point in self.packages: self._draw(frame, "package", point)
        for point in self.spill: self._draw(frame, "spill", point)
        for index, kind in enumerate(self.inventory): self._draw(frame, kind, (38.0 + index * 32.0, 382.0))
        x, y = int(self.position[0]), int(self.position[1]); frame[y - 5:y + 6, x - 5:x + 6] = self.background
        self._draw(frame, "forklift", self.position); return frame

    def audit_state(self):
        return {"seed": self.seed, "position": self.position, "inventory": list(self.inventory),
                "neutralizer_ready": self.neutralizer_ready, "packages": len(self.packages),
                "spill": len(self.spill), "packages_damaged": self.packages_damaged,
                "failed_pairs": self.failed_pairs, "steps": self.steps, "done": self.done}
