"""Crystal Garden: a held-out ecology challenge for the frozen v14 learner.

The learner sees only pixels, anonymous controls, reward and termination.  All
semantic names and verifier state in this module remain private to the game.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from .persistent_workshop import CELL, COLORS, DELTAS, GRID, HEIGHT, MASKS, WIDTH, X0, Y0


class CrystalGardenGame:
    """Grow a counter-resonance and remove spreading crystal blight safely."""
    action_dim = 5

    def __init__(self, seed: int = 420_001, *, spread_interval: int = 37,
                 max_steps: int = 1000):
        self.seed = int(seed); self.spread_interval = int(spread_interval); self.max_steps = int(max_steps)
        rng = np.random.default_rng(seed); self.start = (4, 4)
        permutation = rng.permutation(5).tolist()
        self.control = {permutation[index]: DELTAS[index] for index in range(4)}
        self.interact_action = permutation[4]
        self.shard_names = [f"shard{index}" for index in range(5)]
        self.active_pair = frozenset(rng.choice(self.shard_names, 2, replace=False).tolist())

        altar_form = rng.choice(np.arange(0, 7), 1, replace=False)
        shard_forms = rng.choice(np.arange(7, 16), 5, replace=False)
        ecology_forms = rng.choice(np.arange(16, 28), 4, replace=False)
        kinds = self.shard_names + ["altar", "native", "blight", "resonance", "balance"]
        forms = list(shard_forms) + [int(altar_form[0])] + list(ecology_forms)
        colors = rng.choice(len(COLORS), len(kinds), replace=True)
        self.visual = {kind: (int(form), int(color)) for kind, form, color in zip(kinds, forms, colors)}

        candidates = [(row, col) for row in range(GRID) for col in range(GRID)
                      if (row, col) != self.start and (abs(row - 4) > 1 or abs(col - 4) > 1)]
        rng.shuffle(candidates); self.walls = set()
        for cell in candidates[:14]:
            trial = self.walls | {cell}
            if self._reachable(trial) == GRID * GRID - len(trial): self.walls = trial
        free = [cell for cell in candidates if cell not in self.walls]
        rng.shuffle(free)
        names = self.shard_names + ["altar"]
        self.fixed = {kind: free[index] for index, kind in enumerate(names)}
        cursor = len(names)
        self.initial_native = set(free[cursor:cursor + 11]); cursor += 11
        self.initial_blight = set(free[cursor:cursor + 2])
        self.emblem = rng.integers(45, 225, (3, 8, 16), dtype=np.uint8)
        self._spread_rng = np.random.default_rng(seed + 17_071); self.episode = 0

    def _reachable(self, walls):
        seen = {self.start}; queue = deque([self.start])
        while queue:
            row, col = queue.popleft()
            for dr, dc in DELTAS:
                nxt = row + dr, col + dc
                if (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID and
                        nxt not in walls and nxt not in seen):
                    seen.add(nxt); queue.append(nxt)
        return len(seen)

    def reset(self):
        self.position = self.start; self.inventory = []; self.resonance = False
        self.available_shards = set(self.shard_names)
        self.native = set(self.initial_native); self.blight = set(self.initial_blight)
        self.damaged_native = 0; self.failed_pairs = 0; self.steps = 0; self.done = False
        self.events = []; self.episode += 1
        return self.render()

    def _target_at(self, cell):
        for kind, where in self.fixed.items():
            if where == cell and (kind not in self.shard_names or kind in self.available_shards): return kind
        if cell in self.blight: return "blight"
        if cell in self.native: return "native"
        return None

    def _interact(self):
        target = self._target_at(self.position); event = "no-effect"
        if target in self.shard_names and not self.resonance and target not in self.inventory:
            if len(self.inventory) < 2:
                self.inventory.append(target); self.available_shards.remove(target); event = "shard-gathered"
        elif target == "altar" and len(self.inventory) == 2 and not self.resonance:
            if frozenset(self.inventory) == self.active_pair:
                self.resonance = True; event = "counter-resonance-created"
            else:
                self.available_shards.update(self.inventory); self.failed_pairs += 1; event = "unstable-pair-dissolved"
            self.inventory = []
        elif target == "blight" and self.resonance:
            self.blight.remove(self.position); event = "blight-dissolved"
        elif target == "native" and self.resonance:
            self.native.remove(self.position); self.damaged_native += 1; event = "native-damaged"
        self.events.append({"step": self.steps, "event": event}); return event

    def _spread(self):
        if not self.blight: return None
        blocked = self.walls | set(self.fixed.values()) | self.native | self.blight | {self.position}
        candidates = []
        for row, col in sorted(self.blight):
            for dr, dc in DELTAS:
                nxt = row + dr, col + dc
                if (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID and nxt not in blocked): candidates.append(nxt)
        if not candidates: return None
        cell = candidates[int(self._spread_rng.integers(len(candidates)))]; self.blight.add(cell)
        self.events.append({"step": self.steps, "event": "blight-expanded"}); return cell

    def step(self, action: int):
        if self.done: raise RuntimeError("episode ended")
        action = int(action); event = "blocked"
        if action in self.control:
            dr, dc = self.control[action]; nxt = self.position[0] + dr, self.position[1] + dc
            if 0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID and nxt not in self.walls:
                self.position = nxt; event = "moved"
        elif action == self.interact_action:
            event = self._interact()
        else:
            raise ValueError(action)
        self.steps += 1
        if self.steps % self.spread_interval == 0: self._spread()
        success = not self.blight and self.damaged_native == 0
        failure = self.damaged_native > 0 or self.steps >= self.max_steps
        self.done = success or failure
        reward = 1.0 if success else -1.0 if self.damaged_native else 0.0
        info = {"success": success, "failure": failure and not success,
                "protected_remaining": len(self.native), "protected_damaged": self.damaged_native,
                "threat_remaining": len(self.blight), "event_audit_only": event}
        return self.render(), reward, self.done, info

    def _paint(self, frame, kind, cx, cy):
        form, color = self.visual[kind]; mask, rgb = MASKS[form], COLORS[color]
        for channel in range(3):
            patch = frame[channel, cy - 3:cy + 4, cx - 3:cx + 4]; patch[mask > 0] = rgb[channel]

    def render(self):
        frame = np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8)
        self._paint(frame, "balance", 8, 8)
        if self.resonance:
            self._paint(frame, "resonance", 96, 8)
        else:
            for index, kind in enumerate(self.inventory): self._paint(frame, kind, 77 + 19 * index, 8)
        frame[:, 4:12, 48:64] = self.emblem; frame[:, 22:24, 3:109] = 31
        visible = {(row, col) for row in range(GRID) for col in range(GRID)
                   if abs(row - self.position[0]) + abs(col - self.position[1]) <= 3}
        for row, col in visible:
            y, x = Y0 + row * CELL, X0 + col * CELL
            # The frozen tiled parser's public interface uses this exact wall
            # marker.  Changing it would be an observation-protocol change,
            # not a legitimate held-out semantic variation.
            color = (72, 78, 88) if (row, col) in self.walls else (9, 28, 23)
            frame[:, y:y + CELL, x:x + CELL] = np.asarray(color, dtype=np.uint8)[:, None, None]
        for kind, cell in self.fixed.items():
            if cell in visible and (kind not in self.shard_names or kind in self.available_shards):
                self._paint(frame, kind, X0 + cell[1] * CELL + 5, Y0 + cell[0] * CELL + 5)
        for cell in self.native:
            if cell in visible: self._paint(frame, "native", X0 + cell[1] * CELL + 5, Y0 + cell[0] * CELL + 5)
        for cell in self.blight:
            if cell in visible: self._paint(frame, "blight", X0 + cell[1] * CELL + 5, Y0 + cell[0] * CELL + 5)
        row, col = self.position; cy, cx = Y0 + row * CELL + 5, X0 + col * CELL + 5
        frame[:, cy - 2:cy + 3, cx] = 255; frame[:, cy, cx - 2:cx + 3] = 255
        return frame

    def audit_state(self):
        return {"seed": self.seed, "position": self.position, "inventory": list(self.inventory),
                "resonance": self.resonance, "protected": len(self.native), "threat": len(self.blight),
                "protected_damaged": self.damaged_native, "failed_pairs": self.failed_pairs,
                "steps": self.steps, "done": self.done}
