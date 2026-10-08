"""Held-out dashboard data workflow for the frozen adaptive v16 agent."""
from __future__ import annotations

from collections import deque
import numpy as np

from .persistent_workshop import CELL, COLORS, DELTAS, GRID, HEIGHT, MASKS, WIDTH, X0, Y0


class DataRecoveryConsole:
    """Compile two unknown modules, purge growing corruption, preserve records."""
    action_dim = 5
    raw_height = 440; raw_width = 620

    def __init__(self, seed: int = 510_001, *, spread_interval=41, max_steps=1200):
        self.seed = int(seed); self.spread_interval = int(spread_interval); self.max_steps = int(max_steps)
        rng = np.random.default_rng(seed); self.start = (4, 4)
        permutation = rng.permutation(5).tolist()
        self.control = {permutation[index]: DELTAS[index] for index in range(4)}
        self.interact_action = permutation[4]
        self.module_names = [f"module{index}" for index in range(6)]
        self.active_pair = frozenset(rng.choice(self.module_names, 2, replace=False).tolist())
        compiler_form = rng.choice(np.arange(0, 7), 1, replace=False)
        module_forms = rng.choice(np.arange(7, 16), 6, replace=False)
        data_forms = rng.choice(np.arange(16, 28), 4, replace=False)
        kinds = self.module_names + ["compiler", "record", "corruption", "patch", "recovered"]
        forms = list(module_forms) + [int(compiler_form[0])] + list(data_forms)
        colors = rng.choice(len(COLORS), len(kinds), replace=True)
        self.visual = {kind: (int(form), int(color)) for kind, form, color in zip(kinds, forms, colors)}

        candidates = [(row, col) for row in range(GRID) for col in range(GRID)
                      if (row, col) != self.start and (abs(row - 4) > 1 or abs(col - 4) > 1)]
        rng.shuffle(candidates); self.walls = set()
        for cell in candidates[:13]:
            trial = self.walls | {cell}
            if self._reachable(trial) == GRID * GRID - len(trial): self.walls = trial
        free = [cell for cell in candidates if cell not in self.walls]; rng.shuffle(free)
        names = self.module_names + ["compiler"]
        self.fixed = {kind: free[index] for index, kind in enumerate(names)}
        cursor = len(names); self.initial_records = set(free[cursor:cursor + 9]); cursor += 9
        self.initial_corruption = set(free[cursor:cursor + 2])
        self.emblem = rng.integers(42, 226, (3, 8, 16), dtype=np.uint8)
        self._spread_rng = np.random.default_rng(seed + 81_337); self.episode = 0
        self.scale = 2 + ((seed // 11) % 2); self.viewport_top = 31 + seed % 19
        self.viewport_left = 148 + seed % 97

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
        self.position = self.start; self.inventory = []; self.patch_ready = False
        self.available_modules = set(self.module_names); self.records = set(self.initial_records)
        self.corruption = set(self.initial_corruption); self.damaged_records = 0; self.failed_compiles = 0
        self.steps = 0; self.done = False; self.events = []; self.episode += 1
        return self.render()

    def _target_at(self, cell):
        for kind, where in self.fixed.items():
            if where == cell and (kind not in self.module_names or kind in self.available_modules): return kind
        if cell in self.corruption: return "corruption"
        if cell in self.records: return "record"
        return None

    def _interact(self):
        target = self._target_at(self.position); event = "no-op"
        if target in self.module_names and not self.patch_ready and target not in self.inventory:
            if len(self.inventory) < 2:
                self.inventory.append(target); self.available_modules.remove(target); event = "module-loaded"
        elif target == "compiler" and len(self.inventory) == 2 and not self.patch_ready:
            if frozenset(self.inventory) == self.active_pair:
                self.patch_ready = True; event = "repair-patch-compiled"
            else:
                self.available_modules.update(self.inventory); self.failed_compiles += 1; event = "compile-rejected"
            self.inventory = []
        elif target == "corruption" and self.patch_ready:
            self.corruption.remove(self.position); event = "corruption-purged"
        elif target == "record" and self.patch_ready:
            self.records.remove(self.position); self.damaged_records += 1; event = "record-damaged"
        self.events.append({"step": self.steps, "event": event}); return event

    def _spread(self):
        if not self.corruption: return None
        blocked = self.walls | set(self.fixed.values()) | self.records | self.corruption | {self.position}
        candidates = []
        for row, col in sorted(self.corruption):
            for dr, dc in DELTAS:
                nxt = row + dr, col + dc
                if 0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID and nxt not in blocked: candidates.append(nxt)
        if not candidates: return None
        cell = candidates[int(self._spread_rng.integers(len(candidates)))]; self.corruption.add(cell)
        self.events.append({"step": self.steps, "event": "corruption-replicated"}); return cell

    def step(self, action):
        if self.done: raise RuntimeError("episode ended")
        action = int(action); event = "blocked"
        if action in self.control:
            dr, dc = self.control[action]; nxt = self.position[0] + dr, self.position[1] + dc
            if 0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID and nxt not in self.walls:
                self.position = nxt; event = "cursor-moved"
        elif action == self.interact_action:
            event = self._interact()
        else:
            raise ValueError(action)
        self.steps += 1
        if self.steps % self.spread_interval == 0: self._spread()
        success = not self.corruption and self.damaged_records == 0
        failure = self.damaged_records > 0 or self.steps >= self.max_steps
        self.done = success or failure; reward = 1.0 if success else -1.0 if self.damaged_records else 0.0
        info = {"success": success, "failure": failure and not success,
                "protected_remaining": len(self.records), "protected_damaged": self.damaged_records,
                "threat_remaining": len(self.corruption), "event_audit_only": event}
        return self.render(), reward, self.done, info

    def _paint(self, frame, kind, cx, cy):
        form, color = self.visual[kind]; mask, rgb = MASKS[form], COLORS[color]
        for channel in range(3):
            patch = frame[channel, cy - 3:cy + 4, cx - 3:cx + 4]; patch[mask > 0] = rgb[channel]

    def _inner(self):
        frame = np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8); self._paint(frame, "recovered", 8, 8)
        if self.patch_ready:
            self._paint(frame, "patch", 96, 8)
        else:
            for index, kind in enumerate(self.inventory): self._paint(frame, kind, 77 + 19 * index, 8)
        frame[:, 4:12, 48:64] = self.emblem; frame[:, 22:24, 3:109] = 29
        visible = {(row, col) for row in range(GRID) for col in range(GRID)
                   if abs(row - self.position[0]) + abs(col - self.position[1]) <= 3}
        for row, col in visible:
            y, x = Y0 + row * CELL, X0 + col * CELL
            color = (72, 78, 88) if (row, col) in self.walls else (13, 20, 31)
            frame[:, y:y + CELL, x:x + CELL] = np.asarray(color, dtype=np.uint8)[:, None, None]
        for kind, cell in self.fixed.items():
            if cell in visible and (kind not in self.module_names or kind in self.available_modules):
                self._paint(frame, kind, X0 + cell[1] * CELL + 5, Y0 + cell[0] * CELL + 5)
        for cell in self.records:
            if cell in visible: self._paint(frame, "record", X0 + cell[1] * CELL + 5, Y0 + cell[0] * CELL + 5)
        for cell in self.corruption:
            if cell in visible: self._paint(frame, "corruption", X0 + cell[1] * CELL + 5, Y0 + cell[0] * CELL + 5)
        row, col = self.position; cy, cx = Y0 + row * CELL + 5, X0 + col * CELL + 5
        frame[:, cy - 2:cy + 3, cx] = 255; frame[:, cy, cx - 2:cx + 3] = 255
        return frame

    def render(self):
        rng = np.random.default_rng(self.seed + 303)
        raw = np.zeros((self.raw_height, self.raw_width, 3), dtype=np.uint8); raw[:] = (15, 20, 29)
        raw[:26] = (36, 45, 61); raw[:, :112] = (23, 30, 43); raw[32:35, 125:] = (48, 59, 76)
        for row in range(7):
            y = 50 + row * 49; color = rng.integers(35, 105, 3, dtype=np.uint8)
            raw[y:y + 30, 17:95] = color
        for index in range(4):
            x = 475 + index * 29; raw[68:120 + index * 34, x:x + 18] = rng.integers(28, 90, 3, dtype=np.uint8)
        inner = np.transpose(self._inner(), (1, 2, 0))
        enlarged = np.repeat(np.repeat(inner, self.scale, axis=0), self.scale, axis=1)
        top, left = self.viewport_top, self.viewport_left
        raw[top:top + HEIGHT * self.scale, left:left + WIDTH * self.scale] = enlarged
        return raw

    def audit_state(self):
        return {"seed": self.seed, "scale": self.scale, "viewport": [self.viewport_top, self.viewport_left],
                "position": self.position, "inventory": list(self.inventory), "patch_ready": self.patch_ready,
                "records": len(self.records), "corruption": len(self.corruption),
                "damaged_records": self.damaged_records, "failed_compiles": self.failed_compiles,
                "steps": self.steps, "done": self.done}
