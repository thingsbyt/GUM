"""Immune Savior: a partially observable combination-and-cure game.

The public agent interface is intentionally small: RGB pixels, five primitive
controls, scalar reward, termination, and optional reset. Private semantic names
in this file are used only by the simulator and audit tests; they are never
included in observations supplied to a learner.
"""
from __future__ import annotations

from collections import deque
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .persistent_workshop import CELL, COLORS, DELTAS, GRID, HEIGHT, MASKS, WIDTH, X0, Y0


class ImmuneSaviorGame:
    """Explore a body, synthesize a two-element treatment, and cure safely."""
    action_dim = 5

    def __init__(self, seed: int = 260_031, *, spread_interval: int = 32,
                 max_steps: int = 700):
        self.seed = int(seed); self.spread_interval = int(spread_interval); self.max_steps = int(max_steps)
        rng = np.random.default_rng(seed); self.start = (4, 4)
        permutation = rng.permutation(5).tolist()
        self.control = {permutation[index]: DELTAS[index] for index in range(4)}
        self.interact_action = permutation[4]
        # Four candidates and six possible recipes. Only consequences reveal
        # the two active ingredients.
        self.element_names = [f"element{i}" for i in range(4)]
        self.correct_pair = frozenset(rng.choice(self.element_names, 2, replace=False).tolist())
        station_forms = rng.choice(np.arange(0, 7), 1, replace=False)
        element_forms = rng.choice(np.arange(7, 16), 4, replace=False)
        cell_forms = rng.choice(np.arange(16, 28), 4, replace=False)
        kinds = self.element_names + ["synthesizer", "healthy", "disease", "treatment", "cure"]
        forms = list(element_forms) + [int(station_forms[0])] + list(cell_forms)
        colors = rng.choice(len(COLORS), len(kinds), replace=True)
        self.visual = {kind: (int(form), int(color)) for kind, form, color in zip(kinds, forms, colors)}

        candidates = [(r, c) for r in range(GRID) for c in range(GRID)
                      if (r, c) != self.start and (abs(r - 4) > 1 or abs(c - 4) > 1)]
        rng.shuffle(candidates); self.walls = set()
        for cell in candidates[:12]:
            trial = self.walls | {cell}
            if self._reachable(trial) == GRID * GRID - len(trial): self.walls = trial
        free = [cell for cell in candidates if cell not in self.walls]
        rng.shuffle(free)
        self.fixed = {kind: free[index] for index, kind in enumerate(self.element_names + ["synthesizer"])}
        cursor = 5
        self.initial_healthy = set(free[cursor:cursor + 10]); cursor += 10
        self.initial_disease = set(free[cursor:cursor + 2])
        self.emblem = rng.integers(40, 230, (3, 8, 16), dtype=np.uint8)
        self._spread_rng = np.random.default_rng(seed + 991)
        self.episode = 0

    def _reachable(self, walls):
        seen = {self.start}; queue = deque([self.start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in DELTAS:
                nxt = r + dr, c + dc
                if 0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID and nxt not in walls and nxt not in seen:
                    seen.add(nxt); queue.append(nxt)
        return len(seen)

    def reset(self, seed: int | None = None):
        self.position = self.start; self.inventory = []; self.treatment = False
        self.available_elements = set(self.element_names); self.healthy = set(self.initial_healthy)
        self.disease = set(self.initial_disease); self.damaged_healthy = 0; self.failed_mixtures = 0
        self.steps = 0; self.done = False; self.episode += 1; self.events = []
        return self.render()

    def _target_at(self, cell):
        for kind, where in self.fixed.items():
            if where == cell and (kind not in self.element_names or kind in self.available_elements): return kind
        if cell in self.disease: return "disease"
        if cell in self.healthy: return "healthy"
        return None

    def _interact(self):
        target = self._target_at(self.position); event = "nothing-happened"
        if target in self.element_names and not self.treatment and target not in self.inventory:
            if len(self.inventory) < 2:
                self.inventory.append(target); self.available_elements.remove(target); event = "element-collected"
        elif target == "synthesizer" and len(self.inventory) == 2 and not self.treatment:
            if frozenset(self.inventory) == self.correct_pair:
                self.treatment = True; event = "treatment-created"
            else:
                self.available_elements.update(self.inventory)
                self.failed_mixtures += 1; event = "mixture-inert"
            self.inventory = []
        elif target == "disease" and self.treatment:
            self.disease.remove(self.position); event = "disease-neutralized"
        elif target == "healthy" and self.treatment:
            self.healthy.remove(self.position); self.damaged_healthy += 1; event = "healthy-cell-damaged"
        self.events.append({"step": self.steps, "event": event})
        return event

    def _spread(self):
        if not self.disease: return None
        candidates = []
        blocked = self.walls | set(self.fixed.values()) | self.healthy | self.disease | {self.position}
        for row, col in sorted(self.disease):
            for dr, dc in DELTAS:
                nxt = row + dr, col + dc
                if 0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID and nxt not in blocked: candidates.append(nxt)
        if not candidates: return None
        cell = candidates[int(self._spread_rng.integers(len(candidates)))]; self.disease.add(cell)
        self.events.append({"step": self.steps, "event": "disease-spread"}); return cell

    def step(self, action: int):
        if self.done: raise RuntimeError("episode ended")
        action = int(action); event = "blocked"
        if action in self.control:
            dr, dc = self.control[action]; nxt = self.position[0] + dr, self.position[1] + dc
            if 0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID and nxt not in self.walls:
                self.position = nxt; event = "moved"
        elif action == self.interact_action: event = self._interact()
        else: raise ValueError(action)
        self.steps += 1
        if self.steps % self.spread_interval == 0: self._spread()
        success = not self.disease and self.damaged_healthy == 0
        failure = self.damaged_healthy > 0 or self.steps >= self.max_steps
        self.done = success or failure
        reward = 1.0 if success else -1.0 if self.damaged_healthy else 0.0
        info = {"success": success, "failure": failure and not success,
                "healthy_preserved": len(self.healthy), "healthy_damaged": self.damaged_healthy,
                "disease_remaining": len(self.disease), "event_audit_only": event}
        return self.render(), reward, self.done, info

    def _paint(self, frame, kind, cx, cy):
        form, color = self.visual[kind]; mask, rgb = MASKS[form], COLORS[color]
        for channel in range(3):
            patch = frame[channel, cy - 3:cy + 4, cx - 3:cx + 4]; patch[mask > 0] = rgb[channel]

    def render(self):
        frame = np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8)
        # Goal token means "body cured"; inventory is presented visually with
        # no semantic text or private object names.
        self._paint(frame, "cure", 8, 8)
        if self.treatment: self._paint(frame, "treatment", 96, 8)
        else:
            for index, kind in enumerate(self.inventory): self._paint(frame, kind, 77 + 19 * index, 8)
        frame[:, 4:12, 48:64] = self.emblem; frame[:, 22:24, 3:109] = 24
        visible = {(r, c) for r in range(GRID) for c in range(GRID)
                   if abs(r - self.position[0]) + abs(c - self.position[1]) <= 3}
        for row, col in visible:
            y, x = Y0 + row * CELL, X0 + col * CELL
            color = (72, 78, 88) if (row, col) in self.walls else (12, 18, 27)
            frame[:, y:y + CELL, x:x + CELL] = np.asarray(color, dtype=np.uint8)[:, None, None]
        for kind, cell in self.fixed.items():
            if cell in visible and (kind not in self.element_names or kind in self.available_elements):
                self._paint(frame, kind, X0 + cell[1] * CELL + 5, Y0 + cell[0] * CELL + 5)
        for cell in self.healthy:
            if cell in visible: self._paint(frame, "healthy", X0 + cell[1] * CELL + 5, Y0 + cell[0] * CELL + 5)
        for cell in self.disease:
            if cell in visible: self._paint(frame, "disease", X0 + cell[1] * CELL + 5, Y0 + cell[0] * CELL + 5)
        row, col = self.position; cy, cx = Y0 + row * CELL + 5, X0 + col * CELL + 5
        frame[:, cy - 2:cy + 3, cx] = 255; frame[:, cy, cx - 2:cx + 3] = 255
        return frame

    def audit_state(self):
        """Private verifier state, never part of the learning observation."""
        return {"seed": self.seed, "position": self.position, "inventory": list(self.inventory),
                "treatment": self.treatment, "healthy": len(self.healthy), "disease": len(self.disease),
                "damaged_healthy": self.damaged_healthy, "failed_mixtures": self.failed_mixtures,
                "steps": self.steps, "done": self.done}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=260_031)
    parser.add_argument("--audit", type=Path); parser.add_argument("--frozen-audit", type=Path)
    parser.add_argument("--worlds", type=int, default=8); args = parser.parse_args(argv)
    game = ImmuneSaviorGame(args.seed); frame = game.reset()
    report = {"format": "immune-savior-game-v1", "seed": args.seed,
              "observation_shape": list(frame.shape), "action_dim": game.action_dim,
              "public_interface": ["RGB pixels", "five primitive controls", "reward", "done"],
              "private_state_not_exposed": ["object types", "correct element pair", "disease locations",
                                            "healthy locations", "control mapping"],
              "mechanics": ["partial observability", "disease spread", "two-element synthesis",
                            "selective treatment", "zero healthy-cell damage requirement"]}
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.frozen_audit:
        from .persistent_workshop import WorkshopMind, _run
        baseline_path = Path(__file__).with_name("persistent_workshop.py")
        baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        rows = []
        mind = WorkshopMind()
        for index in range(args.worlds):
            candidate = ImmuneSaviorGame(270_001 + index * 1009, max_steps=900)
            rows.append(_run(mind, candidate, 900, 90))
        frozen = {"format": "immune-savior-frozen-zero-shot-audit-v1",
            "precommitted_baseline_sha256": "73817e6355c596e9a5764aa8002dcda362668f24a4f8d238c0cbec179587e05d",
            "evaluated_baseline_sha256": baseline_hash,
            "hash_matches_precommit": baseline_hash == "73817e6355c596e9a5764aa8002dcda362668f24a4f8d238c0cbec179587e05d",
            "worlds": args.worlds, "interaction_budget_per_world": 900,
            "successes": sum(row["success"] for row in rows),
            "success_rate": sum(row["success"] for row in rows) / max(1, len(rows)),
            "controls_grounded": sum(row["controls_grounded"] for row in rows),
            "diagnosis": ["single-object inventory representation cannot express a two-element mixture",
                          "positive held-object goal cannot express disease absence plus healthy preservation",
                          "no temporal concept for autonomous disease spread",
                          "no safety model for irreversible healthy-cell damage"],
            "rows": rows}
        args.frozen_audit.parent.mkdir(parents=True, exist_ok=True)
        args.frozen_audit.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
        report["frozen_zero_shot"] = {key: value for key, value in frozen.items() if key != "rows"}
    print(json.dumps(report, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
