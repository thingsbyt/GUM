"""Development canvas for the object-centric learner; no tiles or parser schema."""
from __future__ import annotations

import numpy as np


WIDTH = 480; HEIGHT = 360; STEP = 9


def _masks():
    masks = []
    for index in range(30):
        rng = np.random.default_rng(930_000 + index); mask = np.zeros((11, 11), dtype=bool)
        y = x = 5; mask[y, x] = True
        for _ in range(9 + index % 8):
            dy, dx = ((-1, 0), (0, 1), (1, 0), (0, -1))[int(rng.integers(4))]
            y = int(np.clip(y + dy, 1, 9)); x = int(np.clip(x + dx, 1, 9)); mask[y, x] = True
        mask = np.logical_or(mask, np.roll(mask, 1, 0)); mask = np.logical_or(mask, np.roll(mask, 1, 1))
        masks.append(mask)
    return masks


MASKS = _masks()
PALETTE = np.asarray([(238, 88, 72), (65, 177, 232), (72, 211, 139), (243, 194, 72),
                      (177, 111, 224), (238, 137, 62), (62, 207, 198), (246, 112, 174),
                      (143, 216, 255), (183, 235, 103), (246, 178, 122), (191, 155, 242)], dtype=np.uint8)


class FreeformLab:
    action_dim = 5

    def __init__(self, seed=610_001, *, ingredients=4, spread_interval=20, max_steps=1400):
        self.seed = int(seed); self.ingredient_names = [f"sample{index}" for index in range(ingredients)]
        self.spread_interval = int(spread_interval); self.max_steps = int(max_steps); rng = np.random.default_rng(seed)
        permutation = rng.permutation(5).tolist()
        vectors = ((0, -STEP), (STEP, 0), (0, STEP), (-STEP, 0))
        self.control = {permutation[index]: vectors[index] for index in range(4)}
        self.interact_action = permutation[4]; self.start = (240.0, 180.0)
        self.active_pair = frozenset(rng.choice(self.ingredient_names, 2, replace=False).tolist())
        kinds = self.ingredient_names + ["processor", "protected", "threat", "effector"]
        forms = rng.choice(np.arange(len(MASKS)), len(kinds), replace=False)
        colors = rng.choice(len(PALETTE), len(kinds), replace=True)
        self.visual = {kind: (int(form), int(color)) for kind, form, color in zip(kinds, forms, colors)}
        self.visual["effector"] = (int(forms[-1]), -1)
        background = rng.integers(5, 35, 3, dtype=np.uint8); self.background = tuple(int(x) for x in background)
        # Wide separation ensures an interaction has one unambiguous nearest
        # visual object; positions remain reachable by the unknown controls.
        candidates = [(float(x), float(y)) for y in range(63, 334, STEP * 5)
                      for x in range(51, 457, STEP * 5)
                      if np.hypot(x - self.start[0], y - self.start[1]) > 45]
        rng.shuffle(candidates); names = self.ingredient_names + ["processor"]
        self.fixed = {kind: candidates[index] for index, kind in enumerate(names)}
        cursor = len(names); self.initial_protected = set(candidates[cursor:cursor + 9]); cursor += 9
        self.initial_threat = set(candidates[cursor:cursor + 2]); self.spread_pool = candidates[cursor + 2:]
        self._rng = np.random.default_rng(seed + 4141); self.episode = 0

    def reset(self):
        self.position = self.start; self.inventory = []; self.treatment = False
        self.available = set(self.ingredient_names); self.protected = set(self.initial_protected)
        self.threat = set(self.initial_threat); self.protected_damaged = 0; self.failed_pairs = 0
        self.steps = 0; self.done = False; self.events = []; self.episode += 1
        return self.render()

    def _near(self):
        rows = []
        for kind, point in self.fixed.items():
            if kind not in self.ingredient_names or kind in self.available:
                rows.append((np.hypot(point[0] - self.position[0], point[1] - self.position[1]), kind, point))
        rows += [(np.hypot(point[0] - self.position[0], point[1] - self.position[1]), "threat", point)
                 for point in self.threat]
        rows += [(np.hypot(point[0] - self.position[0], point[1] - self.position[1]), "protected", point)
                 for point in self.protected]
        return min(rows, default=(999, None, None))

    def _interact(self):
        distance, target, point = self._near(); event = "no-effect"
        if distance > 22: return event
        if target in self.ingredient_names and not self.treatment and len(self.inventory) < 2:
            self.inventory.append(target); self.available.remove(target); event = "sample-loaded"
        elif target == "processor" and len(self.inventory) == 2 and not self.treatment:
            if frozenset(self.inventory) == self.active_pair:
                self.treatment = True; event = "treatment-produced"
            else:
                self.available.update(self.inventory); self.failed_pairs += 1; event = "pair-rejected"
            self.inventory = []
        elif target == "threat" and self.treatment:
            self.threat.remove(point); event = "threat-removed"
        elif target == "protected" and self.treatment:
            self.protected.remove(point); self.protected_damaged += 1; event = "protected-damaged"
        return event

    def _spread(self):
        if self.treatment: return None
        occupied = set(self.fixed.values()) | self.protected | self.threat | {self.position}
        choices = [point for point in self.spread_pool if point not in occupied]
        if choices:
            point = choices[int(self._rng.integers(len(choices)))]; self.threat.add(point); return point
        return None

    def step(self, action):
        if self.done: raise RuntimeError("episode ended")
        action = int(action); event = "blocked"
        if action in self.control:
            dx, dy = self.control[action]; x = float(np.clip(self.position[0] + dx, 18, WIDTH - 18))
            y = float(np.clip(self.position[1] + dy, 42, HEIGHT - 18)); self.position = (x, y); event = "moved"
        elif action == self.interact_action:
            event = self._interact()
        else:
            raise ValueError(action)
        self.steps += 1
        if self.steps % self.spread_interval == 0: self._spread()
        success = not self.threat and self.protected_damaged == 0
        failure = self.protected_damaged > 0 or self.steps >= self.max_steps
        self.done = success or failure; reward = 1.0 if success else -1.0 if self.protected_damaged else 0.0
        info = {"success": success, "failure": failure and not success, "protected_damaged": self.protected_damaged,
                "protected_remaining": len(self.protected), "threat_remaining": len(self.threat),
                "event_audit_only": event}
        self.events.append({"step": self.steps, "event": event})
        return self.render(), reward, self.done, info

    def _draw(self, frame, kind, point):
        form, color = self.visual[kind]; mask = MASKS[form]
        rgb = np.asarray((250, 250, 250), dtype=np.uint8) if color < 0 else PALETTE[color]
        x, y = int(round(point[0])), int(round(point[1])); y0, x0 = y - 5, x - 5
        patch = frame[y0:y0 + 11, x0:x0 + 11]; patch[mask] = rgb

    def render(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8); frame[:] = self.background
        for kind, point in self.fixed.items():
            if kind not in self.ingredient_names or kind in self.available: self._draw(frame, kind, point)
        for point in self.protected: self._draw(frame, "protected", point)
        for point in self.threat: self._draw(frame, "threat", point)
        for index, kind in enumerate(self.inventory): self._draw(frame, kind, (32.0 + index * 28.0, 20.0))
        # A cursor is rendered on its own compositing layer, as in an ordinary
        # GUI, so passing over an object does not merge their components.
        x, y = int(self.position[0]), int(self.position[1])
        frame[y - 7:y + 8, x - 7:x + 8] = self.background
        self._draw(frame, "effector", self.position); return frame

    def audit_state(self):
        return {"seed": self.seed, "position": self.position, "inventory": list(self.inventory),
                "treatment": self.treatment, "protected": len(self.protected), "threat": len(self.threat),
                "protected_damaged": self.protected_damaged, "failed_pairs": self.failed_pairs,
                "steps": self.steps, "done": self.done}


def run_object_episode(agent, game, budget=5000, horizon=500):
    used = episodes = damage = 0; success = False; final = {}; trace = []
    while used < budget and not success:
        frame = game.reset(); agent.begin(frame); episodes += 1
        for _ in range(min(horizon, budget - used)):
            action, evidence = agent.act(frame); nxt, reward, done, info = game.step(action)
            learned = agent.observe(frame, action, nxt, reward, done); frame = nxt; used += 1; final = info
            if evidence["reason"] in ("discover-transformation-surface", "execute-two-object-transformation",
                                      "safely-test-or-apply-treatment"):
                trace.append({"interaction": used, **evidence, "learned": learned})
            if done:
                damage += int(info["protected_damaged"]); success = bool(info["success"]); break
    return {"success": success, "interactions": used, "episodes": episodes, "protected_damaged": damage,
            "threat_remaining": int(final.get("threat_remaining", -1)),
            "recipe_learned": agent.memory.get("successful_pair") is not None,
            "controls_grounded": len(agent.memory["controls"]) == 4 and agent.memory["interact"] is not None,
            "trace": trace[-24:]}
