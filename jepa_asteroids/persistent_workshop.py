"""Persistent partially-observable workshop for grounded autonomous learning.

The learner receives RGB frames, five anonymous primitive controls, sparse
terminal reward, and reset signals. It learns its controls from visual motion,
maps a fog-of-war grid, remembers objects after they leave view, proposes
interaction experiments, records transformations, and plans with the causal
rules it discovered. Object ids and recipes remain private to the environment.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


GRID = 9; CELL = 10; X0 = 11; Y0 = 31; HEIGHT = 126; WIDTH = 112
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
# Deliberately excludes the rule-key delimiter used in persistent JSON memory.
EMPTY = "EMPTY"


def _mask_bank() -> list[np.ndarray]:
    rows = []
    for index in range(28):
        rng = np.random.default_rng(810_000 + index); value = np.zeros((7, 7), dtype=np.uint8)
        y = x = 3; value[y, x] = 1
        for _ in range(11 + index % 7):
            dy, dx = DELTAS[int(rng.integers(4))]
            y = int(np.clip(y + dy, 0, 6)); x = int(np.clip(x + dx, 0, 6)); value[y, x] = 1
        # Ensure a single connected, visible glyph.
        grown = value.copy()
        for yy, xx in zip(*np.where(value)):
            grown[max(0, yy - 1):min(7, yy + 2), max(0, xx - 1):min(7, xx + 2)] = 1
        rows.append(grown)
    return rows


MASKS = _mask_bank()
COLORS = np.asarray([(231, 76, 60), (52, 152, 219), (46, 204, 113), (241, 196, 15),
                     (155, 89, 182), (230, 126, 34), (26, 188, 156), (255, 102, 153),
                     (102, 204, 255), (153, 255, 102), (255, 204, 102), (204, 153, 255),
                     (240, 128, 128), (128, 220, 190), (190, 150, 240), (220, 210, 120)], dtype=np.uint8)


def _signature(patch: np.ndarray) -> tuple[str, str] | tuple[None, None]:
    mask = patch.max(0) > 35
    # White is the controllable body, not a workshop object.
    white = (patch.min(0) > 245)
    mask &= ~white
    if mask.sum() < 5: return None, None
    ys, xs = np.where(mask); crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    rgb = patch[:, mask].mean(1).round().astype(np.uint8)
    form = hashlib.sha256(crop.tobytes() + bytes(crop.shape)).hexdigest()[:12]
    identity = hashlib.sha256(crop.tobytes() + rgb.tobytes()).hexdigest()[:16]
    return identity, form


def _crop(frame: np.ndarray, row: int, col: int) -> np.ndarray:
    y, x = Y0 + row * CELL, X0 + col * CELL
    return frame[:, y + 1:y + 9, x + 1:x + 9]


def perceive_workshop(frame: np.ndarray) -> dict:
    """Read pixels into anonymous persistent observations; no private ids leak."""
    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.shape != (3, HEIGHT, WIDTH): raise ValueError(f"unexpected workshop frame {rgb.shape}")
    goal, _ = _signature(rgb[:, 3:13, 3:13])
    held, _ = _signature(rgb[:, 3:13, 91:101])
    context = hashlib.sha256(rgb[:, 4:12, 48:64].tobytes()).hexdigest()[:20]
    avatar = None; floor = set(); walls = set(); objects = []
    for row in range(GRID):
        for col in range(GRID):
            patch = _crop(rgb, row, col)
            if patch.max() == 0: continue
            center = patch[:, 2:6, 2:6]
            body_here = (center.min(0) > 245).sum() >= 4
            if body_here: avatar = (row, col)
            if tuple(int(x) for x in patch[:, 0, 0]) == (72, 78, 88): walls.add((row, col)); continue
            floor.add((row, col))
            # The body can partially occlude an object. Preserve the object's
            # earlier memory instead of hallucinating a new clipped identity.
            if body_here: continue
            signature, form = _signature(patch)
            if signature is not None: objects.append({"signature": signature, "form": form, "cell": (row, col)})
    if goal is None or avatar is None: raise RuntimeError("visual goal or controllable body not perceived")
    return {"context": context, "goal": goal, "held": held or EMPTY, "avatar": avatar,
            "floor": floor, "walls": walls, "objects": objects}


class PersistentWorkshop:
    """A hidden recipe world with fog, obstacles, remapped controls and hazards."""
    action_dim = 5

    def __init__(self, seed: int, *, goal: str = "final"):
        self.seed = int(seed); self.goal_name = goal; rng = np.random.default_rng(seed)
        self.chain_length = 2 + seed % 3
        # Causal roles have recurring morphology but fresh colors and identities.
        station_forms = rng.choice(np.arange(0, 7), 4, replace=False)
        material_forms = rng.choice(np.arange(7, 13), 2, replace=False)
        product_forms = rng.choice(np.arange(13, 28), self.chain_length + 2, replace=False)
        kinds = ["raw0", "raw1"] + [f"station{i}" for i in range(4)] + \
                [f"product{i}" for i in range(self.chain_length)] + ["side", "hazard"]
        forms = list(material_forms) + list(station_forms) + list(product_forms) + [int(rng.choice(np.arange(0, 7)))]
        colors = rng.choice(len(COLORS), len(kinds), replace=True)
        self.visual = {kind: (int(form), int(color)) for kind, form, color in zip(kinds, forms, colors)}
        station_order = rng.permutation(4).tolist()
        self.rules = {}; value = "raw0"
        for index in range(self.chain_length):
            output = f"product{index}"; self.rules[(value, f"station{station_order[index]}")] = output; value = output
        branch_station = station_order[-1]
        self.rules[("raw1", f"station{branch_station}")] = "side"
        self.goal_type = value if goal == "final" else "side"
        # Every world has a different anonymous permutation of primitive controls.
        permutation = rng.permutation(5).tolist()
        self.control = {permutation[index]: DELTAS[index] for index in range(4)}
        self.interact_action = permutation[4]
        self.start = (4, 4)
        candidates = [(r, c) for r in range(GRID) for c in range(GRID)
                      if abs(r - 4) > 2 or abs(c - 4) > 2]
        rng.shuffle(candidates)
        # Add walls only when connectivity is preserved; the central calibration
        # area is always open, but no map or coordinates are supplied to the mind.
        self.walls = set()
        for cell in candidates[:14]:
            trial = self.walls | {cell}
            if self._reachable_count(trial) == GRID * GRID - len(trial): self.walls = trial
        free = [cell for cell in candidates if cell not in self.walls]
        rng.shuffle(free)
        object_kinds = ["raw0", "raw1"] + [f"station{i}" for i in range(4)] + ["hazard"]
        self.placements = {kind: free[index] for index, kind in enumerate(object_kinds)}
        # Stable raw visual context cue; not a symbolic id exposed to the learner.
        self.emblem = rng.integers(40, 230, (3, 8, 16), dtype=np.uint8)
        self.episode = 0

    def _reachable_count(self, walls: set[tuple[int, int]]) -> int:
        seen = {self.start}; queue = deque([self.start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in DELTAS:
                nxt = r + dr, c + dc
                if 0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID and nxt not in walls and nxt not in seen:
                    seen.add(nxt); queue.append(nxt)
        return len(seen)

    def reset(self, episode_seed: int | None = None):
        self.position = self.start; self.held = None; self.done = False; self.steps = 0; self.episode += 1
        return self.render()

    def step(self, action: int):
        if self.done: raise RuntimeError("episode ended")
        action = int(action); event = "no-visible-effect"
        if action in self.control:
            dr, dc = self.control[action]; nxt = self.position[0] + dr, self.position[1] + dc
            if 0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID and nxt not in self.walls:
                self.position = nxt; event = "body-moved"
        elif action == self.interact_action:
            target = next((kind for kind, cell in self.placements.items() if cell == self.position), None)
            before = self.held
            if target in ("raw0", "raw1"): self.held = target
            elif target == "hazard": self.held = None
            elif target is not None and self.held is not None:
                self.held = self.rules.get((self.held, target), self.held)
            event = "interaction-changed-state" if self.held != before else "interaction-no-effect"
        else: raise ValueError(action)
        self.steps += 1; success = self.held == self.goal_type; self.done = success
        return self.render(), (1.0 if success else 0.0), self.done, {"success": success, "event_audit_only": event}

    def _paint(self, frame: np.ndarray, kind: str, cx: int, cy: int):
        form, color = self.visual[kind]; mask = MASKS[form]; rgb = COLORS[color]
        for channel in range(3):
            patch = frame[channel, cy - 3:cy + 4, cx - 3:cx + 4]; patch[mask > 0] = rgb[channel]

    def render(self):
        frame = np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8)
        self._paint(frame, self.goal_type, 8, 8)
        if self.held is not None: self._paint(frame, self.held, 96, 8)
        frame[:, 4:12, 48:64] = self.emblem
        frame[:, 22:24, 3:109] = 24
        visible = {(r, c) for r in range(GRID) for c in range(GRID)
                   if abs(r - self.position[0]) + abs(c - self.position[1]) <= 3}
        for row, col in visible:
            y, x = Y0 + row * CELL, X0 + col * CELL
            color = (72, 78, 88) if (row, col) in self.walls else (12, 18, 27)
            frame[:, y:y + 10, x:x + 10] = np.asarray(color, dtype=np.uint8)[:, None, None]
        for kind, (row, col) in self.placements.items():
            if (row, col) in visible:
                self._paint(frame, kind, X0 + col * CELL + 5, Y0 + row * CELL + 5)
        row, col = self.position; cy, cx = Y0 + row * CELL + 5, X0 + col * CELL + 5
        frame[:, cy - 2:cy + 3, cx] = 255; frame[:, cy, cx - 2:cx + 3] = 255
        return frame


class WorkshopMind:
    """Learns embodiment, object permanence, affordances and causal programs."""
    format = "wailah-persistent-workshop-mind-v1"

    def __init__(self):
        self.worlds = {}; self.form_affordances = defaultdict(Counter)
        self.experiments = 0; self.rules_discovered = 0; self.subgoals_created = 0
        self.predictions = 0; self.correct_predictions = 0; self.current = None; self.pending = None

    def begin(self, frame):
        view = perceive_workshop(frame); self.current = view["context"]
        self.worlds.setdefault(self.current, {"controls": {}, "tried_controls": [], "interact": None,
            "floor": [], "walls": [], "objects": {}, "rules": {}, "tested": [], "visits": 0})
        self.memory["visits"] += 1; self.pending = None; self._update(view); return self.current

    @property
    def memory(self): return self.worlds[self.current]

    def _update(self, view):
        self.memory["floor"] = sorted({tuple(x) for x in self.memory["floor"]} | view["floor"])
        self.memory["walls"] = sorted({tuple(x) for x in self.memory["walls"]} | view["walls"])
        for obj in view["objects"]:
            self.memory["objects"][obj["signature"]] = {"form": obj["form"], "cell": list(obj["cell"])}

    @staticmethod
    def _key(held, target): return f"{held}>{target}"

    def _controls_ready(self):
        return len(self.memory["controls"]) == 4 and self.memory["interact"] is not None

    def _route(self, start, target):
        floor = {tuple(x) for x in self.memory["floor"]}; queue = deque([(start, [])]); seen = {start}
        inverse = {tuple(delta): int(action) for action, delta in self.memory["controls"].items()}
        while queue:
            cell, actions = queue.popleft()
            if cell == target: return actions
            for delta, action in inverse.items():
                nxt = cell[0] + delta[0], cell[1] + delta[1]
                if nxt in floor and nxt not in seen:
                    seen.add(nxt); queue.append((nxt, actions + [action]))
        return None

    def _held_plan(self, current, goal):
        rules = {}
        for key, outcome in self.memory["rules"].items():
            if outcome not in (EMPTY, "<same>"):
                held, target = key.split(">", 1); rules[(held, target)] = outcome
        queue = deque([(current, [])]); seen = {current}
        while queue:
            state, plan = queue.popleft()
            if state == goal: return plan
            for (held, target), outcome in sorted(rules.items()):
                if held == state and outcome not in seen:
                    seen.add(outcome); queue.append((outcome, plan + [target]))
        return None

    def _frontier_target(self, avatar):
        floor = {tuple(x) for x in self.memory["floor"]}; known = floor | {tuple(x) for x in self.memory["walls"]}
        candidates = []
        for cell in floor:
            unknown = sum(0 <= cell[0] + dr < GRID and 0 <= cell[1] + dc < GRID and
                          (cell[0] + dr, cell[1] + dc) not in known for dr, dc in DELTAS)
            route = self._route(avatar, cell)
            if unknown and route: candidates.append((len(route), cell, route))
        return min(candidates, default=None, key=lambda row: (row[0], row[1]))

    def _reachable_experiment(self, held):
        objects = self.memory["objects"]; tested = set(self.memory["tested"])
        candidates = []
        for signature, row in objects.items():
            key = self._key(held, signature)
            if key in tested: continue
            prior = self.form_affordances[row["form"]]
            novelty = (prior["novel_effect"] + 1) / (prior["tests"] + 2)
            candidates.append((novelty, signature))
        if candidates: return [], max(candidates, key=lambda x: (x[0], x[1]))[1]
        # Find a known sequence that reaches a state with an untested experiment.
        rules = [(key.split(">", 1), outcome) for key, outcome in self.memory["rules"].items()
                 if outcome not in (EMPTY, "<same>")]
        queue = deque([(held, [])]); seen = {held}
        while queue:
            state, plan = queue.popleft()
            for signature in objects:
                if self._key(state, signature) not in tested: return plan, signature
            for (source, target), outcome in rules:
                if source == state and outcome not in seen:
                    seen.add(outcome); queue.append((outcome, plan + [target]))
        return None

    def act(self, frame):
        view = perceive_workshop(frame); self._update(view)
        if not self._controls_ready():
            tried = set(self.memory["tried_controls"])
            action = next(action for action in range(5) if action not in tried)
            self.memory["tried_controls"].append(action)
            self.pending = {"kind": "calibrate", "before": view, "action": action}
            return action, {"reason": "self-calibrate-unknown-control", "prediction": "unknown"}
        plan = self._held_plan(view["held"], view["goal"])
        target_signature = plan[0] if plan else None
        reason = "execute-discovered-causal-plan"
        if target_signature is None:
            frontier = self._frontier_target(view["avatar"])
            if frontier is not None:
                action = frontier[2][0]; self.pending = {"kind": "move", "before": view, "action": action}
                self.subgoals_created += 1
                return action, {"reason": "self-generated-exploration-subgoal",
                                "prediction": "reveal-unseen-space", "target_cell": frontier[1]}
            experiment = self._reachable_experiment(view["held"])
            if experiment is not None:
                causal_prefix, target_signature = experiment
                if causal_prefix:
                    target_signature = causal_prefix[0]; reason = "reach-experimentable-causal-state"
                else:
                    reason = "self-generated-interaction-experiment"; self.subgoals_created += 1
            else:
                target_signature = sorted(self.memory["objects"])[0]; reason = "repeat-low-evidence-experiment"
        target_cell = tuple(self.memory["objects"][target_signature]["cell"])
        route = self._route(view["avatar"], target_cell)
        if route:
            action = route[0]; self.pending = {"kind": "move", "before": view, "action": action}
            return action, {"reason": "navigate-to-subgoal:" + reason, "prediction": "body-moves-toward-object"}
        action = int(self.memory["interact"]); key = self._key(view["held"], target_signature)
        prediction = self.memory["rules"].get(key, "unknown")
        self.pending = {"kind": "interact", "before": view, "action": action,
                        "target": target_signature, "key": key, "prediction": prediction}
        return action, {"reason": reason, "prediction": prediction,
                        "hypothesis": f"interact({view['held']},{target_signature})"}

    def observe(self, before, action, after, reward, done):
        later = perceive_workshop(after); self._update(later); pending = self.pending or {}
        prior = pending.get("before", perceive_workshop(before)); kind = pending.get("kind")
        if kind == "calibrate":
            dr = later["avatar"][0] - prior["avatar"][0]; dc = later["avatar"][1] - prior["avatar"][1]
            if (dr, dc) in DELTAS: self.memory["controls"][str(int(action))] = [dr, dc]
            if len(self.memory["tried_controls"]) == 5:
                moving = {int(x) for x in self.memory["controls"]}
                remaining = sorted(set(range(5)) - moving)
                if len(remaining) == 1: self.memory["interact"] = remaining[0]
        elif kind == "interact":
            key, target = pending["key"], pending["target"]
            outcome = later["held"] if later["held"] != prior["held"] else "<same>"
            if key not in self.memory["tested"]: self.memory["tested"].append(key)
            if key not in self.memory["rules"] and outcome != "<same>": self.rules_discovered += 1
            self.memory["rules"][key] = outcome; self.experiments += 1
            form = self.memory["objects"][target]["form"]; self.form_affordances[form]["tests"] += 1
            if outcome not in ("<same>", EMPTY, target): self.form_affordances[form]["novel_effect"] += 1
            if pending["prediction"] != "unknown":
                self.predictions += 1; self.correct_predictions += int(pending["prediction"] == outcome)
        self.pending = None
        return {"reward": float(reward), "done": bool(done), "held": later["held"]}

    def status(self):
        invented = Counter()
        for rows in self.form_affordances.values():
            ratio = rows["novel_effect"] / max(1, rows["tests"])
            invented["high-transform" if ratio >= .3 else "low-transform"] += 1
        return {"format": self.format, "remembered_worlds": len(self.worlds),
                "causal_rules_discovered": self.rules_discovered,
                "self_generated_subgoals": self.subgoals_created, "interaction_experiments": self.experiments,
                "invented_affordance_groups": dict(invented),
                "prediction_accuracy": self.correct_predictions / max(1, self.predictions)}

    def export(self):
        return self.status() | {"worlds": self.worlds,
            "form_affordances": {key: dict(rows) for key, rows in self.form_affordances.items()},
            "experiments": self.experiments, "rules_discovered": self.rules_discovered,
            "subgoals_created": self.subgoals_created, "predictions": self.predictions,
            "correct_predictions": self.correct_predictions}

    def save(self, path: Path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.export(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format") != cls.format: raise ValueError("unsupported workshop mind")
        result = cls(); result.worlds = value["worlds"]
        result.form_affordances = defaultdict(Counter, {key: Counter(rows)
            for key, rows in value.get("form_affordances", {}).items()})
        result.experiments = int(value.get("experiments", 0)); result.rules_discovered = int(value.get("rules_discovered", 0))
        result.subgoals_created = int(value.get("subgoals_created", 0)); result.predictions = int(value.get("predictions", 0))
        result.correct_predictions = int(value.get("correct_predictions", 0)); return result


def _run(mind: WorkshopMind, world: PersistentWorkshop, budget: int, episode_horizon: int = 90):
    used = episodes = 0; success = False; explanations = []
    while used < budget and not success:
        frame = world.reset(); mind.begin(frame); episodes += 1
        for _ in range(min(episode_horizon, budget - used)):
            action, evidence = mind.act(frame); nxt, reward, done, _ = world.step(action)
            mind.observe(frame, action, nxt, reward, done); used += 1
            if evidence["reason"] in ("self-generated-interaction-experiment", "execute-discovered-causal-plan"):
                explanations.append({"step": used, **evidence})
            frame = nxt
            if done: success = True; break
    memory = mind.memory
    return {"success": success, "interactions": used, "episodes": episodes,
            "controls_grounded": len(memory["controls"]) == 4 and memory["interact"] is not None,
            "remembered_objects": len(memory["objects"]), "rules": len(memory["rules"]),
            "explanations": explanations[-20:]}


def _random(world: PersistentWorkshop, budget: int, seed: int, episode_horizon: int = 90):
    rng = np.random.default_rng(seed); used = episodes = 0; success = False
    while used < budget and not success:
        frame = world.reset(); episodes += 1
        for _ in range(min(episode_horizon, budget - used)):
            frame, reward, done, _ = world.step(int(rng.integers(5))); used += 1
            if done: success = True; break
    return {"success": success, "interactions": used, "episodes": episodes}


def _summary(rows):
    return {"worlds": len(rows), "successes": sum(row["success"] for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_interactions": float(np.mean([row["interactions"] for row in rows]))}


def run_workshop_benchmark(output: Path, *, worlds: int = 12, budget: int = 900, seed: int = 140_021):
    mind = WorkshopMind(); full = []; local = []; random_rows = []; erased = []
    for index in range(worlds):
        world_seed = seed + index * 3571
        full.append(_run(mind, PersistentWorkshop(world_seed), budget))
        local.append(_run(WorkshopMind(), PersistentWorkshop(world_seed), budget))
        random_rows.append(_random(PersistentWorkshop(world_seed), budget, seed + index * 71))
        # Equal total budget, but no memory is allowed to cross a reset.
        remaining = budget; solved = False; spent = attempts = 0
        while remaining and not solved:
            row = _run(WorkshopMind(), PersistentWorkshop(world_seed), min(90, remaining))
            spent += row["interactions"]; remaining -= row["interactions"]; attempts += 1; solved = row["success"]
        erased.append({"success": solved, "interactions": spent, "episodes": attempts})
    revisits = [_run(mind, PersistentWorkshop(seed + i * 3571, goal="branch"), 180)
                for i in range(worlds)]
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    brain = output.with_name("PERSISTENT_WORKSHOP_MIND.json"); mind.save(brain)
    restored = WorkshopMind.load(brain)
    reload_rows = [_run(restored, PersistentWorkshop(seed + i * 3571, goal="branch"), 80)
                   for i in range(min(6, worlds))]
    report = {"format": "wailah-persistent-workshop-audit-v1",
        "protocol": {"seed": seed, "worlds": worlds, "budget": budget, "episode_horizon": 90,
            "inputs": "raw RGB pixels, five anonymous primitive controls, reward, done",
            "partial_observability": "Manhattan-radius-three fog of war; objects disappear from pixels",
            "agent_not_given": ["movement mapping", "interaction control", "map", "object ids", "object classes",
                                "recipes", "chain length", "machine identities", "subgoals", "solutions"],
            "engineered_priors": ["tiled spatial parser", "persistent memory", "generic graph search",
                                  "interaction hypothesis format"]},
        "aggregate": {"growing_workshop_mind": _summary(full),
                      "local_minds_without_cross_world_affordances": _summary(local),
                      "random_primitive_controls": _summary(random_rows),
                      "memory_erased_each_episode": _summary(erased),
                      "old_worlds_changed_goal": _summary(revisits),
                      "reloaded_memory": _summary(reload_rows)},
        "growth": mind.status(), "per_world": full,
        "learning_to_learn": {"first_quartile_mean": float(np.mean([x["interactions"] for x in full[:max(1, worlds // 4)]])),
            "last_quartile_mean": float(np.mean([x["interactions"] for x in full[-max(1, worlds // 4):]])),
            "local_only_mean": float(np.mean([x["interactions"] for x in local])),
            "matched_mean_interactions_saved": float(np.mean(
                [baseline["interactions"] - grown["interactions"] for grown, baseline in zip(full, local)])),
            "cross_world_speedup_vs_local_only": float(
                np.mean([x["interactions"] for x in local]) / np.mean([x["interactions"] for x in full])),
            "matched_worlds_faster": sum(grown["interactions"] < baseline["interactions"]
                                         for grown, baseline in zip(full, local)),
            "note": "first/last quarters contain different generated chain lengths; matched local controls are the causal transfer comparison"},
        "persistent_brain": str(brain), "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8"); return report


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=12); parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--seed", type=int, default=140_021); args = parser.parse_args(argv)
    report = run_workshop_benchmark(args.output, worlds=args.worlds, budget=args.budget, seed=args.seed)
    print(json.dumps(report["aggregate"] | {"growth": report["growth"], "learning": report["learning_to_learn"]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
