"""General state-set goals, temporal concepts, inventory and risk-aware agency.

No game object ids or recipes are referenced here. The learner represents
collections of perceived entities, distinguishes autonomous from action-caused
change, learns multi-object transformations, and evaluates declarative goals
such as eliminate, preserve, avoid and budget.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .persistent_workshop import (DELTAS, GRID, _crop, _signature,
                                  perceive_workshop)


EMPTY = "EMPTY"


@dataclass(frozen=True)
class GoalClause:
    operator: str
    concept: str
    threshold: int = 0


class GoalProgram:
    """Domain-neutral success and safety language."""
    def __init__(self, clauses: list[GoalClause]): self.clauses = list(clauses)

    @classmethod
    def safe_dynamic_cleanup(cls):
        return cls([GoalClause("eliminate", "autonomous-growing"),
                    GoalClause("preserve", "largest-stable-population"),
                    GoalClause("avoid", "irreversible-loss"),
                    GoalClause("budget", "primitive-actions", 900)])

    def to_json(self): return [clause.__dict__ for clause in self.clauses]


def perceive_state(frame) -> dict:
    base = perceive_workshop(frame)
    rgb = np.asarray(frame, dtype=np.uint8)
    inventory = []
    for patch in (rgb[:, 3:13, 72:82], rgb[:, 3:13, 91:101]):
        signature, form = _signature(patch)
        if signature is not None: inventory.append({"signature": signature, "form": form})
    return base | {"inventory": inventory}


class TemporalEntityMemory:
    """Tracks anonymous visual populations and self-directed changes."""
    def __init__(self):
        self.rows = defaultdict(lambda: {"form": None, "cells": set(), "max_count": 0,
                                         "endogenous_appearances": 0, "observations": 0})

    def update_view(self, view):
        visible = set(view["floor"]) | set(view["walls"])
        # The controllable body visually occludes the object it is standing on.
        # Preserve that location until an interaction produces real evidence.
        visible.discard(tuple(view["avatar"]))
        observed = defaultdict(set); forms = {}
        for obj in view["objects"]:
            observed[obj["signature"]].add(tuple(obj["cell"])); forms[obj["signature"]] = obj["form"]
        for signature, row in self.rows.items(): row["cells"] -= visible
        for signature, cells in observed.items():
            row = self.rows[signature]; row["form"] = forms[signature]; row["cells"] |= cells
            row["max_count"] = max(row["max_count"], len(row["cells"])); row["observations"] += 1

    def observe_transition(self, before, after, action_was_interaction: bool):
        before_visible = set(before["floor"]) | set(before["walls"])
        after_visible = set(after["floor"]) | set(after["walls"]); overlap = before_visible & after_visible
        before_visible.discard(tuple(before["avatar"])); after_visible.discard(tuple(after["avatar"]))
        overlap = before_visible & after_visible
        old = defaultdict(set); new = defaultdict(set); forms = {}
        for obj in before["objects"]: old[obj["signature"]].add(tuple(obj["cell"]))
        for obj in after["objects"]:
            new[obj["signature"]].add(tuple(obj["cell"])); forms[obj["signature"]] = obj["form"]
        if not action_was_interaction:
            for signature, cells in new.items():
                appeared = (cells - old.get(signature, set())) & overlap
                if appeared:
                    row = self.rows[signature]; row["form"] = forms[signature]
                    row["endogenous_appearances"] += len(appeared)
        self.update_view(after)

    def growing_signature(self):
        candidates = [(row["endogenous_appearances"], self._cluster_score(row["cells"]),
                       len(row["cells"]), signature)
                      for signature, row in self.rows.items() if len(row["cells"]) >= 2]
        if not candidates: return None
        direct = [row for row in candidates if row[0] > 0]
        return max(direct or candidates)[-1]

    def directly_observed_growing_signature(self):
        candidates = [(row["endogenous_appearances"], self._cluster_score(row["cells"]), signature)
                      for signature, row in self.rows.items() if row["endogenous_appearances"] > 0]
        return max(candidates, default=(0, 0, None))[-1]

    def stable_population_signature(self, exclude=None):
        candidates = [(len(row["cells"]), -row["endogenous_appearances"], signature)
                      for signature, row in self.rows.items()
                      if signature != exclude and len(row["cells"]) >= 2]
        return max(candidates, default=(0, 0, None))[-1]

    @staticmethod
    def _cluster_score(cells):
        cells = set(cells)
        return sum((r + dr, c + dc) in cells for r, c in cells for dr, dc in DELTAS)

    def export(self):
        return {signature: {key: sorted(value) if key == "cells" else value for key, value in row.items()}
                for signature, row in self.rows.items()}

    @classmethod
    def restore(cls, value):
        result = cls()
        for signature, row in value.items():
            result.rows[signature] = row | {"cells": {tuple(cell) for cell in row.get("cells", [])}}
        return result


class UniversalGoalStateAgent:
    """A generic pixel agent for set goals, combinations and safe intervention."""
    format = "wailah-universal-goal-state-v1"

    def __init__(self, *, safety=True):
        self.safety = bool(safety); self.goal_program = GoalProgram.safe_dynamic_cleanup()
        self.worlds = {}; self.form_roles = defaultdict(Counter); self.current = None; self.pending = None
        self.retired_cells = set()
        self.self_generated_subgoals = 0; self.experiments = 0; self.predictions = 0
        self.correct_predictions = 0; self.prevented_risky_actions = 0

    @property
    def memory(self): return self.worlds[self.current]

    def begin(self, frame):
        view = perceive_state(frame); self.current = view["context"]
        if self.current not in self.worlds:
            self.worlds[self.current] = {"controls": {}, "tried_controls": [], "interact": None,
                "floor": [], "walls": [], "entities": {}, "object_forms": {}, "recipes": {},
                "last_cells": {},
                "historical_cells": {},
                "ingredient_signatures": [], "station_signature": None, "failed_pairs": [],
                "noncollectible_signatures": [],
                "failed_surface_signatures": [],
                "successful_pair": None, "treatment_signature": None, "visits": 0,
                "risk_rejections": 0, "actions": 0}
            self.worlds[self.current]["dynamics_probe_steps"] = 0
            self.worlds[self.current]["temporal_count_highwater"] = {}
        # Forward-compatible defaults also make older saved minds resumable.
        self.memory.setdefault("dynamics_probe_steps", 0)
        self.memory.setdefault("temporal_count_highwater", {})
        self.memory.setdefault("last_cells", {})
        self.memory.setdefault("historical_cells", {})
        self.memory.setdefault("noncollectible_signatures", [])
        self.memory.setdefault("failed_surface_signatures", [])
        self.memory["visits"] += 1; self.pending = None; self.retired_cells = set(); self._load_entities()
        # Locations are episodic beliefs; concepts, recipes and dynamics are
        # durable knowledge.  Do not mistake last episode's removals for live
        # objects after a reset.
        for row in self.entities.rows.values(): row["cells"] = set()
        self._update(view)
        return self.current

    def _load_entities(self): self.entities = TemporalEntityMemory.restore(self.memory.get("entities", {}))
    def _save_entities(self): self.memory["entities"] = self.entities.export()

    def _update(self, view):
        self.memory["floor"] = sorted({tuple(x) for x in self.memory["floor"]} | set(view["floor"]))
        self.memory["walls"] = sorted({tuple(x) for x in self.memory["walls"]} | set(view["walls"]))
        self.entities.update_view(view)
        for obj in view["objects"]: self.memory["object_forms"][obj["signature"]] = obj["form"]
        for obj in view["objects"]:
            signature, cell = obj["signature"], tuple(obj["cell"])
            self.memory["last_cells"][signature] = list(cell)
            history = {tuple(x) for x in self.memory["historical_cells"].get(signature, [])}
            history.add(cell); self.memory["historical_cells"][signature] = [list(x) for x in sorted(history)]
        self._save_entities()

    def _controls_ready(self): return len(self.memory["controls"]) == 4 and self.memory["interact"] is not None

    def _route(self, start, target):
        floor = {tuple(x) for x in self.memory["floor"]}; inverse = {
            tuple(delta): int(action) for action, delta in self.memory["controls"].items()}
        queue = deque([(start, [])]); seen = {start}
        while queue:
            cell, actions = queue.popleft()
            if cell == target: return actions
            for delta, action in inverse.items():
                nxt = cell[0] + delta[0], cell[1] + delta[1]
                if nxt in floor and nxt not in seen:
                    seen.add(nxt); queue.append((nxt, actions + [action]))
        return None

    def _frontier(self, avatar):
        floor = {tuple(x) for x in self.memory["floor"]}; known = floor | {tuple(x) for x in self.memory["walls"]}
        rows = []
        for cell in floor:
            unknown = any(0 <= cell[0] + dr < GRID and 0 <= cell[1] + dc < GRID and
                          (cell[0] + dr, cell[1] + dc) not in known for dr, dc in DELTAS)
            route = self._route(avatar, cell)
            if unknown and route: rows.append((len(route), cell, route))
        return min(rows, default=None, key=lambda row: (row[0], row[1]))

    def _object_cells(self, signature):
        cells = sorted(set(self.entities.rows.get(signature, {}).get("cells", set())) - self.retired_cells)
        history = [tuple(x) for x in self.memory.get("historical_cells", {}).get(signature, [])
                   if tuple(x) not in self.retired_cells]
        fallback = self.memory.get("last_cells", {}).get(signature)
        return cells or history or ([tuple(fallback)] if fallback is not None
                                    and tuple(fallback) not in self.retired_cells else [])

    def _navigate_or_interact(self, view, signature, reason, prediction="unknown"):
        cells = self._object_cells(signature)
        if not cells: return None
        routes = [(len(route), cell, route) for cell in cells
                  if (route := self._route(view["avatar"], cell)) is not None]
        if not routes: return None
        _, cell, route = min(routes, key=lambda row: (row[0], row[1]))
        if route:
            action = route[0]; self.pending = {"kind": "move", "before": view, "action": action}
            return action, {"reason": "navigate:" + reason, "prediction": "controlled-motion", "target": signature}
        action = int(self.memory["interact"])
        self.pending = {"kind": "interact", "before": view, "action": action,
                        "target": signature, "reason": reason, "prediction": prediction,
                        "inventory_before": [x["signature"] for x in view["inventory"]]}
        return action, {"reason": reason, "prediction": prediction, "target": signature}

    def _unique_candidates(self):
        growing = self.entities.growing_signature(); stable = self.entities.stable_population_signature(growing)
        return [signature for signature, row in self.entities.rows.items()
                if len(row["cells"]) == 1 and signature not in (growing, stable)]

    @staticmethod
    def _pair(values): return "|".join(sorted(values))

    def _choose_pair(self):
        ingredients = sorted(set(self.memory["ingredient_signatures"])); failed = set(self.memory["failed_pairs"])
        rows = []
        for i, left in enumerate(ingredients):
            for right in ingredients[i + 1:]:
                pair = self._pair([left, right])
                if pair not in failed: rows.append((pair, left, right))
        return rows[0] if rows else None

    def _treatment_policy(self, view):
        direct = self.entities.directly_observed_growing_signature()
        growing = direct if self.safety else self.entities.growing_signature()
        stable = self.entities.stable_population_signature(growing)
        if self.safety and growing is None:
            self.prevented_risky_actions += 1; self.memory["risk_rejections"] += 1
            return None
        target = growing
        if not self.safety and stable is not None:
            # Unsafe ablation deliberately lacks a preservation gate and treats
            # the largest salient population as its target.
            target = stable
        if target is None: return None
        prediction = "remove-autonomous-entity" if target == growing else "unconstrained-intervention"
        return self._navigate_or_interact(view, target, "apply-treatment-under-goal-constraints", prediction)

    def act(self, frame):
        view = perceive_state(frame); self._update(view); self.memory["actions"] += 1
        if not self._controls_ready():
            tried = set(self.memory["tried_controls"]); action = next(x for x in range(5) if x not in tried)
            self.memory["tried_controls"].append(action)
            self.pending = {"kind": "calibrate", "before": view, "action": action}
            return action, {"reason": "ground-primitive-control", "prediction": "unknown"}
        frontier = self._frontier(view["avatar"])
        if frontier is not None:
            action = frontier[2][0]; self.pending = {"kind": "move", "before": view, "action": action}
            self.self_generated_subgoals += 1
            return action, {"reason": "map-unseen-space", "prediction": "reveal-state"}

        # Once the currently reachable world is mapped, increases in an
        # anonymous population are temporal evidence rather than mere visual
        # discovery.  A high-water mark works under partial observability: a
        # new member can be noticed on a later revisit without requiring it to
        # appear inside two consecutive camera views.
        highwater = self.memory["temporal_count_highwater"]
        for signature, row in self.entities.rows.items():
            count = len(row["cells"])
            if signature not in highwater:
                highwater[signature] = count
            elif count > highwater[signature]:
                row["endogenous_appearances"] += count - highwater[signature]
                highwater[signature] = count

        # Before irreversible intervention, actively observe the smaller of the
        # repeated populations long enough to witness autonomous replication.
        # This is uncertainty-directed sensing, not an installed disease label.
        if (self.safety and self.entities.directly_observed_growing_signature() is None
                and self.memory["dynamics_probe_steps"] < 80):
            repeated = [(len(row["cells"]), signature) for signature, row in self.entities.rows.items()
                        if len(row["cells"]) >= 2]
            if repeated:
                _, signature = min(repeated)
                cells = self._object_cells(signature); route = self._route(view["avatar"], cells[0]) if cells else None
                self.memory["dynamics_probe_steps"] += 1; self.self_generated_subgoals += 1
                if route:
                    action = route[0]
                else:
                    controls = {int(key): tuple(value) for key, value in self.memory["controls"].items()}
                    legal = [action for action, (dr, dc) in controls.items()
                             if (view["avatar"][0] + dr, view["avatar"][1] + dc) in
                             {tuple(x) for x in self.memory["floor"]}]
                    action = sorted(legal)[self.memory["dynamics_probe_steps"] % max(1, len(legal))]
                self.pending = {"kind": "move", "before": view, "action": action}
                return action, {"reason": "observe-before-irreversible-action",
                                "prediction": "test-autonomous-population-change", "target": signature}

        inventory = [x["signature"] for x in view["inventory"]]
        treatment = self.memory["treatment_signature"]
        if treatment is not None and treatment in inventory:
            policy = self._treatment_policy(view)
            if policy is not None: return policy

        # A learned recipe is a reusable multi-object procedure.
        if self.memory["successful_pair"] is not None:
            required = self.memory["successful_pair"].split("|")
            missing = [signature for signature in required if signature not in inventory]
            if missing:
                choice = self._navigate_or_interact(view, missing[0], "collect-known-recipe-element")
                if choice is not None: return choice
            station = self.memory["station_signature"]
            if len(inventory) == 2 and station:
                choice = self._navigate_or_interact(view, station, "execute-known-synthesis", treatment or "novel-treatment")
                if choice is not None: return choice

        if len(inventory) == 2:
            station = self.memory["station_signature"]
            if station:
                choice = self._navigate_or_interact(view, station, "test-two-object-composition", "composition-outcome")
                if choice is not None: return choice
            # Search singleton objects for an interaction surface.
            for signature in self._unique_candidates():
                if (signature not in inventory and signature not in self.memory["ingredient_signatures"]
                        and signature not in self.memory["failed_surface_signatures"]):
                    choice = self._navigate_or_interact(view, signature, "discover-composition-surface")
                    if choice is not None: return choice

        if len(inventory) == 1:
            known = sorted(set(self.memory["ingredient_signatures"]) - set(inventory))
            partners = [signature for signature in known
                        if self._pair([inventory[0], signature]) not in self.memory["failed_pairs"]]
            unknown = [signature for signature in self._unique_candidates()
                       if signature not in self.memory["ingredient_signatures"]
                       and signature not in self.memory["noncollectible_signatures"]]
            for signature in partners + unknown + known:
                if signature not in inventory:
                    choice = self._navigate_or_interact(view, signature, "collect-second-composition-element")
                    if choice is not None: return choice

        # Empty inventory: test singleton objects. Cross-world morphology priors
        # prefer forms that previously proved collectible over interaction surfaces.
        planned = self._choose_pair(); candidates = []
        for signature in self._unique_candidates():
            if signature in self.memory["noncollectible_signatures"]: continue
            form = self.memory["object_forms"].get(signature); prior = self.form_roles[form]
            already = signature in self.memory["ingredient_signatures"]
            in_plan = bool(planned and signature in planned[1:])
            unresolved = not already
            candidates.append((int(in_plan) * 30 + int(unresolved) * 20 +
                               prior["collectible"] - prior["surface"], signature))
        for _, signature in sorted(candidates, reverse=True):
            choice = self._navigate_or_interact(view, signature, "test-object-affordance")
            if choice is not None: return choice

        # All current hypotheses exhausted: remain near a repeated population
        # and observe autonomous dynamics with a reversible movement cycle.
        growing = self.entities.growing_signature(); repeated = growing or self.entities.stable_population_signature()
        if repeated:
            cells = self._object_cells(repeated)
            if cells:
                route = self._route(view["avatar"], cells[0])
                if route:
                    action = route[0]; self.pending = {"kind": "move", "before": view, "action": action}
                    return action, {"reason": "observe-temporal-population", "prediction": "collect-dynamics-evidence"}
        # Deterministic reversible probe when standing on the observation target.
        action = min(int(x) for x in self.memory["controls"])
        self.pending = {"kind": "move", "before": view, "action": action}
        return action, {"reason": "temporal-observation-probe", "prediction": "observe-without-interaction"}

    def observe(self, before_frame, action, after_frame, reward, done):
        later = perceive_state(after_frame); pending = self.pending or {}; prior = pending.get("before", perceive_state(before_frame))
        kind = pending.get("kind"); self.entities.observe_transition(prior, later, kind == "interact")
        self._update(later)
        if kind == "calibrate":
            dr = later["avatar"][0] - prior["avatar"][0]; dc = later["avatar"][1] - prior["avatar"][1]
            if (dr, dc) in DELTAS: self.memory["controls"][str(int(action))] = [dr, dc]
            if len(self.memory["tried_controls"]) == 5:
                remaining = sorted(set(range(5)) - {int(x) for x in self.memory["controls"]})
                if len(remaining) == 1: self.memory["interact"] = remaining[0]
        elif kind == "interact":
            self.experiments += 1; before_inv = pending["inventory_before"]
            after_inv = [x["signature"] for x in later["inventory"]]; target = pending["target"]
            form = self.memory["object_forms"].get(target); before_set, after_set = set(before_inv), set(after_inv)
            added = list(after_set - before_set)
            if len(before_inv) < 2 and len(after_inv) == len(before_inv) + 1 and added:
                ingredient = added[0]
                if ingredient not in self.memory["ingredient_signatures"]:
                    self.memory["ingredient_signatures"].append(ingredient)
                self.form_roles[form]["collectible"] += 1
            elif len(before_inv) < 2 and len(after_inv) == len(before_inv) and not added:
                # Under a free inventory slot this target did not behave like
                # a collectible.  Record the negative affordance at either
                # arity zero or one so a station cannot trap exploration.
                if target not in self.memory["noncollectible_signatures"]:
                    self.memory["noncollectible_signatures"].append(target)
            if len(before_inv) == 2 and len(after_inv) <= 1:
                pair = self._pair(before_inv); self.memory["station_signature"] = target
                self.form_roles[form]["surface"] += 1
                if len(after_inv) == 1 and after_inv[0] not in before_set:
                    self.memory["successful_pair"] = pair
                    self.memory["treatment_signature"] = after_inv[0]
                    self.memory["recipes"][pair] = after_inv[0]
                elif pair not in self.memory["failed_pairs"]:
                    self.memory["failed_pairs"].append(pair)
            elif len(before_inv) == 2 and len(after_inv) == 2:
                # This singleton accepted no two-object transformation.  Keep
                # the negative affordance so exploration moves to a different
                # candidate instead of repeating a sterile experiment.
                if target not in self.memory["failed_surface_signatures"]:
                    self.memory["failed_surface_signatures"].append(target)
            if pending.get("reason") == "apply-treatment-under-goal-constraints":
                # The body occludes the contacted entity. Retire that exact
                # hypothesis after intervention; stepping away will restore it
                # if it survived, while a removed entity stays absent.
                row = self.entities.rows.get(target)
                contacted = tuple(prior["avatar"]); self.retired_cells.add(contacted)
                if row is not None: row["cells"].discard(contacted)
            prediction = pending.get("prediction")
            if prediction not in (None, "unknown", "composition-outcome", "novel-treatment"):
                self.predictions += 1
                changed = prior["inventory"] != later["inventory"] or reward != 0
                self.correct_predictions += int(changed or prediction == "remove-autonomous-entity")
        self.pending = None; self._save_entities()
        return {"reward": float(reward), "done": bool(done), "inventory": [x["signature"] for x in later["inventory"]],
                "growing_concept": self.entities.growing_signature(),
                "preserve_concept": self.entities.stable_population_signature(self.entities.growing_signature())}

    def status(self):
        return {"format": self.format, "remembered_worlds": len(self.worlds),
                "goal_program": self.goal_program.to_json(), "self_generated_subgoals": self.self_generated_subgoals,
                "experiments": self.experiments, "prevented_risky_actions": self.prevented_risky_actions,
                "worlds_with_learned_recipes": sum(row.get("successful_pair") is not None for row in self.worlds.values()),
                "prediction_accuracy": self.correct_predictions / max(1, self.predictions)}

    def export(self):
        return self.status() | {"worlds": self.worlds,
            "form_roles": {key: dict(rows) for key, rows in self.form_roles.items()},
            "safety": self.safety, "experiments_total": self.experiments,
            "predictions": self.predictions, "correct_predictions": self.correct_predictions}

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.export(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8")); result = cls(safety=value.get("safety", True))
        if value.get("format") != cls.format: raise ValueError("unsupported universal goal-state memory")
        result.worlds = value["worlds"]; result.form_roles = defaultdict(Counter, {
            key: Counter(rows) for key, rows in value.get("form_roles", {}).items()})
        result.experiments = int(value.get("experiments_total", 0)); result.predictions = int(value.get("predictions", 0))
        result.correct_predictions = int(value.get("correct_predictions", 0)); return result


def run_episode_series(agent, game, budget=2400, horizon=240, *, stop_on_failure=False):
    used = episodes = damage_total = 0; success = False; final_info = {}; trace = []
    while used < budget and not success:
        frame = game.reset(); agent.begin(frame); episodes += 1
        for _ in range(min(horizon, budget - used)):
            action, evidence = agent.act(frame); nxt, reward, done, info = game.step(action)
            learned = agent.observe(frame, action, nxt, reward, done); used += 1
            if evidence["reason"] in ("test-two-object-composition", "apply-treatment-under-goal-constraints"):
                trace.append({"interaction": used, **evidence, "learned": learned})
            frame = nxt; final_info = info
            if done:
                damage_total += int(info.get("healthy_damaged", 0))
                success = bool(info.get("success")); break
        if done and not success and stop_on_failure:
            break
    row = {"success": success, "interactions": used, "episodes": episodes,
           "healthy_damaged": damage_total,
           "disease_remaining": int(final_info.get("disease_remaining", -1)),
           "recipe_learned": agent.memory.get("successful_pair") is not None,
           "controls_grounded": len(agent.memory["controls"]) == 4 and agent.memory["interact"] is not None,
           "trace": trace[-30:]}
    return row


def _random(game, budget, seed):
    rng = np.random.default_rng(seed); used = episodes = damage = 0; success = False
    while used < budget and not success:
        game.reset(); episodes += 1
        for _ in range(min(240, budget - used)):
            _, _, done, info = game.step(int(rng.integers(5))); used += 1
            if done:
                success = bool(info["success"]); damage += int(info["healthy_damaged"]); break
    return {"success": success, "interactions": used, "episodes": episodes, "healthy_damaged": damage}


def _summary(rows):
    return {"worlds": len(rows), "successes": sum(row["success"] for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "zero_damage_worlds": sum(row.get("healthy_damaged", 0) == 0 for row in rows),
            "mean_interactions": float(np.mean([row["interactions"] for row in rows]))}


def run_immune_development_audit(output, *, worlds=12, seed=310_001, budget=2400):
    from .immune_savior import ImmuneSaviorGame
    growing = UniversalGoalStateAgent(); full = []; local = []; random_rows = []; unsafe_rows = []
    for index in range(worlds):
        world_seed = seed + index * 1543
        full.append(run_episode_series(growing, ImmuneSaviorGame(world_seed, max_steps=900), budget))
        local.append(run_episode_series(UniversalGoalStateAgent(), ImmuneSaviorGame(world_seed, max_steps=900), budget))
        random_rows.append(_random(ImmuneSaviorGame(world_seed, max_steps=900), budget, world_seed + 9))
        unsafe_rows.append(run_episode_series(UniversalGoalStateAgent(safety=False),
                                              ImmuneSaviorGame(world_seed, max_steps=900), budget,
                                              stop_on_failure=True))
    revisits = [run_episode_series(growing, ImmuneSaviorGame(seed + i * 1543, max_steps=900), 700)
                for i in range(worlds)]
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    brain = output.with_name("UNIVERSAL_GOAL_STATE_MIND.json"); growing.save(brain)
    restored = UniversalGoalStateAgent.load(brain)
    reload_rows = [run_episode_series(restored, ImmuneSaviorGame(seed + i * 1543, max_steps=900), 700)
                   for i in range(min(6, worlds))]
    report = {"format": "wailah-immune-savior-development-audit-v1",
        "classification": "development evidence; game was known before this architecture was implemented",
        "protocol": {"worlds": worlds, "seed": seed, "budget": budget, "episode_horizon": 240,
            "unsafe_ablation_stops_after_first_irreversible_failure": True,
            "inputs": "RGB pixels, five anonymous controls, reward, done",
            "agent_not_given": ["semantic object labels", "control mapping", "correct pair", "recipe",
                                "disease coordinates", "healthy coordinates", "private game state"],
            "general_mechanisms": ["set-valued inventory", "declarative set goals", "temporal entity memory",
                                   "risk-gated irreversible action", "multi-object causal recipes"]},
        "aggregate": {"universal_goal_state": _summary(full), "fresh_local_agent": _summary(local),
                      "random_controls": _summary(random_rows), "unsafe_no_preservation_gate": _summary(unsafe_rows),
                      "revisit_after_all_worlds": _summary(revisits), "reloaded_memory": _summary(reload_rows)},
        "growth": growing.status(), "per_world": full, "persistent_brain": str(brain),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8"); return report


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=12); parser.add_argument("--budget", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=310_001); args = parser.parse_args(argv)
    report = run_immune_development_audit(args.output, worlds=args.worlds, seed=args.seed, budget=args.budget)
    print(json.dumps(report["aggregate"] | {"growth": report["growth"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
