"""Two-body Immune Savior and a communicating pixel-learning team.

Each guardian has its own partial RGB view, unknown control permutation, and a
one-item inventory.  The hidden recipe always requires one element accessible
to each guardian, and synthesis requires both bodies to interact at the organ
in the same round.  Learners receive pixels, shared reward, and termination;
semantic simulator state is used only in the audit report.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .immune_savior import ImmuneSaviorGame
from .persistent_workshop import CELL, COLORS, DELTAS, GRID, HEIGHT, MASKS, WIDTH, X0, Y0, _signature
from .universal_goal_state import perceive_state


def perceive_team(frame) -> dict:
    """Parse only public pixels, including the two anonymous inventory slots."""
    rgb = np.asarray(frame, dtype=np.uint8); view = perceive_state(rgb)
    inventory = []
    for patch in (rgb[:, 3:13, 72:82], rgb[:, 3:13, 91:101]):
        signature, _ = _signature(patch); inventory.append(signature)
    return view | {"team_inventory": inventory}


class CooperativeImmuneSaviorGame(ImmuneSaviorGame):
    """A cure is physically impossible without both complementary guardians."""

    agents = 2

    def __init__(self, seed=880_001, *, spread_interval=19, max_steps=900):
        super().__init__(seed, spread_interval=spread_interval, max_steps=max_steps)
        rng = np.random.default_rng(seed + 44_003)
        shuffled = list(rng.permutation(self.element_names))
        self.accessible = (set(shuffled[:2]), set(shuffled[2:]))
        self.correct_pair = frozenset((shuffled[int(rng.integers(2))],
                                       shuffled[2 + int(rng.integers(2))]))
        self.starts = ((4, 3), (4, 5)); self.walls -= set(self.starts)
        self.control_maps = []
        self.interact_actions = []
        for _ in range(2):
            order = rng.permutation(5).tolist()
            self.control_maps.append({order[index]: DELTAS[index] for index in range(4)})
            self.interact_actions.append(order[4])

    def reset(self, seed=None):
        self.positions = list(self.starts); self.inventories = [None, None]; self.treatment = False
        self.available_elements = set(self.element_names); self.healthy = set(self.initial_healthy)
        self.disease = set(self.initial_disease); self.damaged_healthy = 0; self.failed_mixtures = 0
        self.steps = 0; self.done = False; self.episode += 1; self.events = []
        return self.observe()

    def _target_at_team(self, cell):
        for kind, where in self.fixed.items():
            if where == cell and (kind not in self.element_names or kind in self.available_elements): return kind
        if cell in self.disease: return "disease"
        if cell in self.healthy: return "healthy"
        return None

    def _spread_team(self):
        if not self.disease: return None
        blocked = self.walls | set(self.fixed.values()) | self.healthy | self.disease | set(self.positions)
        choices = []
        for row, col in sorted(self.disease):
            for dr, dc in DELTAS:
                cell = row + dr, col + dc
                if 0 <= cell[0] < GRID and 0 <= cell[1] < GRID and cell not in blocked:
                    choices.append(cell)
        if not choices: return None
        cell = choices[int(self._spread_rng.integers(len(choices)))]; self.disease.add(cell)
        self.events.append({"step": self.steps, "event": "disease-spread"}); return cell

    def step(self, actions):
        if self.done: raise RuntimeError("episode ended")
        if len(actions) != 2: raise ValueError("two simultaneous actions required")
        actions = [int(value) for value in actions]; events = ["blocked", "blocked"]
        # Movement is simultaneous and bodies may share a cell.
        for index, action in enumerate(actions):
            if action in self.control_maps[index]:
                dr, dc = self.control_maps[index][action]
                nxt = self.positions[index][0] + dr, self.positions[index][1] + dc
                if 0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID and nxt not in self.walls:
                    self.positions[index] = nxt; events[index] = "moved"
        interacting = [actions[index] == self.interact_actions[index] for index in range(2)]

        # Joint synthesis is evaluated first: neither inventory can hold both elements.
        station = self.fixed["synthesizer"]
        if (all(interacting) and self.positions[0] == station and self.positions[1] == station
                and all(value is not None for value in self.inventories) and not self.treatment):
            pair = frozenset(self.inventories)
            self.inventories = [None, None]
            if pair == self.correct_pair:
                self.treatment = True; events = ["joint-treatment-created"] * 2
            else:
                self.available_elements.update(pair); self.failed_mixtures += 1
                events = ["joint-mixture-inert"] * 2
        else:
            for index in range(2):
                if not interacting[index]: continue
                target = self._target_at_team(self.positions[index]); event = "nothing-happened"
                if (target in self.element_names and target in self.accessible[index]
                        and self.inventories[index] is None and not self.treatment):
                    self.inventories[index] = target; self.available_elements.remove(target)
                    event = "private-element-collected"
                elif target == "disease" and self.treatment:
                    self.disease.remove(self.positions[index]); event = "disease-neutralized"
                elif target == "healthy" and self.treatment:
                    self.healthy.remove(self.positions[index]); self.damaged_healthy += 1
                    event = "healthy-cell-damaged"
                events[index] = event

        self.steps += 1
        if self.steps % self.spread_interval == 0 and not self.treatment: self._spread_team()
        success = not self.disease and self.treatment and self.damaged_healthy == 0
        failure = self.damaged_healthy > 0 or self.steps >= self.max_steps
        self.done = success or failure; reward = 1.0 if success else -1.0 if self.damaged_healthy else 0.0
        self.events.append({"step": self.steps, "events": events})
        info = {"success": success, "failure": failure and not success,
                "healthy_preserved": len(self.healthy), "healthy_damaged": self.damaged_healthy,
                "disease_remaining": len(self.disease), "events_audit_only": events}
        return self.observe(), reward, self.done, info

    def _paint_team(self, frame, kind, cx, cy):
        form, color = self.visual[kind]; mask, rgb = MASKS[form], COLORS[color]
        for channel in range(3):
            frame[channel, cy - 3:cy + 4, cx - 3:cx + 4][mask > 0] = rgb[channel]

    def render_for(self, index):
        frame = np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8); self._paint_team(frame, "cure", 8, 8)
        if self.treatment:
            self._paint_team(frame, "treatment", 77, 8); self._paint_team(frame, "treatment", 96, 8)
        else:
            for slot, item in enumerate(self.inventories):
                if item is not None: self._paint_team(frame, item, 77 + 19 * slot, 8)
        frame[:, 4:12, 48:64] = self.emblem; frame[:, 22:24, 3:109] = 24
        position = self.positions[index]
        visible = {(r, c) for r in range(GRID) for c in range(GRID)
                   if abs(r - position[0]) + abs(c - position[1]) <= 3}
        for row, col in visible:
            y, x = Y0 + row * CELL, X0 + col * CELL
            color = (72, 78, 88) if (row, col) in self.walls else (12, 18, 27)
            frame[:, y:y + CELL, x:x + CELL] = np.asarray(color, dtype=np.uint8)[:, None, None]
        for kind, cell in self.fixed.items():
            if cell in visible and (kind not in self.element_names or kind in self.available_elements):
                self._paint_team(frame, kind, X0 + cell[1] * CELL + 5, Y0 + cell[0] * CELL + 5)
        for cell in self.healthy:
            if cell in visible: self._paint_team(frame, "healthy", X0 + cell[1] * CELL + 5, Y0 + cell[0] * CELL + 5)
        for cell in self.disease:
            if cell in visible: self._paint_team(frame, "disease", X0 + cell[1] * CELL + 5, Y0 + cell[0] * CELL + 5)
        row, col = position; cy, cx = Y0 + row * CELL + 5, X0 + col * CELL + 5
        frame[:, cy - 2:cy + 3, cx] = 255; frame[:, cy, cx - 2:cx + 3] = 255
        return frame

    def observe(self): return [self.render_for(0), self.render_for(1)]

    def audit_state(self):
        return {"seed": self.seed, "positions": list(self.positions), "inventories": list(self.inventories),
                "treatment": self.treatment, "healthy": len(self.healthy), "disease": len(self.disease),
                "healthy_damaged": self.damaged_healthy, "failed_mixtures": self.failed_mixtures,
                "steps": self.steps, "done": self.done,
                "correct_pair_audit_only": sorted(self.correct_pair),
                "accessibility_audit_only": [sorted(row) for row in self.accessible]}


class ImmuneBlackboard:
    """Shared grounded beliefs; contains no simulator names or private state."""

    def __init__(self, *, allow_cooperation=True):
        self.floor = set(); self.walls = set(); self.cells = defaultdict(set); self.forms = {}
        self.home = {}; self.controls = [dict(), dict()]; self.tried = [[], []]; self.interact = [None, None]
        self.accessible = [set(), set()]; self.inaccessible = [set(), set()]
        self.station = None; self.failed_surfaces = set(); self.failed_pairs = set(); self.successful_pair = None
        self.surface_target = None; self.highwater = {}; self.growth = Counter(); self.map_complete = False
        self.messages = []; self.experiments = 0; self.prevented_risky_actions = 0
        self.cooperation_mode = "independent"; self.capacity_evidence = [0, 0]
        self.solo_probe_targets = [set(), set()]; self.cooperation_decision = None
        self.occluded_before = set()
        self.dynamics_probe_steps = [0, 0]
        self.allow_cooperation = bool(allow_cooperation); self.rounds = 0

    def update_view(self, view, *, temporal=True):
        was_complete = self.map_complete
        previous = {signature: set(cells) for signature, cells in self.cells.items()}
        visible = set(view["floor"]) | set(view["walls"]); visible.discard(tuple(view["avatar"]))
        observed = defaultdict(set)
        for obj in view["objects"]:
            signature, cell = obj["signature"], tuple(obj["cell"])
            observed[signature].add(cell); self.forms[signature] = obj["form"]; self.home.setdefault(signature, cell)
        for signature in list(self.cells): self.cells[signature] -= visible
        for signature, cells in observed.items(): self.cells[signature] |= cells
        self.floor |= set(view["floor"]); self.walls |= set(view["walls"])
        became_complete = not self.map_complete and len(self.floor | self.walls) == GRID * GRID
        self.map_complete = self.map_complete or became_complete
        if was_complete and temporal:
            for signature, cells in self.cells.items():
                appeared = set(cells) - previous.get(signature, set())
                causal_candidates = appeared - self.occluded_before
                if signature in previous and causal_candidates:
                    self.growth[signature] += len(causal_candidates)

    def growing(self):
        rows = [(count, signature) for signature, count in self.growth.items()
                if count >= 2]
        return max(rows, default=(0, None))[-1]

    def stable(self):
        growing = self.growing(); rows = [(len(cells), signature) for signature, cells in self.cells.items()
                                          if signature != growing and len(cells) >= 2]
        return max(rows, default=(0, None))[-1]

    def singletons(self):
        excluded = {self.growing(), self.stable()}
        return sorted(signature for signature, cells in self.cells.items()
                      if len(cells) == 1 and signature not in excluded)

    @staticmethod
    def pair(values): return "|".join(sorted(values))

    def publish(self, sender, kind, claim, evidence):
        row = {"sender": int(sender), "kind": str(kind), "claim": claim,
               "evidence": str(evidence), "message_id": ""}
        row["message_id"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
        self.messages.append(row)

    def assess_cooperation(self, inventory):
        """Adopt teamwork only after independent behavior exposes a dependency."""
        if self.cooperation_mode != "independent" or not self.allow_cooperation: return
        saturated = all(value is not None for value in inventory)
        bounded = all(value > 0 for value in self.capacity_evidence)
        complementary = (all(self.accessible) and not (self.accessible[0] & self.accessible[1]))
        independent_failures = all(self.solo_probe_targets)
        if saturated and bounded and complementary and independent_failures:
            self.cooperation_mode = "coordinated"
            self.cooperation_decision = {"necessary": True,
                "decision_round": self.rounds,
                "evidence": ["both local inventories saturated", "distinct collectible affordances",
                             "independent completion probes had no effect", "two different held elements visible"]}
            self.publish(-1, "coordination-decision", "cooperation-necessary",
                         "; ".join(self.cooperation_decision["evidence"]))


class CooperativeGuardian:
    def __init__(self, index, board): self.index = int(index); self.board = board; self.pending = None

    def _route(self, start, target):
        inverse = {tuple(delta): int(action) for action, delta in self.board.controls[self.index].items()}
        queue = deque([(tuple(start), [])]); seen = {tuple(start)}
        while queue:
            cell, route = queue.popleft()
            if cell == tuple(target): return route
            for delta, action in inverse.items():
                nxt = cell[0] + delta[0], cell[1] + delta[1]
                if nxt in self.board.floor and nxt not in seen:
                    seen.add(nxt); queue.append((nxt, route + [action]))
        return None

    def _go(self, view, signature, reason):
        cells = list(self.board.cells.get(signature, set())) or ([self.board.home[signature]]
                if signature in self.board.home else [])
        routes = [(len(route), cell, route) for cell in cells
                  if (route := self._route(view["avatar"], cell)) is not None]
        if not routes: return None
        _, cell, route = min(routes, key=lambda row: (row[0], row[1]))
        if route:
            self.pending = {"kind": "move", "reason": reason, "target": signature, "cell": cell}
            return route[0]
        self.pending = {"kind": "interact", "reason": reason, "target": signature, "cell": cell}
        return int(self.board.interact[self.index])

    def _frontier(self, avatar):
        known = self.board.floor | self.board.walls; rows = []
        for cell in self.board.floor:
            if not any(0 <= cell[0] + dr < GRID and 0 <= cell[1] + dc < GRID and
                       (cell[0] + dr, cell[1] + dc) not in known for dr, dc in DELTAS): continue
            route = self._route(avatar, cell)
            if route: rows.append((len(route), cell, route))
        if not rows: return None
        distance = min(row[0] for row in rows); choices = sorted(row for row in rows if row[0] == distance)
        return choices[0 if self.index == 0 else -1]

    def act(self, frame, reserved=None):
        reserved = reserved if reserved is not None else set(); view = perceive_team(frame)
        self.board.update_view(view, temporal=False); controls = self.board.controls[self.index]
        if len(controls) < 4 or self.board.interact[self.index] is None:
            remaining = [value for value in range(5) if value not in self.board.tried[self.index]]
            if remaining:
                action = remaining[0]
            else:
                unresolved = [value for value in range(5) if str(value) not in controls]
                action = unresolved[len(self.board.tried[self.index]) % len(unresolved)]
            self.board.tried[self.index].append(action)
            self.pending = {"kind": "calibrate", "before": view, "action": action}; return action
        frontier = self._frontier(view["avatar"])
        if frontier is not None:
            self.pending = {"kind": "move", "reason": "divide-and-map"}; return frontier[2][0]

        inventory = view["team_inventory"]; own, other = inventory[self.index], inventory[1 - self.index]
        growing = self.board.growing()
        if own is not None and own == other and growing is not None:  # treatment marker in both slots
            candidates = [cell for cell in self.board.cells.get(growing, set()) if cell not in reserved]
            routes = [(len(route), cell, route) for cell in candidates
                      if (route := self._route(view["avatar"], cell)) is not None]
            if routes:
                _, cell, route = min(routes, key=lambda row: (row[0], row[1])); reserved.add(cell)
                if route:
                    self.pending = {"kind": "move", "reason": "safe-parallel-treatment", "cell": cell}
                    return route[0]
                self.pending = {"kind": "interact", "reason": "safe-parallel-treatment",
                                "target": growing, "cell": cell}
                return int(self.board.interact[self.index])
            self.board.prevented_risky_actions += 1
            for action, delta in self.board.controls[self.index].items():
                cell = view["avatar"][0] + delta[0], view["avatar"][1] + delta[1]
                if cell in self.board.floor:
                    self.pending = {"kind": "move", "reason": "safe-reversible-wait"}
                    return int(action)
            raise RuntimeError("no reversible safety action available")

        if own is None:
            # Prefer elements that participate in the next untested distributed pair.
            preferred = []
            for left in sorted(self.board.accessible[0]):
                for right in sorted(self.board.accessible[1]):
                    if self.board.pair([left, right]) not in self.board.failed_pairs:
                        preferred.append((left, right)[self.index])
            unresolved = [signature for signature in self.board.singletons()
                if signature != self.board.station and signature not in self.board.inaccessible[self.index]
                and signature not in self.board.accessible[0] and signature not in self.board.accessible[1]]
            # Once every combination among known elements has failed, seek a
            # genuinely unresolved object before repeating an exhausted pair.
            candidates = preferred + unresolved + sorted(self.board.accessible[self.index])
            for signature in dict.fromkeys(candidates):
                choice = self._go(view, signature, "test-private-element-affordance")
                if choice is not None: return choice

        if own is not None and self.board.cooperation_mode == "independent":
            # An agent first attempts to complete the task on its own.  The
            # teammate deliberately chooses the opposite end of the candidate
            # list, so these are independent probes rather than covert joint action.
            candidates = [signature for signature in self.board.singletons()
                          if signature != own and signature not in self.board.solo_probe_targets[self.index]]
            if candidates:
                signature = candidates[0 if self.index == 0 else -1]
                choice = self._go(view, signature, "test-independent-completion")
                if choice is not None: return choice

        if own is not None and other is not None and self.board.cooperation_mode == "coordinated":
            if growing is None:
                # Treatment would make the population test irreversible.  The
                # guardians therefore postpone synthesis and patrol different
                # regions until repeated visual appearance supplies evidence.
                floor = sorted(self.board.floor)
                if floor:
                    cursor = self.board.dynamics_probe_steps[self.index]
                    target = floor[(cursor * 7 + self.index * 37) % len(floor)]
                    self.board.dynamics_probe_steps[self.index] += 1
                    route = self._route(view["avatar"], target)
                    if route:
                        self.pending = {"kind": "move", "reason": "observe-before-irreversible-team-action"}
                        return route[0]
                self.pending = {"kind": "wait", "reason": "observe-before-irreversible-team-action"}
                return int(self.board.interact[self.index])
            if self.board.station is not None:
                choice = self._go(view, self.board.station, "coordinate-joint-synthesis")
                if choice is not None: return choice
            if self.board.surface_target is None:
                candidates = [signature for signature in self.board.singletons()
                              if signature not in set(inventory) and signature not in self.board.failed_surfaces]
                if candidates: self.board.surface_target = candidates[0]
            if self.board.surface_target is not None:
                choice = self._go(view, self.board.surface_target, "discover-joint-synthesis-surface")
                if choice is not None: return choice

        self.pending = {"kind": "wait", "reason": "wait-for-teammate"}
        return int(self.board.interact[self.index])


class CooperativeImmuneTeam:
    """Two local controllers plus an explicit experience-sharing blackboard."""

    def __init__(self, *, allow_cooperation=True):
        self.board = ImmuneBlackboard(allow_cooperation=allow_cooperation)
        self.guardians = [CooperativeGuardian(i, self.board) for i in range(2)]

    def begin(self, frames):
        for guardian in self.guardians: guardian.pending = None
        for frame in frames: self.board.update_view(perceive_team(frame), temporal=False)

    def act(self, frames):
        self.board.occluded_before = {tuple(perceive_team(frame)["avatar"]) for frame in frames}
        reserved = set(); return [guardian.act(frames[i], reserved) for i, guardian in enumerate(self.guardians)]

    def observe(self, before_frames, actions, after_frames, reward, done):
        self.board.rounds += 1
        before = [perceive_team(frame) for frame in before_frames]
        after = [perceive_team(frame) for frame in after_frames]
        before_inventory, after_inventory = before[0]["team_inventory"], after[0]["team_inventory"]
        for index, guardian in enumerate(self.guardians):
            pending = guardian.pending or {}; kind = pending.get("kind")
            if kind == "calibrate":
                old, new = before[index]["avatar"], after[index]["avatar"]
                delta = new[0] - old[0], new[1] - old[1]
                if delta in DELTAS: self.board.controls[index][str(int(actions[index]))] = list(delta)
                if len(self.board.controls[index]) == 4:
                    remaining = set(range(5)) - {int(value) for value in self.board.controls[index]}
                    if len(remaining) == 1: self.board.interact[index] = remaining.pop()

        # Infer local collectibility from the public inventory pixels.
        for index, guardian in enumerate(self.guardians):
            pending = guardian.pending or {}; target = pending.get("target")
            if pending.get("kind") != "interact" or target is None: continue
            if before_inventory[index] is None and after_inventory[index] is not None:
                self.board.accessible[index].add(after_inventory[index]); self.board.cells[target].discard(tuple(pending["cell"]))
                self.board.publish(index, "affordance", after_inventory[index], "inventory changed after contact")
            elif (before_inventory[index] is None and after_inventory[index] is None
                  and pending.get("reason") == "test-private-element-affordance"):
                self.board.inaccessible[index].add(target)
            elif (before_inventory[index] is not None and after_inventory == before_inventory
                  and pending.get("reason") == "test-independent-completion"):
                self.board.capacity_evidence[index] += 1
                self.board.solo_probe_targets[index].add(target)

        both_interacted = all((guardian.pending or {}).get("kind") == "interact"
                              for guardian in self.guardians)
        same_target = ((self.guardians[0].pending or {}).get("target") ==
                       (self.guardians[1].pending or {}).get("target"))
        had_pair = all(value is not None for value in before_inventory) and before_inventory[0] != before_inventory[1]
        if both_interacted and same_target and had_pair:
            target = (self.guardians[0].pending or {}).get("target"); changed = after_inventory != before_inventory
            self.board.experiments += 1
            if changed:
                self.board.station = target; pair = self.board.pair(before_inventory)
                treatment = (after_inventory[0] is not None and after_inventory[0] == after_inventory[1]
                             and after_inventory[0] not in before_inventory)
                if treatment:
                    self.board.successful_pair = pair
                    self.board.publish(0, "joint-recipe", pair, "both inventories became a new shared visual state")
                else:
                    self.board.failed_pairs.add(pair)
                    for signature in before_inventory:
                        if signature in self.board.home: self.board.cells[signature].add(self.board.home[signature])
                self.board.surface_target = None
            elif target is not None:
                self.board.failed_surfaces.add(target); self.board.surface_target = None

        self.board.assess_cooperation(after_inventory)

        # Merge both newly observed local views only after attributing actions.
        for view in after: self.board.update_view(view, temporal=True)
        for guardian in self.guardians:
            pending = guardian.pending or {}
            if pending.get("reason") == "safe-parallel-treatment":
                self.board.cells[pending.get("target")].discard(tuple(pending.get("cell")))
            guardian.pending = None
        return {"reward": float(reward), "done": bool(done), "map_complete": self.board.map_complete,
                "growing_concept": self.board.growing(), "messages": len(self.board.messages)}

    def status(self):
        return {"controls_grounded": [len(row) == 4 and self.board.interact[i] is not None
                                      for i, row in enumerate(self.board.controls)],
                "map_complete": self.board.map_complete, "learned_private_affordances":
                    [len(row) for row in self.board.accessible],
                "joint_recipe_learned": self.board.successful_pair is not None,
                "joint_experiments": self.board.experiments, "shared_messages": len(self.board.messages),
                "cooperation_mode": self.board.cooperation_mode,
                "cooperation_allowed": self.board.allow_cooperation,
                "cooperation_decision": self.board.cooperation_decision,
                "independent_probe_counts": [len(row) for row in self.board.solo_probe_targets],
                "dynamics_probe_steps": list(self.board.dynamics_probe_steps),
                "growing_concept_learned": self.board.growing() is not None,
                "prevented_risky_actions": self.board.prevented_risky_actions}


def run_team_episode(team, game, budget=1500, horizon=900):
    used = episodes = damage = 0; success = False; info = {}; trace = []
    while used < budget and not success:
        frames = game.reset(); team.begin(frames); episodes += 1
        for _ in range(min(horizon, budget - used)):
            actions = team.act(frames); later, reward, done, info = game.step(actions)
            learned = team.observe(frames, actions, later, reward, done); used += 1
            if learned["messages"] > (trace[-1]["messages"] if trace else 0) or reward != 0:
                trace.append({"round": used, "actions": actions, **learned})
            frames = later
            if done:
                success = bool(info["success"]); damage += int(info["healthy_damaged"]); break
    return {"success": success, "joint_rounds": used, "primitive_actions": used * 2,
            "episodes": episodes, "healthy_damaged": damage,
            "disease_remaining": int(info.get("disease_remaining", -1)),
            "team": team.status(), "trace": trace[-40:]}


def _random_pair(game, budget, seed):
    rng = np.random.default_rng(seed); frames = game.reset(); damage = 0
    for used in range(1, budget + 1):
        frames, _, done, info = game.step([int(rng.integers(5)), int(rng.integers(5))])
        if done:
            damage += int(info["healthy_damaged"])
            return {"success": bool(info["success"]), "joint_rounds": used,
                    "primitive_actions": used * 2, "healthy_damaged": damage}
    return {"success": False, "joint_rounds": budget, "primitive_actions": budget * 2,
            "healthy_damaged": damage}


def _summary(rows):
    return {"worlds": len(rows), "successes": sum(row["success"] for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "zero_damage_worlds": sum(row.get("healthy_damaged", 0) == 0 for row in rows),
            "mean_joint_rounds": float(np.mean([row["joint_rounds"] for row in rows]))}


def run_cooperative_audit(output, *, worlds=12, seed=910_001, budget=1500):
    communicating = []; independent_only = []; random_rows = []
    for index in range(worlds):
        world_seed = seed + index * 1877
        communicating.append(run_team_episode(CooperativeImmuneTeam(),
            CooperativeImmuneSaviorGame(world_seed, max_steps=budget), budget))
        independent_only.append(run_team_episode(CooperativeImmuneTeam(allow_cooperation=False),
            CooperativeImmuneSaviorGame(world_seed, max_steps=budget), budget))
        random_rows.append(_random_pair(CooperativeImmuneSaviorGame(world_seed, max_steps=budget),
                                        budget, world_seed + 71))
    report = {"format": "wailah-cooperative-immune-savior-v19-development-audit-v1",
        "classification": "development evidence for simultaneous two-agent grounded cooperation",
        "protocol": {"worlds": worlds, "seed": seed, "budget_joint_rounds": budget,
            "agents": 2, "observations": "two private partial RGB views",
            "controls": "independently remapped five-action vocabularies",
            "cooperation_required_by_world_but_not_disclosed_to_agents": ["one-item capacity", "complementary element access",
                                     "simultaneous joint synthesis"],
            "agent_not_given": ["object labels", "controls", "element accessibility", "recipe",
                                "synthesis location", "disease identity", "healthy identity", "coordinates",
                                "whether cooperation is necessary"]},
        "aggregate": {"communicating_pair": _summary(communicating),
                      "same_agents_cooperation_disabled": _summary(independent_only),
                      "random_pair": _summary(random_rows)},
        "per_world": communicating,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8"); return report


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=12); parser.add_argument("--budget", type=int, default=1500)
    args = parser.parse_args(argv); report = run_cooperative_audit(args.output, worlds=args.worlds, budget=args.budget)
    print(json.dumps(report["aggregate"], indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
