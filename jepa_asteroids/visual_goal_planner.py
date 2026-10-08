"""Reward-light visual goal inference and short-horizon predictive planning.

The planner receives only categorical pixels, available actions, and learned
action effects.  It does not contain game IDs, instructions, or named goals.
It complements graph/procedure memory in three places where exact-state replay
is weak: multiple simultaneously controllable bodies, approaching visual
hazards, and feature goals learned from sparse progress.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations, permutations
import math

import numpy as np


def _features(grid: np.ndarray) -> np.ndarray:
    """Compact scene statistics suitable for an action-conditioned model."""
    background = int(np.unique(grid, return_counts=True)[0][
        int(np.argmax(np.unique(grid, return_counts=True)[1]))])
    occupied = grid != background
    horizontal = float(np.mean(grid[:, 1:] != grid[:, :-1])) if grid.shape[1] > 1 else 0.0
    vertical = float(np.mean(grid[1:, :] != grid[:-1, :])) if grid.shape[0] > 1 else 0.0
    symmetry_x = float(np.mean(grid == grid[:, ::-1]))
    symmetry_y = float(np.mean(grid == grid[::-1, :]))
    values, counts = np.unique(grid, return_counts=True)
    probabilities = counts.astype(float) / max(1, grid.size)
    entropy = float(-(probabilities * np.log2(probabilities + 1e-12)).sum()) / 4.0
    return np.asarray([float(occupied.mean()), horizontal, vertical, symmetry_x,
                       symmetry_y, entropy, len(values) / 16.0,
                       float(counts.max()) / max(1, grid.size)], dtype=float)


def _median_vectors(motion, signature: tuple, actions: list[int]) -> dict[int, tuple[int, int]]:
    result = {}
    for action in actions:
        values = motion.get(signature, {}).get(action, [])
        if len(values) < 2:
            continue
        dx = int(round(float(np.median([row[0] for row in values[-32:]]))))
        dy = int(round(float(np.median([row[1] for row in values[-32:]]))))
        if dx or dy:
            result[int(action)] = (dx, dy)
    return result


def _distance_to_color(grid: np.ndarray, color: int, x: float, y: float) -> float:
    ys, xs = np.nonzero(grid == int(color))
    if not len(xs):
        return float(max(grid.shape))
    return float(np.min(np.abs(xs - x) + np.abs(ys - y)))


class VisualGoalPlanner:
    """Learn visual objectives and evaluate actions by predicted consequences."""

    format = "wailah-visual-goal-planner-v1"

    def __init__(self):
        self.relation_attempts = Counter()
        self.relation_success = Counter()
        self.relation_failures = Counter()
        self.hazard_colors = Counter()
        self.color_exposure = Counter()
        self.color_approach = defaultdict(list)
        self.action_feature_deltas = defaultdict(list)
        self.progress_feature_direction = np.zeros(8, dtype=float)
        self.progress_feature_examples = 0
        self.current_goal = None
        self.episode_steps = 0
        self.predictive_plan_steps = 0
        self.dynamic_avoidance_steps = 0
        self.multi_body_plan_steps = 0
        self.feature_goal_steps = 0
        self.goals_tested = 0
        self.last_reason = None
        self.last_prediction = None
        self.last_effectors = {}

    def reset_episode(self) -> None:
        self.current_goal = None
        self.episode_steps = 0
        self.color_approach = defaultdict(list)
        self.last_reason = None
        self.last_prediction = None
        self.last_effectors = {}

    @staticmethod
    def _controllable(motion, actions: list[int]) -> dict[tuple, dict[int, tuple[int, int]]]:
        result = {}
        for signature in motion:
            # Large panels and sweeping overlays often move autonomously and
            # can spuriously correlate with whichever buttons happened to be
            # pressed.  Candidate effectors must be compact persistent bodies.
            if int(signature[1]) > 6 or int(signature[2]) >= 15 or int(signature[3]) >= 15:
                continue
            vectors = _median_vectors(motion, signature, actions)
            if len(set(vectors.values())) >= min(2, len(actions)):
                result[signature] = vectors
        return result

    @staticmethod
    def _has_divergent_effectors(controllable) -> bool:
        """Separate independent bodies from multiple visible parts of one body."""
        rows = list(controllable.values())
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1:]:
                for action in set(left).intersection(right):
                    left_dx, left_dy = left[action]
                    right_dx, right_dy = right[action]
                    if abs(left_dx - right_dx) + abs(left_dy - right_dy) >= 2.0:
                        return True
        return False

    @staticmethod
    def _matching_object(objects: list[dict], signature: tuple, near=None):
        rows = [obj for obj in objects if obj["sig"] == signature]
        if not rows:
            return None
        if near is None:
            return rows[0]
        return min(rows, key=lambda obj: abs(obj["cx"] - near[0]) + abs(obj["cy"] - near[1]))

    @staticmethod
    def _relation_key(source: dict, target: dict) -> str:
        source_shape = ",".join(map(str, source["sig"]))
        target_shape = ",".join(map(str, target["sig"]))
        return f"{source_shape}->{target_shape}"

    def _danger_action(self, grid, objects, actions, controllable, passable_colors,
                       moving_colors):
        if not controllable:
            return None
        controlled = []
        controlled_colors = set()
        for signature, vectors in controllable.items():
            obj = self._matching_object(objects, signature)
            if obj is None:
                remembered = self.last_effectors.get(signature)
                if remembered is not None:
                    # Preserve causal identity briefly through partial visual
                    # occlusion.  The remembered position is episode-local.
                    obj = dict(remembered)
            if obj is not None:
                controlled.append((obj, vectors)); controlled_colors.add(obj["color"])
        if not controlled:
            return None
        values, counts = np.unique(grid, return_counts=True)
        area = {int(value): int(count) for value, count in zip(values, counts)}
        dangers = []
        for color, count in area.items():
            if color in controlled_colors:
                continue
            history = self.color_approach.get(color, [])[-8:]
            approach = -float(np.median(history)) if len(history) >= 2 else 0.0
            learned = float(self.hazard_colors[color]) / max(1e-9, self.color_exposure[color])
            if color in passable_colors and learned < .10:
                continue
            if learned >= .10 or (color in moving_colors and approach > .35
                                  and count >= max(8, grid.size // 80)):
                dangers.append((color, learned, approach, count))
        if not dangers:
            return None
        current_clearance = min(_distance_to_color(grid, color, obj["cx"], obj["cy"])
                                for obj, _ in controlled for color, _, _, _ in dangers)
        if current_clearance > 12 and not any(learned >= 2 for _, learned, _, _ in dangers):
            return None
        effectful_actions = [action for action in actions
                             if any(action in vectors for _, vectors in controlled)]
        if not effectful_actions:
            return None
        scored = []
        for action in effectful_actions:
            clearances = []
            moved = 0
            for obj, vectors in controlled:
                dx, dy = vectors.get(action, (0, 0)); moved += int(bool(dx or dy))
                x, y = obj["cx"] + dx, obj["cy"] + dy
                edge = min(x, y, grid.shape[1] - 1 - x, grid.shape[0] - 1 - y)
                relation = [(_distance_to_color(grid, color, x, y), learned, approach)
                            for color, learned, approach, _ in dangers]
                local = min(distance / (1.0 + .4 * learned + .6 * approach)
                            for distance, learned, approach in relation)
                clearances.append(local + .05 * edge)
            score = min(clearances) + .08 * moved
            scored.append((score, -action, action))
        best = max(scored)
        # Do not seize control when every predicted action is effectively tied.
        if max(row[0] for row in scored) - min(row[0] for row in scored) < .25:
            return None
        self.dynamic_avoidance_steps += 1; self.predictive_plan_steps += 1
        self.last_reason = "predict-visual-collision"
        self.last_prediction = {"clearance_before": current_clearance,
                                "clearance_after_score": float(best[0]),
                                "danger_colors": [row[0] for row in dangers]}
        return int(best[2]), self.last_reason

    def _select_multi_goal(self, objects, controllable):
        sources = [(obj, controllable[obj["sig"]]) for obj in objects if obj["sig"] in controllable]
        if len({obj["sig"] for obj, _ in sources}) < 2:
            return None
        controlled_signatures = set(controllable)
        targets = [obj for obj in objects if obj["sig"] not in controlled_signatures
                   and 2 <= obj["area"] <= 96 and obj["w"] < 15 and obj["h"] < 15]
        if not targets:
            return None
        candidates = []
        for source, vectors in sources:
            for target in targets:
                key = self._relation_key(source, target)
                distance = abs(source["cx"] - target["cx"]) + abs(source["cy"] - target["cy"])
                shape_gap = abs(math.log2(max(1, source["area"])) -
                                math.log2(max(1, target["area"])))
                score = (40.0 * self.relation_success[key] - 12.0 * self.relation_failures[key]
                         + 4.0 / math.sqrt(1 + self.relation_attempts[key])
                         - .08 * distance - .5 * shape_gap)
                candidates.append((score, -distance, key, source, target, vectors))
        if not candidates:
            return None
        _, _, key, source, target, vectors = max(candidates, key=lambda row: row[:3])
        self.current_goal = {"key": key, "source_sig": list(source["sig"]),
                             "source": [source["cx"], source["cy"]],
                             "target_sig": list(target["sig"]),
                             "target": [target["cx"], target["cy"]], "age": 0}
        self.relation_attempts[key] += 1; self.goals_tested += 1
        return source, target, vectors

    def _resolve_multi_goal(self, objects, controllable):
        if self.current_goal is not None and self.current_goal.get("joint"):
            self.current_goal = None
        if self.current_goal is None:
            return self._select_multi_goal(objects, controllable)
        source_sig = tuple(self.current_goal["source_sig"])
        target_sig = tuple(self.current_goal["target_sig"])
        if source_sig not in controllable:
            self.current_goal = None
            return self._select_multi_goal(objects, controllable)
        source = self._matching_object(objects, source_sig, self.current_goal["source"])
        target = self._matching_object(objects, target_sig, self.current_goal["target"])
        self.current_goal["age"] += 1
        if source is None or target is None or self.current_goal["age"] > 64:
            self.relation_failures[self.current_goal["key"]] += 1
            self.current_goal = None
            return self._select_multi_goal(objects, controllable)
        self.current_goal["source"] = [source["cx"], source["cy"]]
        self.current_goal["target"] = [target["cx"], target["cy"]]
        return source, target, controllable[source_sig]

    def _multi_body_action(self, grid, objects, actions, controllable, passable_colors):
        goal = self._resolve_multi_goal(objects, controllable)
        if goal is None:
            return None
        source, target, vectors = goal
        allowed = set(int(color) for color in passable_colors)
        target_colors = {int(target["color"]), int(source["color"])}
        # A small beam search is more stable than one-step greed when controls
        # are remapped or a direct move meets an obstacle.
        beam = [(0.0, source["cx"], source["cy"], None)]
        for depth in range(1, 6):
            expanded = []
            for cost, x, y, first in beam:
                for action, (dx, dy) in vectors.items():
                    nx, ny = x + dx, y + dy
                    ix, iy = int(round(nx)), int(round(ny))
                    if not (0 <= ix < grid.shape[1] and 0 <= iy < grid.shape[0]):
                        continue
                    pixel = int(grid[iy, ix])
                    collision = 0.0 if not allowed or pixel in allowed | target_colors else 5.0
                    distance = abs(nx - target["cx"]) + abs(ny - target["cy"])
                    expanded.append((.25 * cost + distance + collision, nx, ny,
                                     action if first is None else first))
            if not expanded:
                break
            expanded.sort(key=lambda row: (row[0], row[3])); beam = expanded[:24]
        if not beam:
            return None
        action = int(min(beam, key=lambda row: (row[0], row[3]))[3])
        self.multi_body_plan_steps += 1; self.predictive_plan_steps += 1
        self.last_reason = "infer-multi-body-visual-goal"
        self.last_prediction = {"source": [source["cx"], source["cy"]],
                                "target": [target["cx"], target["cy"]],
                                "relation": self.current_goal["key"]}
        return action, self.last_reason

    def _joint_multi_action(self, grid, objects, actions, controllable, passable_colors):
        """Plan one action for several independently controlled visual bodies."""
        sources = []
        for signature, vectors in controllable.items():
            obj = self._matching_object(objects, signature)
            if obj is not None:
                sources.append((obj, vectors))
        if len(sources) < 2:
            return None
        controlled_signatures = set(controllable)
        candidates = [obj for obj in objects if obj["sig"] not in controlled_signatures
                      and 2 <= obj["area"] <= 96 and obj["w"] < 15 and obj["h"] < 15]
        by_color = defaultdict(list)
        for target in candidates:
            by_color[int(target["color"])].append(target)
        assignments = []
        count = len(sources)
        for color, targets in by_color.items():
            if len(targets) < count:
                continue
            nearest = sorted(targets, key=lambda target: min(
                abs(target["cx"] - source[0]["cx"]) + abs(target["cy"] - source[0]["cy"])
                for source in sources))[:6]
            for subset in combinations(nearest, count):
                for ordered in permutations(subset):
                    distance = 0.0; shape_gap = 0.0
                    for (source, _), target in zip(sources, ordered):
                        distance += abs(source["cx"] - target["cx"]) + abs(source["cy"] - target["cy"])
                        shape_gap += abs(math.log2(max(1, source["area"])) -
                                         math.log2(max(1, target["area"])))
                    assignments.append((distance + .5 * shape_gap, color, ordered))
        if not assignments:
            return None
        _, color, targets = min(assignments, key=lambda row: (row[0], row[1]))
        allowed = set(int(value) for value in passable_colors)
        target_colors = {color} | {int(source[0]["color"]) for source in sources}
        start = tuple((source["cx"], source["cy"]) for source, _ in sources)
        beam = [(0.0, start, None)]
        for _depth in range(1, 6):
            expanded = []
            for cost, positions, first in beam:
                for action in actions:
                    moved = []; collision = 0.0
                    for (source, vectors), (x, y) in zip(sources, positions):
                        dx, dy = vectors.get(action, (0, 0)); nx, ny = x + dx, y + dy
                        ix, iy = int(round(nx)), int(round(ny))
                        if not (0 <= ix < grid.shape[1] and 0 <= iy < grid.shape[0]):
                            collision += 20.0
                        else:
                            pixel = int(grid[iy, ix])
                            if allowed and pixel not in allowed | target_colors:
                                collision += 5.0
                        moved.append((nx, ny))
                    distance = sum(abs(x - target["cx"]) + abs(y - target["cy"])
                                   for (x, y), target in zip(moved, targets))
                    expanded.append((.2 * cost + distance + collision, tuple(moved),
                                     action if first is None else first))
            if not expanded:
                break
            expanded.sort(key=lambda row: (row[0], row[2])); beam = expanded[:32]
        if not beam:
            return None
        action = int(beam[0][2])
        relation = "+".join(
            self._relation_key(source, target) for (source, _), target in zip(sources, targets))
        self.current_goal = {"key": relation, "joint": True, "age": 0}
        self.relation_attempts[relation] += 1; self.goals_tested += 1
        self.multi_body_plan_steps += 1; self.predictive_plan_steps += 1
        self.last_reason = "infer-joint-visual-goal"
        self.last_prediction = {"sources": [[row[0]["cx"], row[0]["cy"]] for row in sources],
                                "targets": [[row["cx"], row["cy"]] for row in targets],
                                "relation": relation}
        return action, self.last_reason

    def _feature_goal_action(self, actions):
        if self.progress_feature_examples < 2 or np.linalg.norm(self.progress_feature_direction) < 1e-8:
            return None
        scored = []
        goal = self.progress_feature_direction
        for action in actions:
            rows = self.action_feature_deltas.get(int(action), [])[-64:]
            if len(rows) < 3:
                continue
            predicted = np.median(np.asarray(rows, dtype=float), axis=0)
            score = float(np.dot(predicted, goal) / (np.linalg.norm(predicted) * np.linalg.norm(goal) + 1e-9))
            scored.append((score, -action, action))
        if not scored or max(row[0] for row in scored) < .15:
            return None
        action = int(max(scored)[2]); self.feature_goal_steps += 1; self.predictive_plan_steps += 1
        self.last_reason = "predict-learned-feature-goal"
        self.last_prediction = {"examples": self.progress_feature_examples,
                                "score": float(max(scored)[0])}
        return action, self.last_reason

    def recommend(self, grid: np.ndarray, objects: list[dict], actions: list[int], motion,
                  passable_colors, allow_feature_goal: bool = True):
        self.episode_steps += 1
        controllable = self._controllable(motion, actions)
        multi_axis = {signature: vectors for signature, vectors in controllable.items()
                      if any(dx for dx, _ in vectors.values()) and any(dy for _, dy in vectors.values())}
        divergent = self._has_divergent_effectors(multi_axis)
        moving_colors = {int(signature[0]) for signature, by_action in motion.items()
                         if sum(len(values) for values in by_action.values()) >= 2}
        if divergent:
            # In opposed-control scenes, relative approach itself is useful
            # predictive evidence even when a large moving region cannot be
            # tracked as one stable connected component.
            moving_colors.update(map(int, np.unique(grid)))
        recommendation = self._danger_action(grid, objects, actions, controllable,
                                             set(passable_colors), moving_colors)
        if recommendation is not None:
            return recommendation
        if divergent:
            recommendation = self._multi_body_action(grid, objects, actions, multi_axis,
                                                     set(passable_colors))
            if recommendation is not None:
                return recommendation
        if allow_feature_goal:
            return self._feature_goal_action(actions)
        return None

    def observe(self, before: np.ndarray, action: int, after: np.ndarray, progress: bool,
                done: bool, before_objects: list[dict], after_objects: list[dict], motion) -> None:
        before_features, after_features = _features(before), _features(after)
        delta = after_features - before_features
        self.action_feature_deltas[int(action)].append(delta.tolist())
        self.action_feature_deltas[int(action)] = self.action_feature_deltas[int(action)][-128:]
        if progress:
            rate = 1.0 / (self.progress_feature_examples + 1)
            self.progress_feature_direction = ((1.0 - rate) * self.progress_feature_direction + rate * delta)
            self.progress_feature_examples += 1
            if self.current_goal is not None:
                self.relation_success[self.current_goal["key"]] += 1
        controllable = self._controllable(motion, list({int(action)} | set(motion_action
            for rows in motion.values() for motion_action in rows)))
        controlled_before = [obj for obj in before_objects if obj["sig"] in controllable]
        controlled_after = [obj for obj in after_objects if obj["sig"] in controllable]
        for obj in controlled_before + controlled_after:
            self.last_effectors[obj["sig"]] = dict(obj)
        if not controlled_before:
            controlled_before = [dict(obj) for signature, obj in self.last_effectors.items()
                                 if signature in controllable]
        if not controlled_after:
            # Match a partially occluded remnant by color and proximity while
            # retaining the learned causal signature in memory.
            proxies = []
            for signature, remembered in self.last_effectors.items():
                if signature not in controllable:
                    continue
                same_color = [obj for obj in after_objects if obj["color"] == remembered["color"]]
                if same_color:
                    proxy = min(same_color, key=lambda obj: abs(obj["cx"] - remembered["cx"]) +
                                abs(obj["cy"] - remembered["cy"]))
                    if abs(proxy["cx"] - remembered["cx"]) + abs(proxy["cy"] - remembered["cy"]) <= 8:
                        proxies.append(proxy)
            controlled_after = proxies or [dict(obj) for signature, obj in self.last_effectors.items()
                                           if signature in controllable]
        colors = set(map(int, np.unique(before))) | set(map(int, np.unique(after)))
        controlled_colors = {obj["color"] for obj in controlled_before + controlled_after}
        for color in colors - controlled_colors:
            if not controlled_before or not controlled_after:
                continue
            old_distance = min(_distance_to_color(before, color, obj["cx"], obj["cy"])
                               for obj in controlled_before)
            new_distance = min(_distance_to_color(after, color, obj["cx"], obj["cy"])
                               for obj in controlled_after)
            self.color_approach[color].append(new_distance - old_distance)
            self.color_approach[color] = self.color_approach[color][-16:]
            if old_distance <= 12:
                self.color_exposure[color] += 1.0 / (1.0 + old_distance)
        if done and not progress and controlled_before:
            # Failure is evidence about what was visually close immediately
            # beforehand. Credit colors smoothly instead of declaring a label.
            for color in colors - controlled_colors:
                distance = min(min(_distance_to_color(before, color, obj["cx"], obj["cy"]),
                                   _distance_to_color(after, color, obj["cx"], obj["cy"]))
                               for obj in controlled_before)
                if distance <= 12:
                    self.hazard_colors[color] += 1.0 / (1.0 + distance)
            if self.current_goal is not None:
                self.relation_failures[self.current_goal["key"]] += 1
        if progress or done:
            self.current_goal = None

    def status(self) -> dict:
        return {"format": self.format, "predictive_plan_steps": self.predictive_plan_steps,
                "dynamic_avoidance_steps": self.dynamic_avoidance_steps,
                "multi_body_plan_steps": self.multi_body_plan_steps,
                "feature_goal_steps": self.feature_goal_steps,
                "visual_goals_tested": self.goals_tested,
                "successful_visual_relations": int(sum(self.relation_success.values())),
                "learned_hazard_colors": {str(k): float(v) for k, v in self.hazard_colors.items()
                                          if v > 0},
                "hazard_color_exposure": {str(k): float(v) for k, v in self.color_exposure.items()
                                          if v > 0},
                "progress_feature_examples": self.progress_feature_examples,
                "last_reason": self.last_reason, "last_prediction": self.last_prediction}

    def export(self) -> dict:
        return self.status() | {
            "relation_attempts": dict(self.relation_attempts),
            "relation_success": dict(self.relation_success),
            "relation_failures": dict(self.relation_failures),
            "action_feature_deltas": {str(k): v for k, v in self.action_feature_deltas.items()},
            "progress_feature_direction": self.progress_feature_direction.tolist()}

    def restore(self, value: dict) -> None:
        if value.get("format") != self.format:
            raise ValueError("unsupported visual goal planner format")
        self.relation_attempts = Counter({str(k): int(v) for k, v in value.get("relation_attempts", {}).items()})
        self.relation_success = Counter({str(k): int(v) for k, v in value.get("relation_success", {}).items()})
        self.relation_failures = Counter({str(k): int(v) for k, v in value.get("relation_failures", {}).items()})
        self.hazard_colors = Counter({int(k): float(v) for k, v in value.get("learned_hazard_colors", {}).items()})
        self.color_exposure = Counter({int(k): float(v) for k, v in value.get("hazard_color_exposure", {}).items()})
        self.action_feature_deltas = defaultdict(list, {
            int(k): [[float(x) for x in row] for row in rows]
            for k, rows in value.get("action_feature_deltas", {}).items()})
        self.progress_feature_direction = np.asarray(value.get("progress_feature_direction", [0.0] * 8), dtype=float)
        self.progress_feature_examples = int(value.get("progress_feature_examples", 0))
        self.predictive_plan_steps = int(value.get("predictive_plan_steps", 0))
        self.dynamic_avoidance_steps = int(value.get("dynamic_avoidance_steps", 0))
        self.multi_body_plan_steps = int(value.get("multi_body_plan_steps", 0))
        self.feature_goal_steps = int(value.get("feature_goal_steps", 0))
        self.goals_tested = int(value.get("visual_goals_tested", 0))
        self.last_reason = value.get("last_reason"); self.last_prediction = value.get("last_prediction")
