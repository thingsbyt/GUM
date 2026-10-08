"""Object-centric continual learner for free-form pixel canvases.

Unlike the tiled learners, this agent discovers connected visual entities,
identifies its effector through action-caused motion, grounds control vectors,
tracks anonymous populations, learns object combinations, and protects stable
populations when testing irreversible interventions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def _hwc(frame):
    value = np.asarray(frame, dtype=np.uint8)
    if value.ndim != 3: raise ValueError("expected RGB image")
    if value.shape[0] == 3: value = np.transpose(value, (1, 2, 0))
    if value.shape[-1] != 3: raise ValueError("expected RGB image")
    return value


def segment_components(frame):
    """Position-invariant connected components with no object labels."""
    rgb = _hwc(frame); background = np.median(rgb.reshape(-1, 3), axis=0).round().astype(np.uint8)
    foreground = np.max(np.abs(rgb.astype(np.int16) - background.astype(np.int16)), axis=2) > 18
    labels, count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    rows = []
    for label, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None: continue
        local = labels[slices] == label; local_y, local_x = np.where(local); area = len(local_x)
        if area < 7 or area > 1800: continue
        y0, y1 = slices[0].start, slices[0].stop; x0, x1 = slices[1].start, slices[1].stop
        mask = local; colors = rgb[slices][local]; mean = colors.mean(0).round().astype(np.uint8)
        form = hashlib.sha256(mask.tobytes() + bytes(mask.shape)).hexdigest()[:12]
        signature = hashlib.sha256(mask.tobytes() + mean.tobytes() + bytes(mask.shape)).hexdigest()[:16]
        rows.append({"signature": signature, "form": form,
                     "centroid": (float(x0 + local_x.mean()), float(y0 + local_y.mean())),
                     "area": area, "bbox": (x0, y0, x1, y1)})
    context = hashlib.sha256(bytes(background) + np.asarray(rgb.shape[:2], dtype=np.uint16).tobytes()).hexdigest()[:20]
    return {"shape": list(rgb.shape), "background": background.tolist(), "context": context,
            "components": rows}


def _group(view):
    rows = defaultdict(list)
    for component in view["components"]: rows[component["signature"]].append(component)
    return rows


def _changed(before, after, signature, threshold=3.0):
    old = [row["centroid"] for row in _group(before).get(signature, [])]
    new = [row["centroid"] for row in _group(after).get(signature, [])]
    if len(old) != len(new): return True
    if not old: return False
    unmatched = list(new)
    for point in old:
        distances = [np.hypot(point[0] - other[0], point[1] - other[1]) for other in unmatched]
        index = int(np.argmin(distances))
        if distances[index] > threshold: return True
        unmatched.pop(index)
    return False


class ObjectCentricAgent:
    format = "wailah-object-centric-agent-v1"

    def __init__(self):
        self.worlds = {}; self.current = None; self.pending = None
        self.form_roles = defaultdict(Counter); self.self_generated_subgoals = 0; self.experiments = 0

    @property
    def memory(self): return self.worlds[self.current]

    def begin(self, frame):
        view = segment_components(frame); self.current = view["context"]
        if self.current not in self.worlds:
            self.worlds[self.current] = {"controls": {}, "tried": [], "interact": None,
                "effector_signature": None, "entities": {}, "ingredients": [], "noncollectible": [],
                "failed_surfaces": [], "station": None, "failed_pairs": [], "successful_pair": None,
                "endogenous": {}, "max_counts": {}, "sightings": {}, "visits": 0, "actions": 0}
        self.memory.setdefault("sightings", {})
        self.memory["visits"] += 1; self.pending = None; self.carried = []; self.candidate_pair = None
        self.treatment_ready = False; self.treatment_active = False; self.probe_steps = 0; self.active_target = None
        self._update(view); return self.current

    def _update(self, view):
        groups = _group(view); entities = {}
        for signature, rows in groups.items():
            entities[signature] = {"form": rows[0]["form"], "positions": [list(row["centroid"]) for row in rows],
                                   "count": len(rows)}
            self.memory["max_counts"][signature] = max(self.memory["max_counts"].get(signature, 0), len(rows))
            self.memory["sightings"][signature] = self.memory["sightings"].get(signature, 0) + 1
        self.memory["entities"] = entities

    def _ready(self): return len(self.memory["controls"]) == 4 and self.memory["interact"] is not None

    def _effector(self, view):
        signature = self.memory["effector_signature"]
        rows = _group(view).get(signature, []) if signature else []
        return rows[0]["centroid"] if len(rows) == 1 else None

    def _direct_growing(self):
        rows = [(count, signature) for signature, count in self.memory["endogenous"].items() if count > 0]
        return max(rows, default=(0, None))[-1]

    def _stable(self, exclude=None):
        rows = [(row["count"], signature) for signature, row in self.memory["entities"].items()
                if signature != exclude and signature != self.memory["effector_signature"] and row["count"] >= 2]
        return max(rows, default=(0, None))[-1]

    def _singletons(self):
        growing = self._direct_growing(); stable = self._stable(growing)
        return [signature for signature, row in self.memory["entities"].items()
                if row["count"] == 1 and signature not in (self.memory["effector_signature"], growing, stable)
                and signature not in self.carried and self.memory["sightings"].get(signature, 0) >= 3]

    @staticmethod
    def _pair(values): return "|".join(sorted(values))

    def _choose_pair(self):
        values = sorted(set(self.memory["ingredients"])); failed = set(self.memory["failed_pairs"])
        for index, left in enumerate(values):
            for right in values[index + 1:]:
                if self._pair([left, right]) not in failed: return left, right
        return None

    def _target_position(self, signature, effector):
        positions = [tuple(x) for x in self.memory["entities"].get(signature, {}).get("positions", [])]
        if not positions: return None
        return min(positions, key=lambda point: np.hypot(point[0] - effector[0], point[1] - effector[1]))

    def _approach(self, view, signature, reason):
        effector = self._effector(view)
        if effector is None: return None
        target = (tuple(self.active_target["position"])
                  if self.active_target and self.active_target["signature"] == signature
                  else self._target_position(signature, effector))
        if target is None: return None
        distance = np.hypot(target[0] - effector[0], target[1] - effector[1])
        # Stop before the effector touches the target component.  Interaction
        # at a learned-world proximity keeps the two visual objects separable.
        if distance <= 18.0:
            action = int(self.memory["interact"])
            self.pending = {"kind": "interact", "before": view, "target": signature,
                            "reason": reason, "carried_before": list(self.carried), "pair": self.candidate_pair}
            return action, {"reason": reason, "target": signature, "prediction": "interaction-effect"}
        choices = []
        for action, delta in self.memory["controls"].items():
            nxt = effector[0] + delta[0], effector[1] + delta[1]
            choices.append((np.hypot(target[0] - nxt[0], target[1] - nxt[1]), int(action)))
        action = min(choices)[1]; self.pending = {"kind": "move", "before": view, "action": action}
        self.active_target = {"signature": signature, "position": list(target), "reason": reason}
        return action, {"reason": "navigate:" + reason, "target": signature, "prediction": "reduce-distance"}

    def act(self, frame):
        view = segment_components(frame); self._update(view); self.memory["actions"] += 1
        if not self._ready():
            tried = set(self.memory["tried"]); remaining = [action for action in range(5) if action not in tried]
            if not remaining: raise RuntimeError("could not ground free-form controls")
            action = remaining[0]; self.memory["tried"].append(action)
            self.pending = {"kind": "calibrate", "before": view, "action": action}
            return action, {"reason": "discover-effector-and-control", "prediction": "visual-motion"}

        # Observe long enough to distinguish autonomous population growth from
        # object changes caused by interaction.
        if self._direct_growing() is None and self.probe_steps < 44:
            actions = sorted(int(action) for action in self.memory["controls"])
            action = actions[self.probe_steps % len(actions)]; self.probe_steps += 1
            self.self_generated_subgoals += 1
            self.pending = {"kind": "move", "before": view, "action": action}
            return action, {"reason": "observe-autonomous-change", "prediction": "population-dynamics"}

        if self.active_target is not None:
            choice = self._approach(view, self.active_target["signature"], self.active_target["reason"])
            if choice: return choice
            self.active_target = None

        growing = self._direct_growing()
        if (self.treatment_ready or self.treatment_active) and growing:
            choice = self._approach(view, growing, "safely-test-or-apply-treatment")
            if choice: return choice

        required = None
        if self.memory["successful_pair"]:
            required = self.memory["successful_pair"].split("|")
        else:
            planned = self._choose_pair(); required = list(planned) if planned else None

        if len(self.carried) == 2:
            station = self.memory["station"]
            if station:
                self.candidate_pair = self._pair(self.carried)
                choice = self._approach(view, station, "execute-two-object-transformation")
                if choice: return choice
            for signature in self._singletons():
                if signature not in self.memory["ingredients"] and signature not in self.memory["failed_surfaces"]:
                    self.candidate_pair = self._pair(self.carried)
                    choice = self._approach(view, signature, "discover-transformation-surface")
                    if choice: return choice

        if len(self.carried) < 2:
            targets = []
            if required:
                targets.extend(signature for signature in required
                               if signature not in self.carried and signature in self.memory["entities"])
            candidates = [signature for signature in self._singletons()
                          if signature not in self.memory["noncollectible"] and signature not in targets]
            # If every pair among known ingredients failed, seek an unresolved
            # object instead of rebuilding the same exhausted pair forever.
            if not required:
                unknown = [signature for signature in candidates if signature not in self.memory["ingredients"]]
                candidates = unknown or candidates
            targets.extend(candidates)
            # Learned morphology priors reorder experiments but never install a role.
            targets.sort(key=lambda signature: (
                int(bool(required and signature in required)),
                self.form_roles[self.memory["entities"][signature]["form"]]["collectible"]), reverse=True)
            for signature in targets:
                choice = self._approach(view, signature, "test-or-collect-object")
                if choice: return choice

        # Reversible motion supplies more temporal evidence when hypotheses are exhausted.
        action = min(int(x) for x in self.memory["controls"])
        self.pending = {"kind": "move", "before": view, "action": action}
        return action, {"reason": "continue-temporal-observation", "prediction": "population-dynamics"}

    def observe(self, before_frame, action, after_frame, reward, done):
        before = self.pending.get("before") if self.pending else segment_components(before_frame)
        after = segment_components(after_frame); pending = self.pending or {}; kind = pending.get("kind")
        old_groups, new_groups = _group(before), _group(after)
        if kind == "calibrate":
            candidates = []
            for signature in set(old_groups) & set(new_groups):
                if len(old_groups[signature]) != 1 or len(new_groups[signature]) != 1: continue
                old = old_groups[signature][0]["centroid"]; new = new_groups[signature][0]["centroid"]
                delta = new[0] - old[0], new[1] - old[1]; magnitude = np.hypot(*delta)
                if 4 <= magnitude <= 20: candidates.append((magnitude, signature, delta))
            if candidates:
                _, signature, delta = max(candidates)
                self.memory["effector_signature"] = signature
                self.memory["controls"][str(int(action))] = [float(delta[0]), float(delta[1])]
            if len(self.memory["controls"]) == 4:
                remaining = set(range(5)) - {int(x) for x in self.memory["controls"]}
                if len(remaining) == 1: self.memory["interact"] = remaining.pop()
        elif kind == "move":
            for signature in set(old_groups) | set(new_groups):
                if signature == self.memory["effector_signature"]: continue
                increase = len(new_groups.get(signature, [])) - len(old_groups.get(signature, []))
                if increase > 0:
                    self.memory["endogenous"][signature] = self.memory["endogenous"].get(signature, 0) + increase
        elif kind == "interact":
            self.experiments += 1; target = pending["target"]; reason = pending["reason"]
            changed_target = _changed(before, after, target)
            if reason == "test-or-collect-object":
                if changed_target:
                    if target not in self.memory["ingredients"]: self.memory["ingredients"].append(target)
                    if target not in self.carried: self.carried.append(target)
                    form = self.memory["entities"].get(target, {}).get("form")
                    self.form_roles[form]["collectible"] += 1
                else:
                    appeared = [signature for signature in set(new_groups) - set(old_groups)
                                if signature != self.memory["effector_signature"]]
                    if len(appeared) == 1:
                        ingredient = appeared[0]
                        if ingredient not in self.memory["ingredients"]: self.memory["ingredients"].append(ingredient)
                        if ingredient not in self.carried: self.carried.append(ingredient)
                    elif target not in self.memory["noncollectible"]:
                        self.memory["noncollectible"].append(target)
            elif reason in ("discover-transformation-surface", "execute-two-object-transformation"):
                ingredient_changed = any(_changed(before, after, signature) for signature in pending["carried_before"])
                if ingredient_changed:
                    self.memory["station"] = target; self.candidate_pair = pending.get("pair")
                    self.carried = []; self.treatment_ready = True
                    form = self.memory["entities"].get(target, {}).get("form")
                    self.form_roles[form]["surface"] += 1
                elif target not in self.memory["failed_surfaces"]:
                    self.memory["failed_surfaces"].append(target)
            elif reason == "safely-test-or-apply-treatment":
                old_count = len(old_groups.get(target, [])); new_count = len(new_groups.get(target, []))
                if new_count < old_count:
                    if self.candidate_pair:
                        self.memory["successful_pair"] = self.candidate_pair
                    self.treatment_active = True; self.treatment_ready = False
                elif self.treatment_ready:
                    if self.candidate_pair and self.candidate_pair not in self.memory["failed_pairs"]:
                        self.memory["failed_pairs"].append(self.candidate_pair)
                    self.treatment_ready = False; self.candidate_pair = None
            self.active_target = None
        self._update(after); self.pending = None
        return {"reward": float(reward), "done": bool(done), "growing": self._direct_growing(),
                "stable": self._stable(self._direct_growing()), "carried": list(self.carried)}

    def status(self):
        return {"format": self.format, "remembered_worlds": len(self.worlds),
                "self_generated_subgoals": self.self_generated_subgoals, "experiments": self.experiments,
                "worlds_with_recipes": sum(row.get("successful_pair") is not None for row in self.worlds.values()),
                "discovered_forms": len(self.form_roles)}

    def save(self, path):
        value = self.status() | {"worlds": self.worlds,
            "form_roles": {str(key): dict(rows) for key, rows in self.form_roles.items() if key is not None}}
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format") != cls.format: raise ValueError("unsupported object-centric mind")
        result = cls(); result.worlds = value["worlds"]
        result.form_roles = defaultdict(Counter, {key: Counter(rows) for key, rows in value.get("form_roles", {}).items()})
        result.self_generated_subgoals = int(value.get("self_generated_subgoals", 0))
        result.experiments = int(value.get("experiments", 0)); return result
