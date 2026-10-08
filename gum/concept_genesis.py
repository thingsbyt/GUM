"""Unsupervised concept formation from raw visual transitions.

The learner is not given event classes or a requested number of concepts.  It
learns a vocabulary of pixel changes, represents each transition as a bag of
those learned visual tokens, and selects its own event partition using an
internal silhouette score.  Separate worlds test later transfer.  Successful
event sequences are compiled into reusable skills.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .storage import atomic_write_json
from sklearn.preprocessing import StandardScaler


ACTION_COUNT = 5


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def frame_id(frame: np.ndarray) -> str:
    value = np.ascontiguousarray(frame)
    return hashlib.sha256(str(value.shape).encode() + value.tobytes()).hexdigest()[:20]


def _connected_mask(rng: np.random.Generator, area: int, height: int, width: int,
                    occupied: np.ndarray) -> np.ndarray:
    """Generate a presentation-specific connected shape with an exact area."""
    for _ in range(200):
        y = int(rng.integers(5, height - 5)); x = int(rng.integers(5, width - 5))
        points = {(y, x)}; frontier = [(y, x)]
        while len(points) < area:
            py, px = frontier[int(rng.integers(len(frontier)))]
            dy, dx = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
            candidate = (int(np.clip(py + dy, 3, height - 4)), int(np.clip(px + dx, 3, width - 4)))
            points.add(candidate); frontier.append(candidate)
        mask = np.zeros((height, width), dtype=bool)
        for py, px in points: mask[py, px] = True
        padded = np.zeros_like(mask)
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                padded |= np.roll(np.roll(mask, dy, 0), dx, 1)
        if not np.any(padded & occupied): return mask
    raise RuntimeError("could not place visual event surface")


class GenesisWorld:
    """Raw-pixel sparse-reward world whose visual mechanisms are never named."""

    def __init__(self, seed: int, target: tuple[int, ...], presentation: int = 0,
                 height: int = 48, width: int = 48):
        if not target or any(value not in range(ACTION_COUNT) for value in target):
            raise ValueError("target contains an invalid hidden mechanism")
        self.seed = int(seed); self.target = tuple(map(int, target)); self.presentation = int(presentation)
        self.height = int(height); self.width = int(width); rng = np.random.default_rng(self.seed)
        self.control_to_mechanism = rng.permutation(ACTION_COUNT).tolist()
        background = rng.integers(52, 116, 3); self.background = background.astype(np.int16)
        occupied = np.zeros((height, width), dtype=bool)
        areas = (20, 44, 20, 44, 20, 20)
        masks = []
        for area in areas:
            mask = _connected_mask(rng, area, height, width, occupied)
            occupied |= mask; masks.append(mask)
        self.effects = {
            0: ((masks[0], 34),),
            1: ((masks[1], 34),),
            2: ((masks[2], -34),),
            3: ((masks[3], -34),),
            4: ((masks[4], 34), (masks[5], -34)),
        }
        self.channel_order = rng.permutation(3)
        self.world_id = hashlib.sha256(_canonical({"seed": self.seed, "presentation": self.presentation,
            "horizon": len(self.target), "shape": [height, width]})).hexdigest()[:16]
        self.reset()

    @property
    def public_spec(self) -> dict:
        return {"world_id": self.world_id, "observation": "raw-rgb-pixels",
                "action_count": ACTION_COUNT, "horizon": len(self.target), "reward": "terminal-scalar"}

    def reset(self) -> np.ndarray:
        self.canvas = np.broadcast_to(self.background, (self.height, self.width, 3)).copy()
        self.history = []; self.steps = 0
        return self.render()

    def render(self) -> np.ndarray:
        frame = np.clip(self.canvas[:, :, self.channel_order], 0, 255).astype(np.uint8)
        return np.moveaxis(frame, -1, 0)

    def step(self, action: int):
        action = int(action)
        if action not in range(ACTION_COUNT): raise ValueError("invalid anonymous action")
        mechanism = int(self.control_to_mechanism[action])
        for mask, amount in self.effects[mechanism]: self.canvas[mask] += int(amount)
        self.canvas = np.clip(self.canvas, 0, 255)
        self.history.append(mechanism); self.steps += 1
        done = self.steps >= len(self.target); success = bool(done and tuple(self.history) == self.target)
        return self.render(), (1.0 if success else 0.0), done, {"success": success}

    def audit(self) -> dict:
        return {"world_id": self.world_id, "target_mechanisms": list(self.target),
                "control_to_mechanism": list(self.control_to_mechanism),
                "presentation": self.presentation}


class GenesisEncoder:
    """Two-level unsupervised visual vocabulary selected without class labels."""

    format = "gum-genesis-encoder-v1"

    def __init__(self, seed: int = 73):
        self.seed = int(seed); self.pixel_centers = None; self.event_centers = None
        self.event_mean = None; self.event_scale = None; self.selection = []

    @staticmethod
    def _delta(before, after) -> np.ndarray:
        left = np.moveaxis(np.asarray(before, dtype=np.float32), 0, -1)
        right = np.moveaxis(np.asarray(after, dtype=np.float32), 0, -1)
        delta = (right - left).reshape(-1, 3)
        changed = np.max(np.abs(delta), axis=1) >= 3.0
        return delta[changed] / 255.0

    @staticmethod
    def _choose_partition(values: np.ndarray, minimum: int, maximum: int, seed: int):
        rows = []
        upper = min(int(maximum), len(values) - 1, len(np.unique(np.round(values, 6), axis=0)))
        for count in range(int(minimum), upper + 1):
            model = KMeans(n_clusters=count, n_init=24, random_state=seed + count).fit(values)
            labels = model.labels_
            if len(np.unique(labels)) != count: continue
            score = float(silhouette_score(values, labels))
            rows.append({"clusters": count, "silhouette": score, "model": model})
        if not rows: raise RuntimeError("not enough distinct visual evidence to form concepts")
        # Prefer the simpler explanation when scores are effectively tied.
        best_score = max(row["silhouette"] for row in rows)
        eligible = [row for row in rows if row["silhouette"] >= best_score - 0.002]
        best = min(eligible, key=lambda row: row["clusters"])
        return best, [{"clusters": row["clusters"], "silhouette": row["silhouette"]} for row in rows]

    def fit(self, transitions: list[tuple[np.ndarray, np.ndarray]]) -> dict:
        if len(transitions) < 10: raise ValueError("more independent transitions are required")
        pixel_rows = [self._delta(before, after) for before, after in transitions]
        if any(len(row) == 0 for row in pixel_rows): raise ValueError("an observed transition contained no visual evidence")
        pixels = np.concatenate(pixel_rows, axis=0)
        pixel_best, pixel_scores = self._choose_partition(pixels, 2, 8, self.seed)
        self.pixel_centers = pixel_best["model"].cluster_centers_
        event_features = np.stack([self._event_feature(row) for row in pixel_rows])
        scaler = StandardScaler().fit(event_features); standardized = scaler.transform(event_features)
        event_best, event_scores = self._choose_partition(standardized, 2, 9, self.seed + 1000)
        self.event_mean = scaler.mean_; self.event_scale = scaler.scale_
        self.event_centers = event_best["model"].cluster_centers_
        self.selection = {"pixel_candidates": pixel_scores, "event_candidates": event_scores,
                          "selected_pixel_tokens": len(self.pixel_centers),
                          "selected_event_concepts": len(self.event_centers)}
        return dict(self.selection)

    def _pixel_labels(self, pixels: np.ndarray) -> np.ndarray:
        distances = ((pixels[:, None, :] - self.pixel_centers[None, :, :]) ** 2).sum(2)
        return np.argmin(distances, axis=1)

    def _event_feature(self, pixels: np.ndarray) -> np.ndarray:
        labels = self._pixel_labels(pixels); counts = np.bincount(labels, minlength=len(self.pixel_centers)).astype(float)
        # These are generic distribution statistics, not named event types.
        magnitudes = np.linalg.norm(pixels, axis=1)
        return np.concatenate([counts, [len(pixels), magnitudes.mean(), magnitudes.std()]])

    def encode(self, before, after) -> str:
        if self.event_centers is None: raise RuntimeError("visual vocabulary has not been learned")
        feature = self._event_feature(self._delta(before, after))
        standardized = (feature - self.event_mean) / self.event_scale
        index = int(np.argmin(((self.event_centers - standardized[None, :]) ** 2).sum(1)))
        center = np.round(self.event_centers[index], 6).tolist()
        return "genesis-" + hashlib.sha256(_canonical(center)).hexdigest()[:12]

    @property
    def concept_ids(self) -> list[str]:
        if self.event_centers is None: return []
        return ["genesis-" + hashlib.sha256(_canonical(np.round(center, 6).tolist())).hexdigest()[:12]
                for center in self.event_centers]

    def to_json(self) -> dict:
        return {"format": self.format, "seed": self.seed,
                "pixel_centers": self.pixel_centers.tolist(), "event_centers": self.event_centers.tolist(),
                "event_mean": self.event_mean.tolist(), "event_scale": self.event_scale.tolist(),
                "selection": self.selection}

    @classmethod
    def from_json(cls, value: dict):
        if value.get("format") != cls.format: raise ValueError("unsupported genesis encoder")
        result = cls(value["seed"]); result.pixel_centers = np.asarray(value["pixel_centers"], dtype=float)
        result.event_centers = np.asarray(value["event_centers"], dtype=float)
        result.event_mean = np.asarray(value["event_mean"], dtype=float)
        result.event_scale = np.asarray(value["event_scale"], dtype=float)
        result.selection = dict(value["selection"]); return result


@dataclass
class GenesisSkill:
    skill_id: str
    sequence: tuple[str, ...]
    support: int = 0
    successes: int = 0
    failures: int = 0
    contexts: set[str] = field(default_factory=set)

    @property
    def confidence(self): return (self.successes + self.support) / max(1, self.successes + self.failures + self.support)

    def to_json(self):
        return {"skill_id": self.skill_id, "sequence": list(self.sequence), "support": self.support,
                "successes": self.successes, "failures": self.failures, "confidence": self.confidence,
                "contexts": sorted(self.contexts)}


class ConceptGenesisMind:
    format = "gum-concept-genesis-mind-v1"

    def __init__(self, encoder: GenesisEncoder | None = None):
        self.encoder = encoder or GenesisEncoder(); self.skills: dict[str, GenesisSkill] = {}
        self.contexts = {}; self.concept_evidence = {key: 0 for key in self.encoder.concept_ids}
        self.interactions = 0; self.created = len(self.encoder.concept_ids); self.merges = 0; self.splits = 0

    @staticmethod
    def context_id(world: GenesisWorld) -> str:
        return hashlib.sha256(_canonical(world.public_spec) + world.reset().tobytes()).hexdigest()[:20]

    def discover(self, worlds: Iterable[GenesisWorld]) -> dict:
        transitions = []
        for world in worlds:
            for action in range(ACTION_COUNT):
                before = world.reset(); after, _, _, _ = world.step(action)
                transitions.append((before, after)); self.interactions += 1
        selection = self.encoder.fit(transitions); self.created = len(self.encoder.concept_ids)
        self.concept_evidence = {key: 0 for key in self.encoder.concept_ids}
        for before, after in transitions: self.concept_evidence[self.encoder.encode(before, after)] += 1
        return {"transitions": len(transitions), **selection,
                "concept_support": dict(sorted(self.concept_evidence.items()))}

    def ground(self, world: GenesisWorld, trace=None):
        context = self.context_id(world); mapping = {}
        for action in range(ACTION_COUNT):
            before = world.reset(); after, reward, done, _ = world.step(action)
            if trace: trace(before, action, after, reward, done)
            concept = self.encoder.encode(before, after); mapping[action] = concept
            self.concept_evidence[concept] = self.concept_evidence.get(concept, 0) + 1; self.interactions += 1
        if len(set(mapping.values())) != ACTION_COUNT:
            raise RuntimeError("learned concepts do not distinguish the available mechanisms")
        return context, mapping, ACTION_COUNT

    @staticmethod
    def _run(world, actions, trace=None):
        observation = world.reset(); steps = 0; reward = 0.0; done = False
        for action in actions:
            before = observation; observation, reward, done, _ = world.step(int(action)); steps += 1
            if trace: trace(before, int(action), observation, reward, done)
            if done: break
        return bool(done and reward > 0), steps

    def _compile(self, sequence, context):
        sequence = tuple(sequence); skill_id = "genesis-skill-" + hashlib.sha256(_canonical(sequence)).hexdigest()[:12]
        skill = self.skills.get(skill_id)
        if skill is None: skill = GenesisSkill(skill_id, sequence); self.skills[skill_id] = skill
        skill.support += 1; skill.contexts.add(context); return skill

    @staticmethod
    def _cancelled(cancel) -> bool:
        if cancel is None: return False
        return bool(cancel.is_set() if hasattr(cancel, "is_set") else cancel())

    def acquire(self, world: GenesisWorld, trace=None, *, max_candidates: int = 100_000,
                max_interactions: int = 500_000, cancel=None):
        """Acquire a skill within explicit search and interaction budgets."""
        if int(max_candidates) < 1 or int(max_interactions) < 1:
            raise ValueError("acquisition budgets must be positive")
        if self._cancelled(cancel):
            return {"success": False, "reason": "cancelled", "interactions": 0, "candidates": 0}
        if int(max_interactions) < ACTION_COUNT:
            return {"success": False, "reason": "interaction-budget", "interactions": 0, "candidates": 0}
        try: context, mapping, interactions = self.ground(world, trace)
        except RuntimeError as error:
            return {"success": False, "reason": "unsupported-world", "detail": str(error),
                    "interactions": 0, "candidates": 0}
        length = world.public_spec["horizon"]
        for tried, candidate in enumerate(itertools.product(range(ACTION_COUNT), repeat=length), 1):
            if self._cancelled(cancel):
                return {"success": False, "reason": "cancelled", "context": context,
                        "interactions": interactions, "candidates": tried - 1}
            if tried > int(max_candidates):
                return {"success": False, "reason": "candidate-budget", "context": context,
                        "interactions": interactions, "candidates": tried - 1}
            if interactions + length > int(max_interactions):
                return {"success": False, "reason": "interaction-budget", "context": context,
                        "interactions": interactions, "candidates": tried - 1}
            success, steps = self._run(world, candidate, trace); interactions += steps; self.interactions += steps
            if success:
                skill = self._compile([mapping[action] for action in candidate], context); skill.successes += 1
                self.contexts[context] = {"mapping": {str(k): v for k, v in mapping.items()}, "skill_id": skill.skill_id}
                return {"success": True, "context": context, "skill_id": skill.skill_id,
                        "interactions": interactions, "candidates": tried, "reason": "learned"}
        return {"success": False, "reason": "search-exhausted", "context": context,
                "interactions": interactions, "candidates": ACTION_COUNT ** length}

    def solve(self, world: GenesisWorld, trace=None):
        try: context, mapping, interactions = self.ground(world, trace)
        except RuntimeError as error:
            return {"success": False, "reason": "unsupported-world", "detail": str(error), "interactions": 0}
        inverse = {v: k for k, v in mapping.items()}; tried = 0; incompatible = 0
        for skill in sorted(self.skills.values(), key=lambda row: (-row.confidence, row.skill_id)):
            if len(skill.sequence) != world.public_spec["horizon"]: continue
            if any(value not in inverse for value in skill.sequence): incompatible += 1; continue
            success, steps = self._run(world, [inverse[value] for value in skill.sequence], trace)
            interactions += steps; self.interactions += steps; tried += 1
            skill.successes += int(success); skill.failures += int(not success)
            if success:
                self.contexts[context] = {"mapping": {str(k): v for k, v in mapping.items()}, "skill_id": skill.skill_id}
                return {"success": True, "context": context, "skill_id": skill.skill_id,
                        "interactions": interactions, "skills_tried": tried}
        reason = "unknown-concept" if incompatible and not tried else "no-compatible-skill"
        return {"success": False, "reason": reason, "context": context,
                "interactions": interactions, "skills_tried": tried, "incompatible_skills": incompatible}

    def revisit(self, world: GenesisWorld, trace=None):
        context = self.context_id(world); memory = self.contexts.get(context)
        if memory is None: return self.solve(world, trace)
        inverse = {concept: int(action) for action, concept in memory["mapping"].items()}
        skill = self.skills.get(memory["skill_id"])
        if skill is None or any(value not in inverse for value in skill.sequence):
            return {"success": False, "reason": "stale-memory", "context": context, "interactions": 0}
        success, steps = self._run(world, [inverse[x] for x in skill.sequence], trace)
        self.interactions += steps; skill.successes += int(success); skill.failures += int(not success)
        return {"success": success, "context": context, "skill_id": skill.skill_id, "interactions": steps}

    def compose(self, world: GenesisWorld, trace=None):
        try: context, mapping, interactions = self.ground(world, trace)
        except RuntimeError as error:
            return {"success": False, "reason": "unsupported-world", "detail": str(error), "interactions": 0}
        inverse = {v: k for k, v in mapping.items()}; tried = 0
        base = [row for row in sorted(self.skills.values(), key=lambda row: row.skill_id)
                if len(row.sequence) * 2 == world.public_spec["horizon"]]
        for left, right in itertools.product(base, repeat=2):
            sequence = left.sequence + right.sequence
            if any(value not in inverse for value in sequence): continue
            success, steps = self._run(world, [inverse[x] for x in sequence], trace)
            interactions += steps; self.interactions += steps; tried += 1
            if success:
                skill = self._compile(sequence, context); skill.successes += 1
                self.contexts[context] = {"mapping": {str(k): v for k, v in mapping.items()}, "skill_id": skill.skill_id}
                return {"success": True, "context": context, "skill_id": skill.skill_id,
                        "interactions": interactions, "compositions_tried": tried,
                        "parents": [left.skill_id, right.skill_id]}
        return {"success": False, "reason": "no-compatible-composition", "context": context, "interactions": interactions,
                "compositions_tried": tried}

    def status(self):
        return {"format": self.format, "concepts_created": self.created, "concept_merges": self.merges,
                "concept_splits": self.splits, "concept_evidence": dict(sorted(self.concept_evidence.items())),
                "skills_created": len(self.skills), "known_contexts": len(self.contexts),
                "interactions": self.interactions, "encoder_selection": self.encoder.selection,
                "skills": [row.to_json() for row in sorted(self.skills.values(), key=lambda row: row.skill_id)]}

    def save(self, path: Path):
        value = self.status() | {"encoder": self.encoder.to_json(), "contexts": self.contexts}
        atomic_write_json(Path(path), value, sort_keys=True)

    @classmethod
    def load(cls, path: Path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format") != cls.format: raise ValueError("unsupported concept genesis mind")
        result = cls(GenesisEncoder.from_json(value["encoder"])); result.created = int(value["concepts_created"])
        result.merges = int(value["concept_merges"]); result.splits = int(value["concept_splits"])
        result.concept_evidence = {k: int(v) for k, v in value["concept_evidence"].items()}
        result.contexts = dict(value["contexts"]); result.interactions = int(value["interactions"])
        for row in value.get("skills", []):
            skill = GenesisSkill(row["skill_id"], tuple(row["sequence"]), int(row["support"]),
                                 int(row["successes"]), int(row["failures"]), set(row["contexts"]))
            result.skills[skill.skill_id] = skill
        return result
