"""Autonomous opaque concept creation and reusable skill compilation.

The learner sees rendered frames, anonymous actions, terminal reward, and done.
It invents identifiers for stable structural transition signatures, grounds local
controls by intervention, and compiles successful concept sequences as skills.
No human semantic names are stored in learner memory.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .storage import atomic_write_json


ACTION_COUNT = 5
SLOTS = 10
ROWS, COLS, CELL, GAP = 2, 5, 5, 1


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def frame_id(frame: np.ndarray) -> str:
    value = np.ascontiguousarray(frame)
    return hashlib.sha256(str(value.shape).encode() + value.tobytes()).hexdigest()[:20]


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    remaining = set(map(tuple, np.argwhere(mask)))
    components = []
    while remaining:
        root = remaining.pop(); stack = [root]; component = [root]
        while stack:
            y, x = stack.pop()
            for point in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if point in remaining:
                    remaining.remove(point); stack.append(point); component.append(point)
        components.append(component)
    return components


def structural_signature(before: np.ndarray, after: np.ndarray) -> tuple[int, int]:
    """Identity-free visual-change signature: brighter and darker components."""
    left = np.moveaxis(np.asarray(before, dtype=np.float32), 0, -1)
    right = np.moveaxis(np.asarray(after, dtype=np.float32), 0, -1)
    luminance = right.mean(2) - left.mean(2)
    changed = np.abs(luminance) >= 7.0
    components = _connected_components(changed)
    positive = negative = 0
    for component in components:
        value = float(np.mean([luminance[y, x] for y, x in component]))
        if value > 0: positive += 1
        elif value < 0: negative += 1
    if positive + negative == 0:
        raise ValueError("transition contained no stable visual change")
    return positive, negative


class MechanismWorld:
    """Surface-randomized sparse-reward world with hidden causal operators."""

    def __init__(self, seed: int, target: tuple[int, ...], presentation: int):
        if not target or any(value not in range(ACTION_COUNT) for value in target):
            raise ValueError("target must contain hidden primitive indices")
        self.seed = int(seed); self.target = tuple(map(int, target)); self.presentation = int(presentation)
        rng = np.random.default_rng(self.seed)
        self.control_to_operator = rng.permutation(ACTION_COUNT).tolist()
        self.surface_permutation = rng.permutation(SLOTS).tolist()
        raw = rng.uniform(.45, 1.0, 3); self.tint = raw / raw.max()
        self.world_id = hashlib.sha256(_canonical({"seed": self.seed, "presentation": self.presentation,
            "length": len(self.target)})).hexdigest()[:16]
        self.history = []; self.state = np.full(SLOTS, 6, dtype=np.int16); self.steps = 0

    @property
    def public_spec(self) -> dict:
        return {"world_id": self.world_id, "observation": "rgb-pixels", "action_count": ACTION_COUNT,
                "horizon": len(self.target), "reward": "terminal-scalar"}

    def reset(self) -> np.ndarray:
        self.history = []; self.state = np.full(SLOTS, 6, dtype=np.int16); self.steps = 0
        return self.render()

    def _apply(self, operator: int) -> None:
        if operator == 0: self.state[0] += 1
        elif operator == 1: self.state[[1, 2]] += 1
        elif operator == 2: self.state[3] -= 1
        elif operator == 3: self.state[[4, 5]] -= 1
        elif operator == 4:
            self.state[6] += 1; self.state[7] -= 1
        else: raise ValueError(operator)

    def step(self, action: int):
        action = int(action)
        if action not in range(ACTION_COUNT): raise ValueError("invalid anonymous action")
        operator = int(self.control_to_operator[action]); self._apply(operator)
        self.history.append(operator); self.steps += 1
        done = self.steps >= len(self.target); success = bool(done and tuple(self.history) == self.target)
        return self.render(), (1.0 if success else 0.0), done, {"success": success}

    def render(self) -> np.ndarray:
        height = ROWS * CELL + (ROWS + 1) * GAP
        width = COLS * CELL + (COLS + 1) * GAP
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        visible = self.state[np.asarray(self.surface_permutation)]
        for slot, value in enumerate(visible):
            row, col = divmod(slot, COLS); y = GAP + row * (CELL + GAP); x = GAP + col * (CELL + GAP)
            brightness = int(np.clip(18 + int(value) * 18, 0, 250))
            color = np.clip(self.tint * brightness, 0, 255).astype(np.uint8)
            if self.presentation % 3 == 1: color = color[[1, 2, 0]]
            elif self.presentation % 3 == 2: color = color[[2, 0, 1]]
            frame[y:y + CELL, x:x + CELL] = color
        return np.moveaxis(frame, -1, 0)

    def audit(self) -> dict:
        return {"world_id": self.world_id, "target_operators": list(self.target),
                "control_to_operator": list(self.control_to_operator),
                "surface_permutation": list(self.surface_permutation), "presentation": self.presentation}


@dataclass
class ConceptRecord:
    concept_id: str
    signature: tuple[int, int]
    evidence: int = 0
    contexts: set[str] = field(default_factory=set)

    def to_json(self):
        return {"concept_id": self.concept_id, "signature": list(self.signature),
                "evidence": self.evidence, "contexts": sorted(self.contexts)}


@dataclass
class SkillRecord:
    skill_id: str
    sequence: tuple[str, ...]
    support: int = 0
    successful_uses: int = 0
    failed_uses: int = 0
    source_contexts: set[str] = field(default_factory=set)

    @property
    def confidence(self) -> float:
        return (self.successful_uses + self.support) / max(1, self.successful_uses + self.failed_uses + self.support)

    def to_json(self):
        return {"skill_id": self.skill_id, "sequence": list(self.sequence), "support": self.support,
                "successful_uses": self.successful_uses, "failed_uses": self.failed_uses,
                "confidence": self.confidence, "source_contexts": sorted(self.source_contexts)}


class ConceptSkillFactory:
    """Growing memory of self-created transition concepts and executable skills."""

    format = "gum-concept-skill-factory-v1"

    def __init__(self):
        self.concepts: dict[tuple[int, int], ConceptRecord] = {}
        self.skills: dict[str, SkillRecord] = {}
        self.context_memory: dict[str, dict] = {}
        self.concept_events = []; self.skill_events = []
        self.mapping_conflicts = 0; self.mapping_branches = 0; self.interactions = 0

    @staticmethod
    def context_id(world: MechanismWorld) -> str:
        return hashlib.sha256(_canonical(world.public_spec) + world.reset().tobytes()).hexdigest()[:20]

    def observe_concept(self, before, after, context: str) -> str:
        signature = structural_signature(before, after)
        record = self.concepts.get(signature)
        if record is None:
            concept_id = "concept-" + hashlib.sha256(_canonical(signature)).hexdigest()[:10]
            record = ConceptRecord(concept_id, signature)
            self.concepts[signature] = record
            self.concept_events.append({"event": "concept-created", "concept_id": concept_id,
                                        "structural_signature": list(signature), "context": context})
        record.evidence += 1; record.contexts.add(context); return record.concept_id

    def ground(self, world: MechanismWorld, trace=None) -> tuple[str, dict[int, str], int]:
        context = self.context_id(world); learned = {}
        for action in range(ACTION_COUNT):
            before = world.reset(); after, reward, done, _ = world.step(action)
            if trace is not None: trace(before, action, after, reward, done)
            learned[action] = self.observe_concept(before, after, context); self.interactions += 1
        if len(set(learned.values())) != ACTION_COUNT:
            raise RuntimeError("visual interventions did not produce distinct concepts")
        previous = self.context_memory.get(context, {}).get("action_to_concept")
        if previous and {int(k): v for k, v in previous.items()} != learned:
            self.mapping_conflicts += 1; self.mapping_branches += 1
        return context, learned, ACTION_COUNT

    def _compile(self, sequence: Iterable[str], context: str) -> SkillRecord:
        sequence = tuple(sequence); skill_id = "skill-" + hashlib.sha256(_canonical(sequence)).hexdigest()[:10]
        record = self.skills.get(skill_id)
        if record is None:
            record = SkillRecord(skill_id, sequence); self.skills[skill_id] = record
            self.skill_events.append({"event": "skill-created", "skill_id": skill_id,
                                      "concept_sequence": list(sequence), "context": context})
        record.support += 1; record.source_contexts.add(context); return record

    @staticmethod
    def _run(world: MechanismWorld, actions: Iterable[int], trace=None):
        observation = world.reset(); steps = 0; reward = 0.0; done = False
        for action in actions:
            before = observation; observation, reward, done, info = world.step(int(action)); steps += 1
            if trace is not None: trace(before, int(action), observation, reward, done)
            if done: break
        return bool(done and reward > 0), steps

    def acquire(self, world: MechanismWorld, *, max_candidates: int | None = None, trace=None) -> dict:
        context, mapping, interactions = self.ground(world, trace=trace); length = int(world.public_spec["horizon"])
        limit = ACTION_COUNT ** length if max_candidates is None else int(max_candidates)
        tried = 0
        for candidate in itertools.islice(itertools.product(range(ACTION_COUNT), repeat=length), limit):
            success, steps = self._run(world, candidate, trace=trace); tried += 1; interactions += steps; self.interactions += steps
            if success:
                sequence = tuple(mapping[action] for action in candidate)
                skill = self._compile(sequence, context); skill.successful_uses += 1
                self.context_memory[context] = {"action_to_concept": {str(k): v for k, v in mapping.items()},
                                                "skill_id": skill.skill_id}
                return {"success": True, "context": context, "skill_id": skill.skill_id,
                        "interactions": interactions, "candidates": tried, "sequence": list(sequence)}
        return {"success": False, "context": context, "interactions": interactions, "candidates": tried}

    @staticmethod
    def _inverse(mapping: dict[int, str]) -> dict[str, int]:
        return {concept: action for action, concept in mapping.items()}

    def revisit(self, world: MechanismWorld, trace=None) -> dict:
        context = self.context_id(world); memory = self.context_memory.get(context)
        if memory is None: return self.solve_with_skills(world, trace=trace)
        mapping = {int(k): v for k, v in memory["action_to_concept"].items()}; inverse = self._inverse(mapping)
        skill = self.skills[memory["skill_id"]]; actions = [inverse[value] for value in skill.sequence]
        success, steps = self._run(world, actions, trace=trace); self.interactions += steps
        skill.successful_uses += int(success); skill.failed_uses += int(not success)
        return {"success": success, "context": context, "skill_id": skill.skill_id,
                "interactions": steps, "grounding_interactions": 0, "skills_tried": 1}

    def solve_with_skills(self, world: MechanismWorld, trace=None) -> dict:
        context, mapping, interactions = self.ground(world, trace=trace); inverse = self._inverse(mapping); tried = 0
        for skill in sorted(self.skills.values(), key=lambda row: (-row.confidence, row.skill_id)):
            if len(skill.sequence) != world.public_spec["horizon"]: continue
            actions = [inverse[value] for value in skill.sequence]; success, steps = self._run(world, actions, trace=trace)
            tried += 1; interactions += steps; self.interactions += steps
            skill.successful_uses += int(success); skill.failed_uses += int(not success)
            if success:
                self.context_memory[context] = {"action_to_concept": {str(k): v for k, v in mapping.items()},
                                                "skill_id": skill.skill_id}
                return {"success": True, "context": context, "skill_id": skill.skill_id,
                        "interactions": interactions, "grounding_interactions": ACTION_COUNT,
                        "skills_tried": tried}
        return {"success": False, "context": context, "interactions": interactions,
                "grounding_interactions": ACTION_COUNT, "skills_tried": tried}

    def solve_composed(self, world: MechanismWorld, *, trace=None) -> dict:
        context, mapping, interactions = self.ground(world, trace=trace); inverse = self._inverse(mapping); tried = 0
        base = [row for row in sorted(self.skills.values(), key=lambda row: row.skill_id)
                if len(row.sequence) * 2 == world.public_spec["horizon"]]
        for left, right in itertools.product(base, repeat=2):
            sequence = left.sequence + right.sequence; actions = [inverse[value] for value in sequence]
            success, steps = self._run(world, actions, trace=trace); tried += 1; interactions += steps; self.interactions += steps
            if success:
                skill = self._compile(sequence, context); skill.successful_uses += 1
                self.context_memory[context] = {"action_to_concept": {str(k): v for k, v in mapping.items()},
                                                "skill_id": skill.skill_id}
                return {"success": True, "context": context, "skill_id": skill.skill_id,
                        "interactions": interactions, "grounding_interactions": ACTION_COUNT,
                        "compositions_tried": tried, "parents": [left.skill_id, right.skill_id]}
        return {"success": False, "context": context, "interactions": interactions,
                "grounding_interactions": ACTION_COUNT, "compositions_tried": tried}

    def share_portable_knowledge(self) -> dict:
        return {"format": "gum-portable-concept-skills-v1",
                "concepts": [row.to_json() for row in self.concepts.values()],
                "skills": [row.to_json() for row in self.skills.values()]}

    def import_portable_knowledge(self, payload: dict) -> None:
        if payload.get("format") != "gum-portable-concept-skills-v1": raise ValueError("bad portable memory")
        for row in payload.get("concepts", []):
            signature = tuple(map(int, row["signature"])); current = self.concepts.get(signature)
            if current is None:
                current = ConceptRecord(row["concept_id"], signature); self.concepts[signature] = current
            current.evidence = max(current.evidence, int(row["evidence"])); current.contexts.update(row["contexts"])
        for row in payload.get("skills", []):
            current = self.skills.get(row["skill_id"])
            if current is None:
                current = SkillRecord(row["skill_id"], tuple(row["sequence"])); self.skills[current.skill_id] = current
            current.support = max(current.support, int(row["support"])); current.source_contexts.update(row["source_contexts"])

    def status(self) -> dict:
        return {"format": self.format, "concepts_created": len(self.concepts), "skills_created": len(self.skills),
                "known_contexts": len(self.context_memory), "concept_evidence": sum(x.evidence for x in self.concepts.values()),
                "interactions": self.interactions, "mapping_conflicts": self.mapping_conflicts,
                "mapping_branches": self.mapping_branches,
                "concepts": [row.to_json() for row in sorted(self.concepts.values(), key=lambda x: x.concept_id)],
                "skills": [row.to_json() for row in sorted(self.skills.values(), key=lambda x: x.skill_id)]}

    def save(self, path: Path) -> None:
        value = self.status() | {"context_memory": self.context_memory,
            "concept_events": self.concept_events, "skill_events": self.skill_events}
        atomic_write_json(Path(path), value, sort_keys=True)

    @classmethod
    def load(cls, path: Path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format") != cls.format: raise ValueError("unsupported concept-skill memory")
        result = cls()
        for row in value.get("concepts", []):
            record = ConceptRecord(row["concept_id"], tuple(row["signature"]), int(row["evidence"]), set(row["contexts"]))
            result.concepts[record.signature] = record
        for row in value.get("skills", []):
            record = SkillRecord(row["skill_id"], tuple(row["sequence"]), int(row["support"]),
                                 int(row["successful_uses"]), int(row["failed_uses"]), set(row["source_contexts"]))
            result.skills[record.skill_id] = record
        result.context_memory = dict(value.get("context_memory", {})); result.concept_events = list(value.get("concept_events", []))
        result.skill_events = list(value.get("skill_events", [])); result.interactions = int(value.get("interactions", 0))
        result.mapping_conflicts = int(value.get("mapping_conflicts", 0)); result.mapping_branches = int(value.get("mapping_branches", 0))
        return result
