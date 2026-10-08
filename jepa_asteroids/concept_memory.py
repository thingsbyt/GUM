"""Growing visual ontology and concept-level causal procedure memory."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math

from .role_schema import CausalRoleSchema


def object_concept(obj: dict) -> str:
    """Position-independent object type induced from visible geometry."""
    area_bin = min(15, int(math.log2(max(1, int(obj["area"])))))
    width_bin = min(15, int(obj["w"])); height_bin = min(15, int(obj["h"]))
    return f"object:c{int(obj['color'])}:a{area_bin}:w{width_bin}:h{height_bin}"


def _direction(a: dict, b: dict) -> str:
    dx, dy = float(b["cx"] - a["cx"]), float(b["cy"] - a["cy"])
    if abs(dx) + abs(dy) <= 6: return "near"
    if abs(dx) >= abs(dy): return "right-of" if dx > 0 else "left-of"
    return "below" if dy > 0 else "above"


def scene_concepts(objects: list[dict]) -> set[str]:
    concepts = {object_concept(obj) for obj in objects}
    # Relations are expressed between learned object types, not coordinates.
    # Bound the quadratic expansion while retaining rare/small entities.
    ranked = sorted(objects, key=lambda obj: (obj["area"], obj["color"]))[:10]
    for index, first in enumerate(ranked):
        for second in ranked[index + 1:]:
            left, right = object_concept(first), object_concept(second)
            concepts.add(f"relation:{left}:{_direction(first, second)}:{right}")
    return concepts


def concept_state_key(objects: list[dict]) -> str:
    payload = json.dumps(sorted(scene_concepts(objects)), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


class ConceptMemory:
    """Induce entities, relations, events, dependencies, and abstract plans."""
    format = "wailah-growing-concepts-v1"

    def __init__(self):
        self.concept_counts = Counter()
        self.event_counts = Counter()
        self.relation_counts = Counter()
        self.action_events = defaultdict(Counter)
        self.contact_attempts = Counter()
        self.contact_effects = defaultdict(Counter)
        self.progress_credit = Counter()
        self.dependencies = defaultdict(Counter)
        self.episode_events: list[str] = []
        self.successful_concept_plans: list[list[str]] = []
        self.observed_this_episode = Counter()
        self.novel_concepts = 0
        self.novel_events = 0
        self.last_events: list[str] = []
        self.role_schema = CausalRoleSchema()

    def reset_episode(self) -> None:
        self.episode_events = []
        self.observed_this_episode = Counter()
        self.last_events = []
        self.role_schema.reset_episode()

    @staticmethod
    def disappearance(concept: str) -> str:
        return f"event:disappeared:{concept}"

    @staticmethod
    def appearance(concept: str) -> str:
        return f"event:appeared:{concept}"

    def note_contact_attempt(self, concept: str) -> None:
        self.contact_attempts[concept] += 1
        self.role_schema.note_contact(concept)

    def _match(self, before: list[dict], after: list[dict]):
        before_groups = defaultdict(list); after_groups = defaultdict(list)
        for obj in before: before_groups[object_concept(obj)].append(obj)
        for obj in after: after_groups[object_concept(obj)].append(obj)
        matched, disappeared, appeared = [], [], []
        for concept in set(before_groups) | set(after_groups):
            old, new = list(before_groups[concept]), list(after_groups[concept])
            while old and new:
                distance, i, j = min((abs(a["cx"] - b["cx"]) + abs(a["cy"] - b["cy"]), i, j)
                                     for i, a in enumerate(old) for j, b in enumerate(new))
                matched.append((concept, old.pop(i), new.pop(j), distance))
            disappeared.extend((concept, obj) for obj in old)
            appeared.extend((concept, obj) for obj in new)
        return matched, disappeared, appeared

    def observe(self, before: list[dict], action: int, after: list[dict], progress: bool) -> list[str]:
        before_scene, after_scene = scene_concepts(before), scene_concepts(after)
        for concept in after_scene:
            if self.concept_counts[concept] == 0: self.novel_concepts += 1
            self.concept_counts[concept] += 1
            if concept.startswith("relation:"): self.relation_counts[concept] += 1
        matched, disappeared, appeared = self._match(before, after)
        events = []
        for concept, old, new, distance in matched:
            if distance >= .5:
                dx = int(math.copysign(1, new["cx"] - old["cx"])) if new["cx"] != old["cx"] else 0
                dy = int(math.copysign(1, new["cy"] - old["cy"])) if new["cy"] != old["cy"] else 0
                events.append(f"event:moved:{concept}:dx{dx}:dy{dy}")
        events.extend(self.disappearance(concept) for concept, _ in disappeared)
        events.extend(self.appearance(concept) for concept, _ in appeared)
        # Relation creation/destruction captures switches, alignment, enclosure,
        # and other global changes that cannot be reduced to object contact.
        for relation in sorted(after_scene - before_scene):
            if relation.startswith("relation:"): events.append(f"event:relation-created:{relation}")
        for relation in sorted(before_scene - after_scene):
            if relation.startswith("relation:"): events.append(f"event:relation-destroyed:{relation}")
        if progress: events.append("event:progress")

        unique_events = list(dict.fromkeys(events))
        for event in unique_events:
            if self.event_counts[event] == 0: self.novel_events += 1
            self.event_counts[event] += 1; self.action_events[int(action)][event] += 1
            self.observed_this_episode[event] += 1
            if event not in self.episode_events: self.episode_events.append(event)
        for event in unique_events:
            if event.startswith("event:disappeared:^"):  # reserved future composite marker
                self.contact_effects[event].update(unique_events)

        if progress:
            object_events = [event for event in self.episode_events
                             if event.startswith(("event:disappeared:", "event:appeared:"))][-8:]
            relation_events = [event for event in self.episode_events
                               if event.startswith(("event:relation-created:",
                                                    "event:relation-destroyed:"))][-8:]
            selected = set(object_events + relation_events)
            causal = [event for event in self.episode_events if event in selected]
            for event in causal: self.progress_credit[event] += 1
            for first, second in zip(causal, causal[1:]): self.dependencies[first][second] += 1
            if causal and causal not in self.successful_concept_plans:
                self.successful_concept_plans.append(causal)
            self.role_schema.learn_from_progress()
        self.last_events = unique_events
        return unique_events

    def target_priority(self, obj: dict) -> tuple[float, str]:
        concept = object_concept(obj); event = self.disappearance(concept)
        progress = 4.0 * self.progress_credit[event]
        prerequisite = 2.0 * sum(self.dependencies[event].values())
        novelty = 2.5 / math.sqrt(1 + self.contact_attempts[concept])
        rarity = 1.0 / math.sqrt(1 + self.concept_counts[concept])
        plan_bonus = 0.0
        for plan in self.successful_concept_plans:
            remaining = [token for token in plan if self.observed_this_episode[token] == 0]
            if remaining and remaining[0] == event: plan_bonus = max(plan_bonus, 12.0)
        role_bonus = self.role_schema.priority(concept)
        return progress + prerequisite + novelty + rarity + plan_bonus + role_bonus, concept

    def status(self) -> dict:
        return {"format": self.format,
                "object_and_relation_concepts": len(self.concept_counts),
                "event_concepts": len(self.event_counts),
                "novel_concepts_discovered": self.novel_concepts,
                "novel_events_discovered": self.novel_events,
                "learned_dependencies": sum(len(row) for row in self.dependencies.values()),
                "successful_concept_plans": len(self.successful_concept_plans),
                "contact_concepts_tested": len(self.contact_attempts),
                "most_credited_events": self.progress_credit.most_common(8),
                "role_schema": self.role_schema.status()}

    def export(self) -> dict:
        return self.status() | {
            "concept_counts": dict(self.concept_counts), "event_counts": dict(self.event_counts),
            "contact_attempts": dict(self.contact_attempts), "progress_credit": dict(self.progress_credit),
            "dependencies": {event: dict(row) for event, row in self.dependencies.items()},
            "successful_concept_plans_data": self.successful_concept_plans,
            "role_schema_data": self.role_schema.export(),
        }

    def restore(self, value: dict) -> None:
        if value.get("format") != self.format: raise ValueError("unsupported concept memory format")
        self.concept_counts = Counter({str(k): int(v) for k, v in value.get("concept_counts", {}).items()})
        self.event_counts = Counter({str(k): int(v) for k, v in value.get("event_counts", {}).items()})
        self.contact_attempts = Counter({str(k): int(v) for k, v in value.get("contact_attempts", {}).items()})
        self.progress_credit = Counter({str(k): int(v) for k, v in value.get("progress_credit", {}).items()})
        self.dependencies = defaultdict(Counter, {str(k): Counter(v)
                                                  for k, v in value.get("dependencies", {}).items()})
        self.successful_concept_plans = [[str(x) for x in row]
                                         for row in value.get("successful_concept_plans_data", [])]
        if value.get("role_schema_data"): self.role_schema.restore(value["role_schema_data"])
