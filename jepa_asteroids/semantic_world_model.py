"""Shared semantic causal theory and active cross-context transfer.

The model compresses color- and identity-specific pixel events into structural
event kinds, learns event grammars that preceded real progress, and carries
those grammars across otherwise isolated world memories.  In a new context it
performs interventions to identify which local controls produce the next
abstract event, rather than transferring numeric actions or coordinates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re


_MOVE = re.compile(r":dx(-?\d+):dy(-?\d+)$")


def semantic_event(event: str) -> str | None:
    """Map a grounded visual event to an identity-free causal predicate."""
    if event.startswith("event:semantic:"):
        # Extensible sensors may publish a grounded domain predicate while
        # retaining the same event channel used by visual perception.
        value = event.split(":", 3)[2]
        return value or None
    if event.startswith("event:appeared:"):
        return "entity-appeared"
    if event.startswith("event:disappeared:"):
        return "entity-disappeared"
    if event.startswith("event:moved:"):
        match = _MOVE.search(event)
        if not match:
            return "entity-moved"
        dx, dy = map(int, match.groups())
        axis = "horizontal" if dx else "vertical" if dy else "stationary"
        return f"entity-moved-{axis}"
    if event.startswith("event:relation-created:"):
        return "relation-created"
    if event.startswith("event:relation-destroyed:"):
        return "relation-destroyed"
    return None


def _transition_semantics(events: list[str]) -> list[str]:
    values = [semantic_event(event) for event in events]
    values = [value for value in values if value is not None]
    # Object persistence changes are more stable than the many relations they
    # incidentally create or destroy. Motion remains useful when it is the only
    # structural event in the transition.
    persistent = [value for value in values
                  if value in ("entity-appeared", "entity-disappeared")]
    if persistent:
        return list(dict.fromkeys(persistent))
    motion = [value for value in values if value.startswith("entity-moved")]
    if motion:
        return list(dict.fromkeys(motion))
    return list(dict.fromkeys(values))[:1]


class SharedSemanticWorldModel:
    """Learn reusable causal event grammars and ground them by intervention."""

    format = "wailah-shared-semantic-world-model-v1"

    def __init__(self):
        self.templates = Counter()
        self.context_attempts = defaultdict(lambda: defaultdict(Counter))
        self.context_effects = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
        self.context_matches = Counter(); self.context_mismatches = Counter()
        self.context_progress = Counter()
        self.templates_discovered = 0; self.transfer_actions = 0
        self.counterfactual_queries = 0; self.correct_predictions = 0
        self.predictions = 0; self.active_context = "default"
        self.reset_episode("default")

    @staticmethod
    def _phase_key(phase: int) -> str:
        return str(int(phase))

    def _best_template(self) -> tuple[str, ...] | None:
        if not self.templates:
            return None
        return max(self.templates, key=lambda row: (self.templates[row], len(row), row))

    def reset_episode(self, context: str) -> None:
        self.active_context = str(context); self.phase = 0
        self.episode_semantics = []; self.pending = None; self.calls = 0

    def recommend(self, context: str, actions: list[int]):
        context = str(context); actions = sorted(map(int, actions)); self.calls += 1
        template = self._best_template()
        # One lucky success is evidence worth remembering, but not enough to
        # steer behavior. Transfer earns control only after the same abstract
        # suffix independently precedes progress at least twice.
        if not template or self.templates[template] < 2 or not actions:
            return None
        if self.phase >= len(template):
            self.phase = 0
        # Reject a transferred theory after repeated clean contradictions. An
        # ungrounded theory receives sparse probes until the first match; a
        # grounded theory may guide continuously.
        matches, mismatches = self.context_matches[context], self.context_mismatches[context]
        # Give every available intervention two chances before rejecting an
        # imported theory.  Otherwise a correct mapping assigned to the last
        # control tested would be rejected before it could be discovered.
        if matches == 0 and mismatches >= max(4, 2 * len(actions)):
            return None
        if matches == 0 and self.calls % 5:
            return None
        target = template[self.phase]; phase = self._phase_key(self.phase)
        rows = self.context_effects[context][phase]
        scored = []
        for action in actions:
            outcomes = rows[action]; total = sum(outcomes.values())
            target_hits = outcomes[target]
            probability = target_hits / max(1, total)
            entropy = 0.0
            for count in outcomes.values():
                p = count / max(1, total); entropy -= p * math.log(max(p, 1e-12))
            uncertainty = 1.0 / math.sqrt(1 + self.context_attempts[context][phase][action])
            scored.append((5.0 * probability + entropy + uncertainty,
                           -self.context_attempts[context][phase][action], -action, action))
        action = int(max(scored)[-1]); self.transfer_actions += 1
        if sum(self.context_effects[context][phase][action].values()) == 0:
            self.counterfactual_queries += 1
        self.pending = {"context": context, "phase": self.phase,
                        "target": target, "action": action}
        reason = ("execute-semantic-causal-theory" if matches else
                  "test-transferred-causal-theory")
        return action, reason

    def observe(self, context: str, action: int, events: list[str],
                progress: bool, done: bool) -> None:
        context = str(context); semantics = _transition_semantics(events)
        template = self._best_template()
        phase_before = self.phase
        phase_key = self._phase_key(phase_before)
        self.context_attempts[context][phase_key][int(action)] += 1
        for value in semantics:
            self.context_effects[context][phase_key][int(action)][value] += 1
            if not self.episode_semantics or self.episode_semantics[-1] != value:
                self.episode_semantics.append(value)

        deliberate_probe = (self.pending is not None and
                             self.pending["context"] == context)
        if deliberate_probe:
            self.predictions += 1
            matched = self.pending["target"] in semantics
            self.correct_predictions += int(matched)
        else:
            matched = bool(template and self.phase < len(template) and
                           template[self.phase] in semantics)

        if template and self.phase < len(template):
            target = template[self.phase]
            if target in semantics:
                self.phase += 1; self.context_matches[context] += 1
            elif semantics:
                # Only a prediction the semantic model deliberately chose is
                # evidence against its cross-world theory. Outcomes generated
                # by other controllers remain useful passive observations.
                if deliberate_probe:
                    self.context_mismatches[context] += 1
                # Retain a valid first event as the beginning of a new match.
                self.phase = 1 if template[0] in semantics else 0

        if progress:
            # Consecutive duplicates usually reflect animation, not additional
            # causal steps. Keep the compact successful grammar only.
            # The true successful chain is a repeatable suffix; incidental
            # exploration before it varies.  Count all bounded suffixes so
            # repeated experience statistically isolates the stable grammar.
            history = self.episode_semantics[-10:]
            for length in range(2, len(history) + 1):
                sequence = tuple(history[-length:])
                if self.templates[sequence] == 0:
                    self.templates_discovered += 1
                self.templates[sequence] += 1
            self.context_progress[context] += 1
            if template and self.phase >= len(template):
                self.context_matches[context] += 2
        if done:
            self.phase = 0; self.episode_semantics = []
        self.pending = None

    def status(self) -> dict:
        template = self._best_template()
        return {"format": self.format, "templates": len(self.templates),
                "templates_discovered": self.templates_discovered,
                "best_template": list(template) if template else None,
                "best_template_support": self.templates[template] if template else 0,
                "experienced_contexts": len(self.context_attempts),
                "grounded_contexts": sum(value > 0 for value in self.context_matches.values()),
                "transfer_actions": self.transfer_actions,
                "counterfactual_queries": self.counterfactual_queries,
                "prediction_accuracy": self.correct_predictions / max(1, self.predictions)}

    def export(self) -> dict:
        return self.status() | {
            "templates_data": [{"sequence": list(sequence), "count": count}
                               for sequence, count in self.templates.items()],
            "context_attempts": {context: {phase: dict(rows) for phase, rows in phases.items()}
                                 for context, phases in self.context_attempts.items()},
            "context_effects": {context: {phase: {str(action): dict(outcomes)
                for action, outcomes in rows.items()} for phase, rows in phases.items()}
                for context, phases in self.context_effects.items()},
            "context_matches": dict(self.context_matches),
            "context_mismatches": dict(self.context_mismatches),
            "context_progress": dict(self.context_progress),
            "correct_predictions": self.correct_predictions,
            "predictions": self.predictions}

    def restore(self, value: dict) -> None:
        if value.get("format") != self.format:
            raise ValueError("unsupported semantic world model format")
        self.templates = Counter({tuple(row["sequence"]): int(row["count"])
                                  for row in value.get("templates_data", [])})
        self.context_attempts = defaultdict(lambda: defaultdict(Counter))
        for context, phases in value.get("context_attempts", {}).items():
            for phase, rows in phases.items():
                self.context_attempts[str(context)][str(phase)] = Counter(
                    {int(action): int(count) for action, count in rows.items()})
        self.context_effects = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
        for context, phases in value.get("context_effects", {}).items():
            for phase, rows in phases.items():
                for action, outcomes in rows.items():
                    self.context_effects[str(context)][str(phase)][int(action)] = Counter(outcomes)
        self.context_matches = Counter({str(k): int(v)
                                        for k, v in value.get("context_matches", {}).items()})
        self.context_mismatches = Counter({str(k): int(v)
                                           for k, v in value.get("context_mismatches", {}).items()})
        self.context_progress = Counter({str(k): int(v)
                                         for k, v in value.get("context_progress", {}).items()})
        self.templates_discovered = int(value.get("templates_discovered", len(self.templates)))
        self.transfer_actions = int(value.get("transfer_actions", 0))
        self.counterfactual_queries = int(value.get("counterfactual_queries", 0))
        self.correct_predictions = int(value.get("correct_predictions", 0))
        self.predictions = int(value.get("predictions", 0))
