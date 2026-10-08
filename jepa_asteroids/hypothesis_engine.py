"""Domain-independent hypothesis testing and reusable procedure memory.

The engine does not know game names, objects, controls, or goals.  It tracks
four broad explanations for useful behavior--spatial contact, contextual
interaction, temporal persistence, and remembered procedure--and allocates
experiments when the currently preferred explanation stops producing novelty.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import math


@dataclass
class Procedure:
    """A visually grounded action chain that previously preceded progress."""
    steps: list[tuple[str, int, str]]
    successes: int = 1
    failures: int = 0

    @property
    def entry_state(self) -> str:
        return self.steps[0][0]

    @property
    def reliability(self) -> float:
        return (self.successes + 1.0) / (self.successes + self.failures + 2.0)


@dataclass
class HypothesisRecord:
    attempts: int = 0
    progress_credit: float = 0.0
    novelty_credit: float = 0.0
    contradictions: int = 0

    @property
    def score(self) -> float:
        return ((1.0 + self.progress_credit + .15 * self.novelty_credit) /
                (2.0 + self.attempts + self.contradictions))


class CausalHypothesisEngine:
    """Select informative interventions and retain successful procedures."""
    modes = ("spatial-contact", "contextual-interaction", "temporal-persistence",
             "graph-frontier", "procedure-replay")

    def __init__(self):
        self.records = {mode: HypothesisRecord() for mode in self.modes}
        self.context_actions = defaultdict(Counter)
        self.outcomes = defaultdict(Counter)
        self.action_changes = defaultdict(list)
        self.action_runs = Counter()
        self.recent = deque(maxlen=96)
        self.procedures: list[Procedure] = []
        self.active_procedure: Procedure | None = None
        self.active_index = 0
        self.last_mode = "graph-frontier"
        self.last_state = None
        self.stagnation = 0
        self.novel_outcomes = 0
        self.abandoned_hypotheses = 0
        self.experiments = 0
        self.procedure_replays = 0
        self.procedure_aborts = 0

    def reset_episode(self) -> None:
        self.active_procedure = None
        self.active_index = 0
        self.last_state = None
        self.stagnation = 0
        self.recent.clear()

    def _procedure_action(self, state: str, actions: list[int]) -> int | None:
        if self.active_procedure is not None:
            expected, action, _ = self.active_procedure.steps[self.active_index]
            if expected == state and action in actions:
                self.last_mode = "procedure-replay"
                self.procedure_replays += 1
                return int(action)
            self.active_procedure.failures += 1
            self.procedure_aborts += 1
            self.active_procedure = None; self.active_index = 0
        candidates = [p for p in self.procedures if p.entry_state == state and p.reliability >= .5]
        if candidates:
            self.active_procedure = max(candidates,
                                        key=lambda p: (p.reliability, p.successes, -len(p.steps)))
            self.active_index = 0
            action = self.active_procedure.steps[0][1]
            if action in actions:
                self.last_mode = "procedure-replay"
                self.procedure_replays += 1
                return int(action)
            self.active_procedure = None
        return None

    def recommend(self, state: str, actions: list[int], step: int,
                  spatial_ready: bool, movement_actions: set[int],
                  global_counts: Counter) -> tuple[str, int | None]:
        """Return an explanatory mode and, when needed, an experiment action."""
        procedure_action = self._procedure_action(state, actions)
        if procedure_action is not None:
            return "procedure-replay", procedure_action

        # Force an epistemic probe occasionally even while a spatial planner is
        # productive.  Under stagnation, cycle through alternative explanations
        # rather than repeating a failed strategy forever.
        if self.stagnation >= 9:
            phase = (self.stagnation - 9) % 24
            if phase < 12:
                candidates = ["contextual-interaction", "temporal-persistence", "graph-frontier"]
                if spatial_ready: candidates.append("spatial-contact")
                total = sum(self.records[name].attempts for name in candidates) + 1
                # Evidence-weighted uncertainty: productive hypotheses rise,
                # but under-tested explanations retain a principled chance.
                mode = max(candidates, key=lambda name: (
                    self.records[name].score + .8 * math.sqrt(
                        math.log(total + 2) / (1 + self.records[name].attempts)), name))
            else:
                mode = "spatial-contact" if spatial_ready else "graph-frontier"
            if phase == 0:
                self.abandoned_hypotheses += 1
        elif step > 0 and step % 17 == 0:
            mode = "contextual-interaction"
        elif spatial_ready:
            mode = "spatial-contact"
        else:
            mode = "graph-frontier"

        self.last_mode = mode
        if mode not in ("contextual-interaction", "temporal-persistence"):
            return mode, None

        scored = []
        for action in actions:
            trials = self.context_actions[state][action]
            outcome_counts = self.outcomes.get((state, action), Counter())
            total = sum(outcome_counts.values())
            entropy = 0.0
            for count in outcome_counts.values():
                probability = count / max(1, total)
                entropy -= probability * math.log(max(probability, 1e-12))
            changes = self.action_changes.get(action, [])
            effect = sum(changes[-32:]) / max(1, len(changes[-32:]))
            nonmovement = 1.0 if action not in movement_actions else 0.0
            uncertainty = 2.0 / math.sqrt(1 + trials)
            diversity = .35 / math.sqrt(1 + global_counts[action])
            temporal = .6 / math.sqrt(1 + self.action_runs[(action, 4)]) if mode == "temporal-persistence" else 0.0
            scored.append((uncertainty + entropy + effect + nonmovement + diversity + temporal, action))
        self.experiments += 1
        return mode, int(max(scored)[1])

    def observe(self, state: str, action: int, next_state: str, changed_fraction: float,
                progress: bool, done: bool) -> None:
        mode = self.last_mode
        record = self.records[mode]
        record.attempts += 1
        self.context_actions[state][action] += 1
        key = (state, action); novel = self.outcomes[key][next_state] == 0
        self.outcomes[key][next_state] += 1
        self.action_changes[int(action)].append(float(changed_fraction))
        if novel:
            self.novel_outcomes += 1
            record.novelty_credit += 1.0
        if progress:
            record.progress_credit += 8.0
        if done and not progress:
            record.contradictions += 1

        recent_destinations = [row[2] for row in list(self.recent)[-8:]]
        looping = next_state in recent_destinations
        self.recent.append((state, int(action), next_state, mode))
        if progress:
            # Store the shortest useful suffix available, with visual-state
            # guards at every step so it aborts instead of blindly executing in
            # a changed world.
            path = [(s, a, nxt) for s, a, nxt, _ in list(self.recent)[-64:]]
            if path:
                duplicate = next((p for p in self.procedures if p.steps == path), None)
                if duplicate: duplicate.successes += 1
                else: self.procedures.append(Procedure(path))
            self.stagnation = 0
        elif state != next_state and not looping:
            self.stagnation = max(0, self.stagnation - 2)
        else:
            self.stagnation += 1

        if self.active_procedure is not None:
            expected_next = self.active_procedure.steps[self.active_index][2]
            if next_state == expected_next:
                self.active_index += 1
                if self.active_index >= len(self.active_procedure.steps):
                    if progress: self.active_procedure.successes += 1
                    self.active_procedure = None; self.active_index = 0
            else:
                self.active_procedure.failures += 1
                self.procedure_aborts += 1
                self.active_procedure = None; self.active_index = 0
        self.last_state = next_state

    def status(self) -> dict:
        return {
            "hypotheses": {name: {"attempts": row.attempts,
                                   "progress_credit": row.progress_credit,
                                   "novelty_credit": row.novelty_credit,
                                   "contradictions": row.contradictions,
                                   "score": row.score}
                           for name, row in self.records.items()},
            "stagnation": self.stagnation,
            "experiments": self.experiments,
            "novel_outcomes": self.novel_outcomes,
            "abandoned_hypotheses": self.abandoned_hypotheses,
            "learned_procedures": len(self.procedures),
            "procedure_replays": self.procedure_replays,
            "procedure_aborts": self.procedure_aborts,
        }

    def export(self) -> dict:
        """Return durable conceptual memory, excluding transient episode state."""
        return {
            "format": "wailah-causal-hypotheses-v1",
            "records": {name: {"attempts": row.attempts,
                                "progress_credit": row.progress_credit,
                                "novelty_credit": row.novelty_credit,
                                "contradictions": row.contradictions}
                        for name, row in self.records.items()},
            "procedures": [{"steps": [[state, action, nxt] for state, action, nxt in row.steps],
                            "successes": row.successes, "failures": row.failures}
                           for row in self.procedures],
            "action_changes": {str(action): values[-128:]
                               for action, values in self.action_changes.items()},
            "counters": {"novel_outcomes": self.novel_outcomes,
                         "abandoned_hypotheses": self.abandoned_hypotheses,
                         "experiments": self.experiments,
                         "procedure_replays": self.procedure_replays,
                         "procedure_aborts": self.procedure_aborts},
        }

    def restore(self, value: dict) -> None:
        """Restore only validated durable memories from an exported mapping."""
        if value.get("format") != "wailah-causal-hypotheses-v1":
            raise ValueError("unsupported hypothesis memory format")
        for name, payload in value.get("records", {}).items():
            if name not in self.records: continue
            self.records[name] = HypothesisRecord(
                attempts=int(payload.get("attempts", 0)),
                progress_credit=float(payload.get("progress_credit", 0.0)),
                novelty_credit=float(payload.get("novelty_credit", 0.0)),
                contradictions=int(payload.get("contradictions", 0)))
        self.procedures = []
        for payload in value.get("procedures", []):
            steps = [(str(state), int(action), str(nxt)) for state, action, nxt in payload.get("steps", [])]
            if steps:
                self.procedures.append(Procedure(steps, int(payload.get("successes", 1)),
                                                 int(payload.get("failures", 0))))
        self.action_changes = defaultdict(list, {
            int(action): [float(x) for x in values]
            for action, values in value.get("action_changes", {}).items()})
        for name, amount in value.get("counters", {}).items():
            if hasattr(self, name): setattr(self, name, int(amount))
