"""Color-invariant causal roles learned from successful event order."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np


_OBJECT = re.compile(r"object:c\d+:a(\d+):w(\d+):h(\d+)")


def structural_features(concept: str) -> np.ndarray:
    """Describe object structure while deliberately discarding its color."""
    match = _OBJECT.search(concept)
    if not match: raise ValueError(f"not an object concept: {concept}")
    area, width, height = map(float, match.groups())
    scale = max(1.0, width + height)
    return np.asarray((area / 15.0, width / 15.0, height / 15.0,
                       (width - height) / scale, min(width, height) / max(1.0, max(width, height))),
                      dtype=np.float64)


@dataclass
class Prototype:
    mean: np.ndarray
    count: int = 1

    def update(self, value: np.ndarray) -> None:
        self.count += 1
        self.mean += (value - self.mean) / self.count

    def similarity(self, value: np.ndarray) -> float:
        return math.exp(-5.0 * float(np.mean((self.mean - value) ** 2)))


class CausalRoleSchema:
    """Learn prerequisite→terminal structure without retaining visual identity."""
    format = "wailah-causal-roles-v1"

    def __init__(self):
        self.prerequisite: Prototype | None = None
        self.terminal: Prototype | None = None
        self.successes = 0
        self.contact_sequence: list[str] = []
        self.prerequisite_observed = False
        self.transferred_selections = 0

    @property
    def learned(self) -> bool:
        return self.prerequisite is not None and self.terminal is not None

    def reset_episode(self) -> None:
        self.contact_sequence = []
        self.prerequisite_observed = False

    def note_contact(self, concept: str) -> None:
        if not self.contact_sequence or self.contact_sequence[-1] != concept:
            self.contact_sequence.append(concept)
        if self.prerequisite is not None:
            prerequisite_score = self.prerequisite.similarity(structural_features(concept))
            terminal_score = (self.terminal.similarity(structural_features(concept))
                              if self.terminal is not None else 0.0)
            if prerequisite_score >= .8 and prerequisite_score >= terminal_score:
                self.prerequisite_observed = True

    def learn_from_progress(self) -> bool:
        # The final distinct contact produced progress. The immediately
        # preceding distinct contact is the best available prerequisite cause.
        distinct = []
        for concept in self.contact_sequence:
            if not distinct or distinct[-1] != concept: distinct.append(concept)
        if len(distinct) < 2: return False
        prerequisite, terminal = distinct[-2], distinct[-1]
        first, last = structural_features(prerequisite), structural_features(terminal)
        if self.prerequisite is None: self.prerequisite = Prototype(first.copy())
        else: self.prerequisite.update(first)
        if self.terminal is None: self.terminal = Prototype(last.copy())
        else: self.terminal.update(last)
        self.successes += 1
        return True

    def priority(self, concept: str) -> float:
        if not self.learned: return 0.0
        value = structural_features(concept)
        prototype = self.terminal if self.prerequisite_observed else self.prerequisite
        score = prototype.similarity(value)
        if score >= .8: self.transferred_selections += 1
        return 10.0 * score

    def status(self) -> dict:
        return {"format": self.format, "learned": self.learned, "successful_role_chains": self.successes,
                "prerequisite_examples": self.prerequisite.count if self.prerequisite else 0,
                "terminal_examples": self.terminal.count if self.terminal else 0,
                "transferred_selections": self.transferred_selections}

    def export(self) -> dict:
        def dump(value):
            return None if value is None else {"mean": value.mean.tolist(), "count": value.count}
        return self.status() | {"prerequisite": dump(self.prerequisite), "terminal": dump(self.terminal)}

    def restore(self, value: dict) -> None:
        if value.get("format") != self.format: raise ValueError("unsupported role schema format")
        def load(payload):
            return None if payload is None else Prototype(np.asarray(payload["mean"], dtype=np.float64),
                                                          int(payload["count"]))
        self.prerequisite = load(value.get("prerequisite")); self.terminal = load(value.get("terminal"))
        self.successes = int(value.get("successful_role_chains", 0))
        self.transferred_selections = int(value.get("transferred_selections", 0))
