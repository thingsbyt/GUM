"""Visual goal grounding and compositional semantic skill execution.

The composer learns what visual symbols mean by aligning successful terminal
event suffixes with the visible goal.  It then composes the independently
grounded meanings in new orders and discovers their local controls through
intervention.  No symbol dictionary, control map, or solution sequence is
provided by the caller.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

from .semantic_world_model import _transition_semantics


def visual_symbol_sequence(frame) -> list[str]:
    """Read ordered, translation/color-invariant connected visual glyphs."""
    grid = np.asarray(frame, dtype=np.uint8)
    if grid.ndim == 3:
        composed = np.zeros(grid.shape[1:], dtype=np.uint8)
        for layer in grid:
            composed = np.where(layer != 0, layer, composed)
        grid = composed
    if grid.ndim != 2:
        raise ValueError(f"expected a 2-D visual goal, got {grid.shape}")
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    seen = np.zeros_like(grid, dtype=bool); components = []
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            if seen[y, x] or int(grid[y, x]) == background:
                continue
            color = int(grid[y, x]); stack = [(y, x)]; seen[y, x] = True; points = []
            while stack:
                yy, xx = stack.pop(); points.append((yy, xx))
                for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                    ny, nx = yy + dy, xx + dx
                    if (0 <= ny < height and 0 <= nx < width and not seen[ny, nx]
                            and int(grid[ny, nx]) == color):
                        seen[ny, nx] = True; stack.append((ny, nx))
            if len(points) < 2:
                continue
            ys = [p[0] for p in points]; xs = [p[1] for p in points]
            y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
            mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
            for yy, xx in points: mask[yy - y0, xx - x0] = 1
            rows = "/".join("".join(map(str, row)) for row in mask.tolist())
            signature = f"glyph:{mask.shape[0]}x{mask.shape[1]}:{rows}"
            components.append((x0, y0, signature))
    return [signature for _, _, signature in sorted(components)]


class SemanticSkillComposer:
    """Ground a visual event language and compose its skills across worlds."""
    format = "wailah-semantic-skill-composer-v1"

    def __init__(self):
        self.token_evidence = defaultdict(Counter)
        self.context_effects = defaultdict(lambda: defaultdict(Counter))
        self.context_attempts = defaultdict(Counter)
        self.successful_compositions = Counter()
        self.curriculum_selections = 0; self.composed_actions = 0
        self.counterfactual_queries = 0; self.predictions = 0; self.correct_predictions = 0
        self.reset_episode("default")

    def reset_episode(self, context: str, goal_frame=None) -> None:
        self.context = str(context); self.goal_tokens = []
        self.goal_semantics = None; self.phase = 0; self.history = []
        self.pending = None
        if goal_frame is not None:
            self.set_goal(goal_frame)

    def set_goal(self, goal_frame) -> list[str]:
        tokens = visual_symbol_sequence(goal_frame)
        if not tokens:
            raise ValueError("visual goal contains no readable symbols")
        self.goal_tokens = tokens; self.phase = 0; self.history = []; self.pending = None
        self.goal_semantics = self.decode_tokens(tokens)
        return tokens

    def _meaning(self, token: str) -> str | None:
        evidence = self.token_evidence[token]
        if not evidence:
            return None
        ranked = evidence.most_common()
        best, support = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        return best if support >= 2 and support > runner_up else None

    def decode_tokens(self, tokens: list[str]) -> list[str] | None:
        meanings = [self._meaning(token) for token in tokens]
        return None if any(value is None for value in meanings) else meanings

    def curriculum_priority(self, goal_frame) -> float:
        """Prefer experiences that disambiguate unknown or weakly known words."""
        score = 0.0
        for token in visual_symbol_sequence(goal_frame):
            rows = self.token_evidence[token]
            if not rows:
                score += 10.0
            else:
                ranked = rows.most_common(2); best = ranked[0][1]
                margin = best - (ranked[1][1] if len(ranked) > 1 else 0)
                score += 4.0 / math.sqrt(1 + best) + 2.0 / (1 + margin)
        return score

    def choose_curriculum(self, goal_frames: list) -> int:
        if not goal_frames:
            raise ValueError("no curriculum candidates")
        self.curriculum_selections += 1
        return max(range(len(goal_frames)),
                   key=lambda index: (self.curriculum_priority(goal_frames[index]), -index))

    def recommend(self, actions: list[int]):
        actions = sorted(map(int, actions))
        if not actions or self.goal_semantics is None:
            return None
        if self.phase >= len(self.goal_semantics):
            self.phase = 0
        target = self.goal_semantics[self.phase]; rows = self.context_effects[self.context]
        scored = []
        for action in actions:
            outcomes = rows[action]; total = sum(outcomes.values())
            probability = outcomes[target] / max(1, total)
            uncertainty = 1.0 / math.sqrt(1 + self.context_attempts[self.context][action])
            # Reliable causal knowledge dominates; otherwise test the least
            # sampled control. Numeric action identity carries no prior value.
            scored.append((8.0 * probability + uncertainty,
                           -self.context_attempts[self.context][action], -action, action))
        action = int(max(scored)[-1]); self.composed_actions += 1
        if not rows[action]: self.counterfactual_queries += 1
        self.pending = (target, action)
        reason = ("execute-composed-semantic-goal" if rows[action][target]
                  else "ground-composed-semantic-goal")
        return action, reason

    def observe(self, action: int, events: list[str], progress: bool, done: bool) -> None:
        semantics = _transition_semantics(events); action = int(action)
        self.context_attempts[self.context][action] += 1
        for value in semantics:
            self.context_effects[self.context][action][value] += 1
            self.history.append(value)
        if self.pending is not None:
            self.predictions += 1
            self.correct_predictions += int(self.pending[0] in semantics)

        if self.goal_semantics is not None and self.phase < len(self.goal_semantics):
            target = self.goal_semantics[self.phase]
            if target in semantics:
                self.phase += 1
            elif semantics:
                self.phase = 1 if self.goal_semantics[0] in semantics else 0

        if progress and self.goal_tokens:
            # Terminal reward identifies the successful suffix. Alignment by
            # order is a hypothesis about the visible command, accumulated as
            # evidence rather than installed as a dictionary.
            length = len(self.goal_tokens); suffix = self.history[-length:]
            if len(suffix) == length:
                for token, meaning in zip(self.goal_tokens, suffix):
                    self.token_evidence[token][meaning] += 1
                self.successful_compositions[tuple(self.goal_tokens)] += 1
                self.goal_semantics = self.decode_tokens(self.goal_tokens)
        if done:
            self.phase = 0
        self.pending = None

    def status(self) -> dict:
        grounded = {token: meaning for token in self.token_evidence
                    if (meaning := self._meaning(token)) is not None}
        return {"format": self.format, "visual_symbols_observed": len(self.token_evidence),
                "grounded_symbol_meanings": len(grounded),
                "grounded_dictionary": grounded,
                "successful_compositions": len(self.successful_compositions),
                "experienced_contexts": len(self.context_attempts),
                "curriculum_selections": self.curriculum_selections,
                "composed_actions": self.composed_actions,
                "counterfactual_queries": self.counterfactual_queries,
                "prediction_accuracy": self.correct_predictions / max(1, self.predictions)}

    def export(self) -> dict:
        return self.status() | {
            "token_evidence": {token: dict(rows) for token, rows in self.token_evidence.items()},
            "context_effects": {context: {str(action): dict(rows) for action, rows in actions.items()}
                                for context, actions in self.context_effects.items()},
            "context_attempts": {context: dict(rows) for context, rows in self.context_attempts.items()},
            "successful_compositions_data": [
                {"tokens": list(tokens), "count": count}
                for tokens, count in self.successful_compositions.items()],
            "correct_predictions": self.correct_predictions, "predictions": self.predictions}

    def restore(self, value: dict) -> None:
        if value.get("format") != self.format:
            raise ValueError("unsupported semantic composer format")
        self.token_evidence = defaultdict(Counter, {
            str(token): Counter({str(k): int(v) for k, v in rows.items()})
            for token, rows in value.get("token_evidence", {}).items()})
        self.context_effects = defaultdict(lambda: defaultdict(Counter))
        for context, actions in value.get("context_effects", {}).items():
            for action, rows in actions.items():
                self.context_effects[str(context)][int(action)] = Counter(
                    {str(k): int(v) for k, v in rows.items()})
        self.context_attempts = defaultdict(Counter, {
            str(context): Counter({int(k): int(v) for k, v in rows.items()})
            for context, rows in value.get("context_attempts", {}).items()})
        self.successful_compositions = Counter({tuple(row["tokens"]): int(row["count"])
                                                for row in value.get("successful_compositions_data", [])})
        self.curriculum_selections = int(value.get("curriculum_selections", 0))
        self.composed_actions = int(value.get("composed_actions", 0))
        self.counterfactual_queries = int(value.get("counterfactual_queries", 0))
        self.correct_predictions = int(value.get("correct_predictions", 0))
        self.predictions = int(value.get("predictions", 0))
        self.reset_episode("default")
