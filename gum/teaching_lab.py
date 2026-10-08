"""A small, persistent teaching surface for GUM Studio.

The teacher supplies utterances and points at rendered objects.  GUM stores
those demonstrations, induces word-to-visual-feature correspondences, and
uses the learned vocabulary to resolve later references.  The interface does
not pass color or shape names to the learner as a hidden answer key.
"""
from __future__ import annotations

from pathlib import Path
import random

from PIL import Image

from .clarification_challenge import (
    ClarifyingAgent,
    ObjectSpec,
    VisualWordGrounder,
    render_scene,
    run_challenge,
    teach_visual_words,
)


class TeachingLab:
    """Persistent visual-word lessons and grounded dialogue for the Studio."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.root / "VISUAL_WORD_MEMORY.json"
        self.image_path = self.root / "CURRENT_SCENE.png"
        self.lexicon = (
            VisualWordGrounder.load(self.memory_path)
            if self.memory_path.exists()
            else VisualWordGrounder()
        )
        self.agent = ClarifyingAgent(self.lexicon)
        self.scene_number = 0
        self.last_turn = None
        self._make_scene("lesson")

    def _save(self) -> None:
        self.lexicon.save(self.memory_path)

    def _make_scene(self, mode: str) -> None:
        """Create a human-readable scene; only its pixels reach the learner."""
        self.scene_number += 1
        rng = random.Random(91_771 + self.scene_number)
        if mode == "ambiguity":
            specs = [
                ObjectSpec("red", "circle", (25, 30)),
                ObjectSpec("red", "square", (70, 30)),
                ObjectSpec("blue", "triangle", (48, 70)),
            ]
        else:
            combinations = [
                ("red", "circle"), ("blue", "square"), ("green", "triangle"),
                ("green", "circle"), ("red", "triangle"), ("blue", "circle"),
            ]
            rng.shuffle(combinations)
            positions = ((25, 30), (70, 30), (48, 70))
            specs = [ObjectSpec(color, shape, positions[index])
                     for index, (color, shape) in enumerate(combinations[:3])]
        self.frame = render_scene(specs)
        self.scene_mode = mode
        Image.fromarray(self.frame).resize((480, 480), Image.Resampling.NEAREST).save(self.image_path)
        self.agent.pending = ()
        self.last_turn = None

    def reset(self) -> dict:
        self.lexicon = VisualWordGrounder()
        self.agent = ClarifyingAgent(self.lexicon)
        self._make_scene("lesson")
        self._save()
        return self.state("Teaching memory reset.")

    def starter_curriculum(self) -> dict:
        """Run the documented pointing curriculum, not a supplied dictionary."""
        self.lexicon = VisualWordGrounder()
        teach_visual_words(self.lexicon)
        self.agent = ClarifyingAgent(self.lexicon)
        self._save()
        self._make_scene("ambiguity")
        return self.state("Starter lessons complete. Six visual words were induced from demonstrations.")

    def new_scene(self, mode: str = "lesson") -> dict:
        if mode not in {"lesson", "ambiguity"}:
            raise ValueError("scene mode must be 'lesson' or 'ambiguity'")
        self._make_scene(mode)
        return self.state("New scene ready.")

    def demonstrate(self, phrase: str, x: int, y: int) -> dict:
        phrase = phrase.strip()
        if not phrase:
            raise ValueError("type a phrase before pointing")
        # The displayed image is 5x the learner's 96px input.  The browser
        # reports coordinates in that natural displayed image.
        lx = max(0, min(95, round(int(x) / 5)))
        ly = max(0, min(95, round(int(y) / 5)))
        self.lexicon.demonstrate(phrase, self.frame, (lx, ly))
        self._save()
        return self.state(f"Stored pointing example {len(self.lexicon.examples)}. Add varied examples, then learn meanings.")

    def learn(self) -> dict:
        self.lexicon.learn()
        self.agent = ClarifyingAgent(self.lexicon)
        self._save()
        return self.state(f"Learning complete. Grounded {len(self.lexicon.word_to_feature)} words.")

    def begin(self, instruction: str) -> dict:
        self.last_turn = self.agent.begin(self.frame, instruction.strip())
        return self.state(self.last_turn.response)

    def answer(self, text: str) -> dict:
        self.last_turn = self.agent.answer(text.strip())
        return self.state(self.last_turn.response)

    def audit(self) -> dict:
        report = run_challenge(self.root / "reference-audit")
        return {
            "passed": report["passed"],
            "results": report["results"],
            "note": "This reproducibility audit trains and tests a fresh learner from the documented curriculum.",
        }

    def state(self, message: str = "") -> dict:
        turn = None if self.last_turn is None else self.last_turn.to_json()
        return {
            "message": message,
            "examples": len(self.lexicon.examples),
            "learned_words": dict(sorted(self.lexicon.word_to_feature.items())),
            "learned_word_count": len(self.lexicon.word_to_feature),
            "scene_mode": self.scene_mode,
            "scene_number": self.scene_number,
            "pending_clarification": bool(self.agent.pending),
            "last_turn": turn,
        }
