"""Reusable learned perception services available to GUM worlds and operators."""
from __future__ import annotations

from pathlib import Path


class ConceptBridgeService:
    format = "gum-concept-bridge-service-v1"

    def __init__(self, checkpoint: Path):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists(): raise FileNotFoundError(self.checkpoint)
        self.uses = 0

    def identify(self, image: Path, *, variants=8):
        from jepa_asteroids.concept_bridge_identify import identify_file
        result = identify_file(Path(image), self.checkpoint, variants=variants)
        self.uses += 1; return result

    def status(self):
        from jepa_asteroids.concept_bridge import CONCEPTS
        return {"format": self.format, "checkpoint": str(self.checkpoint.resolve()),
                "closed_set_concepts": list(CONCEPTS), "uses": self.uses}
