"""Integrity-checked cumulative learner bundle for GUM School.

The existing symbolic/linear school learner retains the first two Object
Laboratory lessons.  The recurrent meta-policy handles anonymous controls.
This module joins them behind the public Mind interface without rewriting or
mutating either frozen component.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from gum.lineage import file_sha256
from gum.protocol import PublicWorldSpec, Transition
from gum.storage import atomic_write_json

from .learner import CrossSeedSchoolLearner
from .recurrent_meta import RecurrentCausalLearner
from .worlds import CAUSAL_WORKSHOP_ADAPTER


BASE_FILENAME = "SCHOOL_LEARNER.json"
CAUSAL_FILENAME = "CAUSAL_META_POLICY.pt"
BUNDLE_FILENAME = "CUMULATIVE_LEARNER.json"


class CumulativeLearnerError(ValueError):
    pass


class CumulativeSchoolLearner:
    """Dispatch two frozen specialists by the reviewed public adapter id."""

    format = "gum-school-cumulative-learner-v1"

    def __init__(
        self,
        base: CrossSeedSchoolLearner,
        causal: RecurrentCausalLearner,
    ):
        self.base = base
        self.causal = causal
        self._active: CrossSeedSchoolLearner | RecurrentCausalLearner | None = None
        self._active_adapter: str | None = None

    def begin(self, spec: PublicWorldSpec, observation: Any, *, training: bool) -> None:
        self._active_adapter = spec.adapter
        self._active = (
            self.causal if spec.adapter == CAUSAL_WORKSHOP_ADAPTER else self.base
        )
        self._active.begin(spec, observation, training=training)

    def _delegate(self):
        if self._active is None:
            raise CumulativeLearnerError("begin must be called before using the learner")
        return self._active

    def act(self, observation: Any, *, training: bool) -> int:
        return int(self._delegate().act(observation, training=training))

    def observe(self, action: Any, transition: Transition, *, training: bool) -> None:
        self._delegate().observe(action, transition, training=training)

    def finish_episode(self, *, training: bool):
        return self._delegate().finish_episode(training=training)

    def confidence(self) -> float:
        return float(self._delegate().confidence())

    @property
    def boundary_violations(self) -> list[str]:
        # The recurrent learner's observe method has no public_info access; its
        # boundary is covered by a hostile-dictionary test.  The base learner
        # records any unexpected public fields explicitly.
        return list(self.base.boundary_violations)

    def status(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "active_adapter": self._active_adapter,
            "base": self.base.status(),
            "causal": self.causal.status(),
            "routing": {
                CAUSAL_WORKSHOP_ADAPTER: "recurrent-causal-meta-policy",
                "default": "promoted-cross-seed-school-learner",
            },
        }

    @classmethod
    def create_bundle(
        cls,
        directory: Path,
        *,
        base_path: Path,
        causal_path: Path,
    ) -> Path:
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"candidate bundle directory is not empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        base_target = directory / BASE_FILENAME
        causal_target = directory / CAUSAL_FILENAME
        shutil.copy2(Path(base_path), base_target)
        shutil.copy2(Path(causal_path), causal_target)
        # Load both before sealing the bundle so malformed source files cannot
        # acquire a valid bundle manifest.
        CrossSeedSchoolLearner.load(base_target)
        RecurrentCausalLearner.load(causal_target)
        manifest = {
            "format": cls.format,
            "components": {
                BASE_FILENAME: f"sha256:{file_sha256(base_target)}",
                CAUSAL_FILENAME: f"sha256:{file_sha256(causal_target)}",
            },
            "routing": {
                CAUSAL_WORKSHOP_ADAPTER: CAUSAL_FILENAME,
                "default": BASE_FILENAME,
            },
            "training_complete_before_bundle": True,
        }
        path = directory / BUNDLE_FILENAME
        atomic_write_json(path, manifest, backup=False, sort_keys=True)
        return path

    @classmethod
    def augment_existing_base(
        cls,
        directory: Path,
        *,
        causal_path: Path,
    ) -> Path:
        """Add a reviewed recurrent specialist to an engine-cloned base state."""
        directory = Path(directory)
        base_target = directory / BASE_FILENAME
        if not directory.is_dir() or set(
            path.name for path in directory.iterdir()
        ) != {BASE_FILENAME}:
            raise CumulativeLearnerError(
                "existing candidate must contain only the reviewed base learner"
            )
        CrossSeedSchoolLearner.load(base_target)
        causal_target = directory / CAUSAL_FILENAME
        shutil.copy2(Path(causal_path), causal_target)
        RecurrentCausalLearner.load(causal_target)
        manifest = {
            "format": cls.format,
            "components": {
                BASE_FILENAME: f"sha256:{file_sha256(base_target)}",
                CAUSAL_FILENAME: f"sha256:{file_sha256(causal_target)}",
            },
            "routing": {
                CAUSAL_WORKSHOP_ADAPTER: CAUSAL_FILENAME,
                "default": BASE_FILENAME,
            },
            "training_complete_before_bundle": True,
        }
        path = directory / BUNDLE_FILENAME
        atomic_write_json(path, manifest, backup=False, sort_keys=True)
        cls.load_bundle(directory)
        return path

    @classmethod
    def load_bundle(cls, directory: Path) -> "CumulativeSchoolLearner":
        directory = Path(directory)
        path = directory / BUNDLE_FILENAME
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CumulativeLearnerError(f"cumulative bundle is unreadable: {error}") from error
        required = {
            "format", "components", "routing", "training_complete_before_bundle"
        }
        if not isinstance(value, dict) or set(value) != required:
            raise CumulativeLearnerError("cumulative bundle fields differ from the reviewed format")
        if value["format"] != cls.format or value["training_complete_before_bundle"] is not True:
            raise CumulativeLearnerError("unsupported or incomplete cumulative bundle")
        expected_routing = {
            CAUSAL_WORKSHOP_ADAPTER: CAUSAL_FILENAME,
            "default": BASE_FILENAME,
        }
        if value["routing"] != expected_routing:
            raise CumulativeLearnerError("cumulative routing differs from the reviewed policy")
        expected_components = {BASE_FILENAME, CAUSAL_FILENAME}
        if not isinstance(value["components"], dict) or set(value["components"]) != expected_components:
            raise CumulativeLearnerError("cumulative component inventory is invalid")
        for filename, expected_hash in value["components"].items():
            component = directory / filename
            actual_hash = f"sha256:{file_sha256(component)}" if component.is_file() else None
            if actual_hash != expected_hash:
                raise CumulativeLearnerError(f"cumulative component hash differs: {filename}")
        return cls(
            CrossSeedSchoolLearner.load(directory / BASE_FILENAME),
            RecurrentCausalLearner.load(directory / CAUSAL_FILENAME),
        )
