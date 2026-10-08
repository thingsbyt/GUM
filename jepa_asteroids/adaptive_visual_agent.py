"""Learned layout adapter around the byte-frozen universal goal-state core.

The adapter is not told a viewport rectangle or scale.  It fits those latent
parameters by searching pixel-layout hypotheses and maximizing perceptual
coherence under the frozen core's public visual schema.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .persistent_workshop import HEIGHT, WIDTH
from .universal_goal_state import UniversalGoalStateAgent, perceive_state


def dashboard_embed(frame, *, canvas_height=318, canvas_width=426, top=41, left=117, scale=2,
                    seed=0):
    """Development-only interface transform; returns an HWC dashboard image."""
    source = np.transpose(np.asarray(frame, dtype=np.uint8), (1, 2, 0))
    rng = np.random.default_rng(seed)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = np.asarray((18, 23, 34), dtype=np.uint8)
    canvas[:28] = np.asarray((31, 39, 55), dtype=np.uint8)
    canvas[:, :78] = np.asarray((24, 31, 45), dtype=np.uint8)
    for row in range(5):
        y = 48 + row * 43; canvas[y:y + 25, 13:64] = rng.integers(35, 100, (1, 1, 3), dtype=np.uint8)
    enlarged = np.repeat(np.repeat(source, scale, axis=0), scale, axis=1)
    canvas[top:top + HEIGHT * scale, left:left + WIDTH * scale] = enlarged
    return canvas


class LearnedLayoutAdapter:
    """Fits translation and integer scale from raw pixels without coordinates."""
    format = "wailah-learned-layout-adapter-v1"

    def __init__(self):
        self.geometry = None; self.hypotheses_tested = 0; self.frames_adapted = 0
        self.refits = 0; self.confidence = 0.0; self.history = []

    @staticmethod
    def _hwc(frame):
        value = np.asarray(frame, dtype=np.uint8)
        if value.ndim != 3: raise ValueError("expected three-dimensional RGB pixels")
        if value.shape[0] == 3: value = np.transpose(value, (1, 2, 0))
        if value.shape[2] != 3: raise ValueError("expected RGB pixels")
        return value

    @staticmethod
    def _runs(mask):
        padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
        edges = np.diff(padded); starts = np.flatnonzero(edges == 1); ends = np.flatnonzero(edges == -1)
        return zip(starts.tolist(), ends.tolist())

    def _candidate_geometries(self, raw):
        height, width = raw.shape[:2]; found = set()
        for scale in (1, 2, 3):
            needed_h, needed_w = HEIGHT * scale, WIDTH * scale
            if needed_h > height or needed_w > width: continue
            minimum_run = 100 * scale
            for y in range(height):
                row = raw[y]
                same_next = np.all(row[:-1] == row[1:], axis=1)
                nonzero = row[:-1].max(axis=1) > 0
                for start, end in self._runs(same_next & nonzero):
                    # N equal adjacencies describe N+1 equal pixels.
                    if end - start + 1 < minimum_run: continue
                    top = y - 22 * scale; left = start - 3 * scale
                    if 0 <= top <= height - needed_h and 0 <= left <= width - needed_w:
                        found.add((scale, top, left))
        return sorted(found)

    @staticmethod
    def _decode(raw, geometry):
        scale, top, left = geometry
        crop = raw[top:top + HEIGHT * scale, left:left + WIDTH * scale]
        offset = scale // 2
        canonical = crop[offset::scale, offset::scale][:HEIGHT, :WIDTH]
        return np.transpose(canonical, (2, 0, 1)).copy()

    @staticmethod
    def _coherence(canonical):
        # Schema constraints are visual, not hidden coordinates: the public
        # interface contains one nine-pixel white effector and a continuous
        # separator.  They disambiguate a true rescaling from a coincidental
        # parse of dashboard chrome or a zoomed subsection.
        white = np.all(canonical > 245, axis=0)
        if int(white.sum()) != 9: return None
        divider = canonical[:, 22:24, 3:109]
        if divider.max() == 0 or not np.all(divider == divider[:, :1, :1]): return None
        try:
            view = perceive_state(canonical)
        except (RuntimeError, ValueError):
            return None
        mapped = len(view["floor"]) + len(view["walls"])
        if mapped < 4: return None
        return 1500.0 + mapped * 4.0 + len(view["objects"]) * 7.0

    def adapt(self, frame):
        raw = self._hwc(frame); self.frames_adapted += 1
        if self.geometry is not None:
            canonical = self._decode(raw, self.geometry); score = self._coherence(canonical)
            if score is not None:
                self.confidence = min(1.0, self.confidence + 0.02); return canonical
            self.refits += 1; self.geometry = None
        best = None
        for geometry in self._candidate_geometries(raw):
            canonical = self._decode(raw, geometry); score = self._coherence(canonical)
            self.hypotheses_tested += 1
            if score is not None and (best is None or score > best[0]): best = (score, geometry, canonical)
        if best is None: raise RuntimeError("no coherent visual-layout hypothesis found")
        score, self.geometry, canonical = best; self.confidence = min(1.0, score / 1400.0)
        self.history.append({"scale": self.geometry[0], "top": self.geometry[1], "left": self.geometry[2],
                             "score": score})
        return canonical

    def status(self):
        return {"format": self.format, "geometry": list(self.geometry) if self.geometry else None,
                "hypotheses_tested": self.hypotheses_tested, "frames_adapted": self.frames_adapted,
                "refits": self.refits, "confidence": self.confidence, "discoveries": list(self.history)}

    @classmethod
    def restore(cls, value):
        result = cls(); geometry = value.get("geometry")
        result.geometry = tuple(geometry) if geometry else None
        result.hypotheses_tested = int(value.get("hypotheses_tested", 0))
        result.frames_adapted = int(value.get("frames_adapted", 0)); result.refits = int(value.get("refits", 0))
        result.confidence = float(value.get("confidence", 0.0)); result.history = list(value.get("discoveries", []))
        return result


class AdaptiveUniversalAgent:
    """Raw-interface learner composed with an unchanged v14 cognitive core."""
    format = "wailah-adaptive-universal-agent-v1"

    def __init__(self):
        self.adapter = LearnedLayoutAdapter(); self.core = UniversalGoalStateAgent()

    def begin(self, frame): return self.core.begin(self.adapter.adapt(frame))
    def act(self, frame): return self.core.act(self.adapter.adapt(frame))

    def observe(self, before, action, after, reward, done):
        return self.core.observe(self.adapter.adapt(before), action, self.adapter.adapt(after), reward, done)

    @property
    def memory(self): return self.core.memory
    def status(self): return {"format": self.format, "adapter": self.adapter.status(), "core": self.core.status()}

    def save(self, path):
        value = {"format": self.format, "adapter": self.adapter.status(), "core": self.core.export()}
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format") != cls.format: raise ValueError("unsupported adaptive mind")
        result = cls(); result.adapter = LearnedLayoutAdapter.restore(value["adapter"]); core = value["core"]
        result.core.worlds = core["worlds"]
        result.core.form_roles = defaultdict(Counter, {key: Counter(rows) for key, rows in core.get("form_roles", {}).items()})
        result.core.experiments = int(core.get("experiments_total", 0)); result.core.predictions = int(core.get("predictions", 0))
        result.core.correct_predictions = int(core.get("correct_predictions", 0))
        result.core.self_generated_subgoals = int(core.get("self_generated_subgoals", 0))
        result.core.prevented_risky_actions = int(core.get("prevented_risky_actions", 0))
        return result
