"""Mind-protocol wrappers for the two established cooperative visual learners."""
from __future__ import annotations

import json
import hashlib
import pickle
from pathlib import Path

from .protocol import PublicWorldSpec, Transition
from .storage import atomic_write_bytes, atomic_write_json


class CooperativeTeamMind:
    """Adapts a two-body learning team to the common GUM mind lifecycle."""
    format = "gum-cooperative-team-mind-v1"

    def __init__(self, family: str, *, communicate=True):
        self.family = str(family); self.communicate = bool(communicate)
        self.team = self._new_team(); self.before = None; self.spec = None
        self.episodes = 0; self.steps = 0; self.last_status = {}

    def _new_team(self):
        if self.family == "asteroids":
            from jepa_asteroids.cooperative_asteroids import TwoRocketTeam
            return TwoRocketTeam(communicate=self.communicate)
        if self.family in ("immune", "immune-savior"):
            from jepa_asteroids.cooperative_immune_savior import CooperativeImmuneTeam
            return CooperativeImmuneTeam(allow_cooperation=self.communicate)
        raise ValueError(self.family)

    def begin(self, spec: PublicWorldSpec, observation, *, training: bool):
        self.spec = spec; self.before = observation; self.team.begin(observation); self.episodes += 1

    def act(self, observation, *, training: bool):
        self.before = observation
        return [int(value) for value in self.team.act(observation)]

    def observe(self, action, transition: Transition, *, training: bool):
        self.team.observe(self.before, action, transition.observation, transition.reward,
                          transition.terminated or transition.truncated)
        self.before = transition.observation; self.steps += 1; self.last_status = self.team.status()

    def status(self):
        memory = ("world-specific discoveries persist across attempts" if self.family != "asteroids"
                  else "general visual policy retained; shuffled controls re-grounded each episode")
        return {"format": self.format, "family": self.family, "agents": 2,
                "episodes": self.episodes, "steps": self.steps, "memory": memory,
                "team": self.team.status()}

    def save(self, path: Path):
        path = Path(path); state_path = path.with_suffix(".state.pkl")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(state_path, pickle.dumps(self.team, protocol=pickle.HIGHEST_PROTOCOL))
        payload = {"format": self.format, "family": self.family, "communicate": self.communicate,
                   "episodes": self.episodes, "steps": self.steps, "status": self.status(),
                   "state_file": state_path.name,
                   "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
                   "trust_boundary": "load only snapshots created in this GUM workspace"}
        atomic_write_json(path, payload)

    @classmethod
    def load(cls, path: Path, *, trusted: bool = False):
        """Reload a checksum-verified, explicitly trusted local snapshot.

        Pickle is executable input.  A digest detects corruption but cannot make
        an untrusted file safe, so callers must opt in at the boundary.
        """
        if not trusted:
            raise ValueError("refusing to load pickle state without trusted=True")
        path = Path(path); payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != cls.format: raise ValueError("unsupported specialist mind format")
        state_path = path.with_name(payload["state_file"]); data = state_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != payload["state_sha256"]:
            raise ValueError("specialist mind state checksum mismatch")
        mind = cls(payload["family"], communicate=payload["communicate"])
        mind.team = pickle.loads(data); mind.episodes = int(payload["episodes"])
        mind.steps = int(payload["steps"]); mind.last_status = mind.team.status(); return mind

