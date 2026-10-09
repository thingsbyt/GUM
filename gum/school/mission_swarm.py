"""Persistent four-member mission team for the general recurrent learner.

The team is biological scaffolding, not a task specialist.  Members have
durable identities and separate weights, watch the same public experience,
publish reward-grounded experience capsules, and preserve every post-mission
brain snapshot.  No member receives hidden state, semantic action names, event
labels, coordinates, routes, or authored solutions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from typing import Any
import uuid

import numpy as np

from gum.lineage import HashLedger, canonical, file_sha256
from gum.protocol import PublicWorldSpec, Transition
from gum.storage import atomic_write_bytes, atomic_write_json

from .recurrent_meta import RecurrentCausalLearner


TEAM_SIZE = 4
MANIFEST_FILENAME = "MISSION_SWARM.json"
LEDGER_FILENAME = "MISSION_LEDGER.jsonl"
KNOWLEDGE_FILENAME = "KNOWLEDGE_LEDGER.jsonl"


class MissionSwarmError(ValueError):
    pass


@dataclass(frozen=True)
class MemberIdentity:
    member_id: str
    name: str
    role: str
    generation: int
    parent_id: str | None
    seed: int
    born_at_utc: str
    fingerprint: str

    @classmethod
    def create(cls, *, index: int, seed: int, role: str) -> "MemberIdentity":
        body = {
            "format": "gum-mission-member-identity-v1",
            "member_id": str(uuid.uuid4()),
            "name": f"Fortis-{index + 1}",
            "role": role,
            "generation": 0,
            "parent_id": None,
            "seed": int(seed),
            "born_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        fingerprint = hashlib.sha256(canonical(body)).hexdigest()
        values = {key: body[key] for key in body if key != "format"}
        return cls(**values, fingerprint=fingerprint)

    def to_json(self) -> dict[str, Any]:
        return {"format": "gum-mission-member-identity-v1", **asdict(self)}


@dataclass
class _Member:
    identity: MemberIdentity
    learner: RecurrentCausalLearner
    missions: int = 0
    captain_missions: int = 0
    successes: int = 0
    shared_imports: int = 0


class MissionSwarm:
    """Four persistent learners that watch, communicate, and learn as a team."""

    format = "gum-school-mission-swarm-v1"
    roles = ("pathfinder", "challenger", "modeler", "keeper")
    exploration_profiles = (
        {"evaluation_temperature": 0.55, "episodic_action_exploration_mix": 0.55},
        {"evaluation_temperature": 0.75, "episodic_action_exploration_mix": 0.75},
        {"evaluation_temperature": 1.00, "episodic_action_exploration_mix": 0.90},
        {"evaluation_temperature": 1.20, "episodic_action_exploration_mix": 1.00},
    )

    def __init__(self, root: Path, members: list[_Member], *, share_epochs: int = 1):
        if len(members) != TEAM_SIZE:
            raise MissionSwarmError("a mission swarm must contain exactly four members")
        if len({member.identity.member_id for member in members}) != TEAM_SIZE:
            raise MissionSwarmError("mission member identities must be unique")
        if share_epochs < 1:
            raise MissionSwarmError("share_epochs must be positive")
        self.root = Path(root)
        self.members = members
        self.share_epochs = int(share_epochs)
        self.mission_ledger = HashLedger(self.root / LEDGER_FILENAME)
        self.knowledge_ledger = HashLedger(self.root / KNOWLEDGE_FILENAME)
        self.completed_missions = 0
        self.communication_rounds = 0
        self.strategy_revisions = 0
        self._failure_counts: dict[str, int] = {}
        self._last_captains: dict[str, int] = {}
        self._active: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        source_policy: Path,
        seeds: tuple[int, int, int, int] = (8_650_101, 8_650_211, 8_650_307, 8_650_401),
        device: str = "cpu",
        share_epochs: int = 1,
    ) -> "MissionSwarm":
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"mission swarm directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        source = RecurrentCausalLearner.load(Path(source_policy), device=device)
        members = []
        for index, (seed, role, profile) in enumerate(
            zip(seeds, cls.roles, cls.exploration_profiles, strict=True)
        ):
            config = replace(
                source.config,
                training_exploration_mix=0.10,
                episodic_novelty_coefficient=0.02,
                **profile,
            )
            learner = RecurrentCausalLearner(seed, config=config, device=device)
            learner.policy.load_state_dict(source.policy.state_dict())
            identity = MemberIdentity.create(index=index, seed=seed, role=role)
            identity_path = root / "members" / identity.member_id / "IDENTITY.json"
            atomic_write_json(identity_path, identity.to_json(), backup=False, sort_keys=True)
            members.append(_Member(identity, learner))
        result = cls(root, members, share_epochs=share_epochs)
        result.mission_ledger.append(
            "swarm-born",
            {
                "team_size": TEAM_SIZE,
                "source_policy_sha256": f"sha256:{file_sha256(source_policy)}",
                "members": [member.identity.to_json() for member in members],
                "identity_fingerprints_are_authentication_signatures": False,
            },
        )
        result._checkpoint("birth")
        return result

    @staticmethod
    def _observation_key(observation: Any) -> str:
        return RecurrentCausalLearner._observation_key(np.asarray(observation))

    @staticmethod
    def _mission_fingerprint(spec: PublicWorldSpec, observation: Any) -> str:
        body = {
            "adapter": spec.adapter,
            "family": spec.family,
            "action_count": spec.action_count,
            "observation_shape": list(spec.observation_shape),
            "initial_observation": MissionSwarm._observation_key(observation),
        }
        return hashlib.sha256(canonical(body)).hexdigest()

    def _select_captain(self, fingerprint: str) -> int:
        failures = self._failure_counts.get(fingerprint, 0)
        if failures and fingerprint in self._last_captains:
            return (self._last_captains[fingerprint] + 1) % TEAM_SIZE
        order = sorted(
            range(TEAM_SIZE),
            key=lambda index: (self.members[index].captain_missions, index),
        )
        return order[0]

    def begin_mission(
        self,
        spec: PublicWorldSpec,
        observation: Any,
        *,
        training: bool,
        objective: str = "maximize public reward while learning visible consequences",
    ) -> dict[str, Any]:
        if self._active is not None:
            raise MissionSwarmError("finish the active mission before starting another")
        fingerprint = self._mission_fingerprint(spec, observation)
        captain = self._select_captain(fingerprint)
        self._last_captains[fingerprint] = captain
        mission_id = str(uuid.uuid4())
        modes = []
        for index, member in enumerate(self.members):
            member_training = bool(training and index == captain)
            member.learner.begin(spec, observation, training=member_training)
            modes.append(member_training)
        self._active = {
            "mission_id": mission_id,
            "fingerprint": fingerprint,
            "objective": str(objective),
            "training": bool(training),
            "captain": captain,
            "member_training": modes,
            "spec": spec,
            "observations": [],
            "actions": [],
            "rewards": [],
            "proposal_disagreements": 0,
            "proposal_rows": [],
        }
        payload = {
            "mission_id": mission_id,
            "mission_fingerprint": fingerprint,
            "objective": str(objective),
            "training": bool(training),
            "captain_id": self.members[captain].identity.member_id,
            "attempt": self._failure_counts.get(fingerprint, 0) + 1,
            "learner_visible_inputs": [
                "pixels", "previous anonymous action", "scalar reward",
                "termination", "recurrent memory",
            ],
        }
        self.mission_ledger.append("mission-started", payload, run_id=mission_id)
        return payload

    def act(self, observation: Any) -> int:
        if self._active is None:
            raise MissionSwarmError("begin_mission must precede act")
        proposals = []
        confidences = []
        for index, member in enumerate(self.members):
            action = member.learner.act(
                observation,
                training=self._active["member_training"][index],
            )
            proposals.append(int(action))
            confidences.append(float(member.learner.confidence()))
        if self._active["training"]:
            selected = proposals[self._active["captain"]]
        else:
            action_count = self._active["spec"].action_count
            weights = np.asarray(confidences, dtype=np.float64) + 0.25
            votes = np.zeros(action_count, dtype=np.float64)
            for action, weight in zip(proposals, weights, strict=True):
                votes[action] += weight
            selected = int(np.flatnonzero(votes == votes.max())[0])
        self._active["proposal_disagreements"] += int(len(set(proposals)) > 1)
        self._active["proposal_rows"].append({
            "proposals": proposals,
            "confidences": confidences,
            "selected": selected,
        })
        self._active["observations"].append(np.asarray(observation).copy())
        return selected

    def observe(self, action: int, transition: Transition) -> None:
        if self._active is None:
            raise MissionSwarmError("begin_mission and act must precede observe")
        for index, member in enumerate(self.members):
            member.learner.observe(
                action,
                transition,
                training=self._active["member_training"][index],
            )
        self._active["actions"].append(int(action))
        self._active["rewards"].append(float(transition.reward))

    def finish_mission(self, *, success: bool) -> dict[str, Any]:
        if self._active is None:
            raise MissionSwarmError("there is no active mission")
        active = self._active
        for index, member in enumerate(self.members):
            member.learner.finish_episode(
                training=active["member_training"][index]
            )
            member.missions += 1
            member.successes += int(success)
        self.members[active["captain"]].captain_missions += 1
        if active["training"]:
            self.members[active["captain"]].learner.flush_training()
        shared = []
        if active["training"] and success and active["actions"]:
            trajectory = {
                "observations": active["observations"],
                "actions": active["actions"],
                "rewards": active["rewards"],
                "action_count": active["spec"].action_count,
            }
            for index, member in enumerate(self.members):
                if index == active["captain"]:
                    continue
                result = member.learner.fit_self_imitation(
                    [trajectory], epochs=self.share_epochs, batch_episodes=1
                )
                member.shared_imports += 1
                shared.append({
                    "recipient_id": member.identity.member_id,
                    "updates": int(result["updates"]),
                })
            self.communication_rounds += 1
        if not success:
            fingerprint = active["fingerprint"]
            self._failure_counts[fingerprint] = self._failure_counts.get(fingerprint, 0) + 1
            self.strategy_revisions += 1
        experience = self._preserve_experience(active)
        capsule = {
            "mission_id": active["mission_id"],
            "mission_fingerprint": active["fingerprint"],
            "objective": active["objective"],
            "captain_id": self.members[active["captain"]].identity.member_id,
            "success": bool(success),
            "interactions": len(active["actions"]),
            "return": float(sum(active["rewards"])),
            "proposal_disagreements": int(active["proposal_disagreements"]),
            "action_histogram": np.bincount(
                active["actions"], minlength=active["spec"].action_count
            ).tolist(),
            "experience": experience,
            "shared_imports": shared,
            "strategy_changed_before_exact_retry": not success,
            "next_captain_rotates_after_failure": not success,
            "prohibited_information_used": False,
        }
        self.knowledge_ledger.append(
            "experience-capsule", capsule, run_id=active["mission_id"]
        )
        self.mission_ledger.append(
            "mission-completed", capsule, run_id=active["mission_id"]
        )
        self.completed_missions += 1
        self._active = None
        self._checkpoint(capsule["mission_id"])
        return capsule

    def _preserve_experience(self, active: dict[str, Any]) -> dict[str, str]:
        """Atomically preserve every public transition and team proposal."""
        rows = active["proposal_rows"]
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            observations=np.asarray(active["observations"], dtype=np.uint8),
            actions=np.asarray(active["actions"], dtype=np.int64),
            rewards=np.asarray(active["rewards"], dtype=np.float32),
            proposals=np.asarray(
                [row["proposals"] for row in rows], dtype=np.int64
            ),
            confidences=np.asarray(
                [row["confidences"] for row in rows], dtype=np.float32
            ),
            selected=np.asarray(
                [row["selected"] for row in rows], dtype=np.int64
            ),
        )
        path = self.root / "experiences" / f"{active['mission_id']}.npz"
        atomic_write_bytes(path, buffer.getvalue(), backup=False)
        return {
            "path": path.relative_to(self.root).as_posix(),
            "sha256": f"sha256:{file_sha256(path)}",
        }

    def _checkpoint(self, label: str) -> Path:
        snapshot = self.root / "snapshots" / label
        if snapshot.exists():
            raise FileExistsError(f"mission checkpoint already exists: {snapshot}")
        snapshot.mkdir(parents=True)
        brains = {}
        for member in self.members:
            path = snapshot / f"{member.identity.member_id}.pt"
            member.learner.save(path)
            brains[member.identity.member_id] = {
                "path": path.relative_to(self.root).as_posix(),
                "sha256": f"sha256:{file_sha256(path)}",
            }
        payload = {
            "format": self.format,
            "checkpoint": label,
            "completed_missions": self.completed_missions,
            "communication_rounds": self.communication_rounds,
            "strategy_revisions": self.strategy_revisions,
            "share_epochs": self.share_epochs,
            "members": [
                {
                    "identity": member.identity.to_json(),
                    "missions": member.missions,
                    "captain_missions": member.captain_missions,
                    "successes": member.successes,
                    "shared_imports": member.shared_imports,
                    "brain": brains[member.identity.member_id],
                }
                for member in self.members
            ],
            "failure_counts": dict(sorted(self._failure_counts.items())),
            "last_captains": dict(sorted(self._last_captains.items())),
            "mission_ledger": self.mission_ledger.verify(),
            "knowledge_ledger": self.knowledge_ledger.verify(),
        }
        checkpoint_manifest = snapshot / MANIFEST_FILENAME
        atomic_write_json(checkpoint_manifest, payload, backup=False, sort_keys=True)
        pointer = {
            "format": "gum-school-mission-swarm-pointer-v1",
            "checkpoint": label,
            "manifest": checkpoint_manifest.relative_to(self.root).as_posix(),
            "manifest_sha256": f"sha256:{file_sha256(checkpoint_manifest)}",
        }
        atomic_write_json(self.root / MANIFEST_FILENAME, pointer, backup=False, sort_keys=True)
        return checkpoint_manifest

    @classmethod
    def load(cls, root: Path, *, device: str = "cpu") -> "MissionSwarm":
        root = Path(root)
        pointer = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        if pointer.get("format") != "gum-school-mission-swarm-pointer-v1":
            raise MissionSwarmError("unsupported mission swarm pointer")
        manifest_path = root / pointer["manifest"]
        if f"sha256:{file_sha256(manifest_path)}" != pointer.get("manifest_sha256"):
            raise MissionSwarmError("mission swarm manifest hash differs")
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if value.get("format") != cls.format or len(value.get("members", [])) != TEAM_SIZE:
            raise MissionSwarmError("unsupported mission swarm checkpoint")
        members = []
        for row in value["members"]:
            identity_value = row["identity"]
            body = {key: identity_value[key] for key in MemberIdentity.__dataclass_fields__}
            identity = MemberIdentity(**body)
            identity_path = root / "members" / identity.member_id / "IDENTITY.json"
            saved_identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if saved_identity != identity.to_json():
                raise MissionSwarmError("member identity record differs")
            identity_body = {
                key: value for key, value in saved_identity.items()
                if key != "fingerprint"
            }
            if hashlib.sha256(canonical(identity_body)).hexdigest() != identity.fingerprint:
                raise MissionSwarmError("member identity fingerprint differs")
            brain_path = root / row["brain"]["path"]
            if f"sha256:{file_sha256(brain_path)}" != row["brain"]["sha256"]:
                raise MissionSwarmError("member brain hash differs")
            members.append(_Member(
                identity,
                RecurrentCausalLearner.load(brain_path, device=device),
                missions=int(row["missions"]),
                captain_missions=int(row.get("captain_missions", 0)),
                successes=int(row["successes"]),
                shared_imports=int(row["shared_imports"]),
            ))
        result = cls(root, members, share_epochs=int(value["share_epochs"]))
        result.completed_missions = int(value["completed_missions"])
        result.communication_rounds = int(value["communication_rounds"])
        result.strategy_revisions = int(value["strategy_revisions"])
        result._failure_counts = {
            str(key): int(number) for key, number in value["failure_counts"].items()
        }
        result._last_captains = {
            str(key): int(number)
            for key, number in value.get("last_captains", {}).items()
        }
        if not result.mission_ledger.verify()["valid"]:
            raise MissionSwarmError("mission ledger verification failed")
        if not result.knowledge_ledger.verify()["valid"]:
            raise MissionSwarmError("knowledge ledger verification failed")
        archive = result.verify_archive()
        if not archive["valid"]:
            raise MissionSwarmError(
                "mission experience archive verification failed: "
                + "; ".join(archive["errors"])
            )
        return result

    def verify_archive(self) -> dict[str, Any]:
        """Verify every retained mission experience referenced by knowledge."""
        errors = []
        records = 0
        if self.knowledge_ledger.path.exists():
            with self.knowledge_ledger.path.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("event") != "experience-capsule":
                        continue
                    records += 1
                    experience = row.get("payload", {}).get("experience", {})
                    path_value = experience.get("path")
                    expected = experience.get("sha256")
                    if not isinstance(path_value, str) or not isinstance(expected, str):
                        errors.append(f"line {number}: missing experience reference")
                        continue
                    path = self.root / path_value
                    if not path.is_file():
                        errors.append(f"line {number}: experience file is missing")
                    elif f"sha256:{file_sha256(path)}" != expected:
                        errors.append(f"line {number}: experience hash differs")
        return {"valid": not errors, "records": records, "errors": errors}

    def status(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "team_size": TEAM_SIZE,
            "completed_missions": self.completed_missions,
            "communication_rounds": self.communication_rounds,
            "strategy_revisions": self.strategy_revisions,
            "active_mission": None if self._active is None else self._active["mission_id"],
            "members": [
                {
                    "member_id": member.identity.member_id,
                    "name": member.identity.name,
                    "role": member.identity.role,
                    "fingerprint": member.identity.fingerprint,
                    "missions": member.missions,
                    "captain_missions": member.captain_missions,
                    "successes": member.successes,
                    "shared_imports": member.shared_imports,
                }
                for member in self.members
            ],
            "mission_ledger": self.mission_ledger.verify(),
            "knowledge_ledger": self.knowledge_ledger.verify(),
            "experience_archive": self.verify_archive(),
        }
