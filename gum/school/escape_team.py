"""Four independently acting persistent learners for the escape chamber.

This is intentionally separate from :mod:`mission_swarm`, whose four policies
propose actions for one body. Here each identity owns one learner, one recurrent
state, one executed action stream, one reward stream, and one saved brain.
Sharing is explicitly disabled in protocol v1; a later implementation must add
an auditable bounded exchange rather than silently reusing captain imitation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Callable
import uuid

import numpy as np

from gum.lineage import HashLedger, canonical, file_sha256
from gum.storage import atomic_write_bytes, atomic_write_json

from .escape_chamber import (
    ACTION_COUNT,
    CONTRACT_VERSION,
    MEMBER_NAMES,
    MEMBER_SHAPES,
    ROOM_A_ADAPTER,
    TEAM_SIZE,
    EscapeChamberRoomA,
    RewardTable,
    make_escape_chamber,
)
from .recurrent_meta import RecurrentCausalLearner, RecurrentMetaConfig


MANIFEST_FILENAME = "ESCAPE_TEAM.json"
LEDGER_FILENAME = "EPISODE_LEDGER.jsonl"


class EscapeTeamError(ValueError):
    pass


@dataclass(frozen=True)
class EscapeMemberIdentity:
    member_id: str
    name: str
    shape: str
    seed: int
    generation: int
    parent_id: str | None
    born_at_utc: str
    fingerprint: str

    @classmethod
    def create(cls, index: int, seed: int) -> "EscapeMemberIdentity":
        body = {
            "format": "gum-escape-member-identity-v1",
            "member_id": str(uuid.uuid4()),
            "name": MEMBER_NAMES[index],
            "shape": MEMBER_SHAPES[index],
            "seed": int(seed),
            "generation": 0,
            "parent_id": None,
            "born_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        fingerprint = hashlib.sha256(canonical(body)).hexdigest()
        return cls(
            **{key: value for key, value in body.items() if key != "format"},
            fingerprint=fingerprint,
        )

    def to_json(self) -> dict[str, Any]:
        return {"format": "gum-escape-member-identity-v1", **asdict(self)}


@dataclass
class _EscapeMember:
    identity: EscapeMemberIdentity
    learner: RecurrentCausalLearner
    episodes: int = 0
    escapes: int = 0
    returns: float = 0.0


class EscapeTeam:
    format = "gum-cooperative-escape-team-v1"
    pointer_format = "gum-cooperative-escape-team-pointer-v1"

    def __init__(
        self,
        root: Path,
        members: list[_EscapeMember],
        *,
        reward_table: RewardTable,
        initialization: dict[str, Any],
    ):
        if len(members) != TEAM_SIZE:
            raise EscapeTeamError("an escape team must have four members")
        if len({member.identity.member_id for member in members}) != TEAM_SIZE:
            raise EscapeTeamError("escape member identities must be unique")
        self.root = Path(root)
        self.members = members
        self.reward_table = reward_table
        self.reward_table.validate()
        self.initialization = initialization
        self.sharing_mode = "off"
        self.ledger = HashLedger(self.root / LEDGER_FILENAME)
        self.completed_episodes = 0
        self.training_episodes = 0
        self.evaluation_episodes = 0
        self._running = False
        self._current_world: EscapeChamberRoomA | None = None

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        seeds: tuple[int, int, int, int] = (9_310_101, 9_310_211, 9_310_307, 9_310_401),
        source_policy: Path | None = None,
        config: RecurrentMetaConfig | None = None,
        exploration_overrides: dict[str, float] | None = None,
        device: str = "cpu",
        reward_table: RewardTable | None = None,
    ) -> "EscapeTeam":
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"escape team directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        table = reward_table or RewardTable()
        table.validate()

        inherited = None
        if source_policy is not None:
            source_policy = Path(source_policy)
            inherited = RecurrentCausalLearner.load(source_policy, device=device)
            base_config = inherited.config
            initialization = {
                "kind": "identical-inherited-policy-weights",
                "source_policy_sha256": f"sha256:{file_sha256(source_policy)}",
                "optimizer_state_inherited": False,
                "separate_copies_are_prior_independent_learning": False,
            }
        else:
            base_config = config or RecurrentMetaConfig()
            initialization = {
                "kind": "independent-random-weights",
                "source_policy_sha256": None,
                "optimizer_state_inherited": False,
                "separate_copies_are_prior_independent_learning": False,
            }
        shared_config = replace(
            base_config,
            training_exploration_mix=0.20,
            episodic_action_exploration_mix=0.50,
            episodic_novelty_coefficient=0.01,
        )
        overrides = dict(exploration_overrides or {})
        permitted = {
            "training_exploration_mix",
            "episodic_action_exploration_mix",
            "episodic_novelty_coefficient",
            "entropy_coefficient",
        }
        unexpected = sorted(set(overrides).difference(permitted))
        if unexpected:
            raise EscapeTeamError(
                "unsupported exploration override(s): " + ", ".join(unexpected)
            )
        if overrides:
            shared_config = replace(shared_config, **overrides)
        shared_config.validate()
        initialization["exploration_configuration"] = {
            key: getattr(shared_config, key) for key in sorted(permitted)
        }
        members = []
        for index, seed in enumerate(seeds):
            learner = RecurrentCausalLearner(seed, config=shared_config, device=device)
            if inherited is not None:
                learner.policy.load_state_dict(inherited.policy.state_dict())
            identity = EscapeMemberIdentity.create(index, seed)
            atomic_write_json(
                root / "members" / identity.member_id / "IDENTITY.json",
                identity.to_json(),
                backup=False,
                sort_keys=True,
            )
            members.append(_EscapeMember(identity, learner))
        result = cls(
            root,
            members,
            reward_table=table,
            initialization=initialization,
        )
        result.ledger.append(
            "team-born",
            {
                "members": [member.identity.to_json() for member in members],
                "initialization": initialization,
                "reward_table": table.to_json(),
                "sharing_mode": "off",
                "independent_bodies": True,
                "captain_or_joint_action_scheduler": False,
            },
        )
        result._checkpoint("birth")
        return result

    @property
    def current_world(self) -> EscapeChamberRoomA | None:
        return self._current_world

    def run_episode(
        self,
        *,
        seed: int,
        training: bool,
        horizon: int = 240,
        environment_adapter: str = ROOM_A_ADAPTER,
        on_step: Callable[[EscapeChamberRoomA, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if self._running:
            raise EscapeTeamError("an escape-team episode is already running")
        self._running = True
        episode_id = str(uuid.uuid4())
        world = make_escape_chamber(
            environment_adapter,
            seed=int(seed), horizon=int(horizon), reward_table=self.reward_table
        )
        self._current_world = world
        observations = world.reset()
        spec = world.public_spec()
        for member, observation in zip(self.members, observations, strict=True):
            member.learner.begin(spec, observation, training=training)

        observed_rows = []
        next_rows = []
        action_rows = []
        reward_rows = []
        terminated_rows = []
        truncated_rows = []
        active_before_rows = []
        active_after_rows = []
        state_hashes = []
        returns = np.zeros(TEAM_SIZE, dtype=np.float64)
        try:
            while True:
                actions = []
                for index, (member, observation) in enumerate(
                    zip(self.members, observations, strict=True)
                ):
                    if world.escaped[index]:
                        actions.append(None)
                    else:
                        actions.append(member.learner.act(observation, training=training))
                step = world.step(actions)
                observed_rows.append(np.stack(observations))
                next_rows.append(np.stack(step.observations))
                action_rows.append([-1 if value is None else value for value in actions])
                reward_rows.append(step.rewards)
                terminated_rows.append(step.terminated)
                truncated_rows.append(step.truncated)
                active_before_rows.append(step.active_before)
                active_after_rows.append(step.active_after)
                state_hashes.append(world.audit_state()["state_sha256"].removeprefix("sha256:"))
                returns += np.asarray(step.rewards)

                for index, member in enumerate(self.members):
                    if step.active_before[index]:
                        assert actions[index] is not None
                        member.learner.observe(
                            actions[index], step.transition_for(index), training=training
                        )
                    else:
                        member.learner.credit_delayed_reward(
                            step.rewards[index], training=training
                        )
                if on_step is not None:
                    on_step(world, {
                        "episode_id": episode_id,
                        "training": bool(training),
                        "tick": world.steps,
                        "actions": action_rows[-1],
                        "rewards": list(step.rewards),
                        "state_sha256": state_hashes[-1],
                    })
                observations = step.observations
                if step.terminated or step.truncated:
                    break

            updates = []
            for index, member in enumerate(self.members):
                finish = member.learner.finish_episode(training=training)
                flush = member.learner.flush_training() if training else None
                updates.append(flush or finish)
                member.episodes += 1
                member.escapes += int(world.escaped[index])
                member.returns += float(returns[index])

            archive = self._archive_episode(
                episode_id,
                seed=int(seed),
                training=training,
                environment_adapter=environment_adapter,
                observations=observed_rows,
                next_observations=next_rows,
                actions=action_rows,
                rewards=reward_rows,
                terminated=terminated_rows,
                truncated=truncated_rows,
                active_before=active_before_rows,
                active_after=active_after_rows,
                state_hashes=state_hashes,
            )
            capsule = {
                "episode_id": episode_id,
                "seed": int(seed),
                "training": bool(training),
                "treatment": self.reward_table.treatment,
                "environment_adapter": environment_adapter,
                "sharing_mode": self.sharing_mode,
                "steps": world.steps,
                "escaped_count": len(world.escape_order),
                "escape_order": [
                    self.members[index].identity.member_id for index in world.escape_order
                ],
                "legal_maximum_reached": len(world.escape_order) == TEAM_SIZE - 1,
                "returns": returns.tolist(),
                "updates": updates,
                "archive": archive,
                "policy_input_fields": [
                    "member_view_pixels", "previous_executed_anonymous_action",
                    "scalar_reward", "episode_boundary", "own_recurrent_state",
                ],
                "audit_state_given_to_policy": False,
                "scripted_control_used": False,
                "llm_control_used": False,
            }
            self.ledger.append("episode-completed", capsule, run_id=episode_id)
            self.completed_episodes += 1
            self.training_episodes += int(training)
            self.evaluation_episodes += int(not training)
            self._checkpoint(episode_id)
            return capsule
        finally:
            self._running = False

    def _archive_episode(
        self,
        episode_id: str,
        *,
        seed: int,
        training: bool,
        environment_adapter: str,
        observations,
        next_observations,
        actions,
        rewards,
        terminated,
        truncated,
        active_before,
        active_after,
        state_hashes,
    ) -> dict[str, Any]:
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            observations=np.asarray(observations, dtype=np.uint8),
            next_observations=np.asarray(next_observations, dtype=np.uint8),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            terminated=np.asarray(terminated, dtype=np.bool_),
            truncated=np.asarray(truncated, dtype=np.bool_),
            active_before=np.asarray(active_before, dtype=np.bool_),
            active_after=np.asarray(active_after, dtype=np.bool_),
            state_hashes=np.asarray(state_hashes, dtype="S64"),
        )
        path = self.root / "episodes" / f"{episode_id}.npz"
        atomic_write_bytes(path, buffer.getvalue(), backup=False)
        return {
            "format": "gum-cooperative-escape-transitions-v1",
            "path": path.relative_to(self.root).as_posix(),
            "sha256": f"sha256:{file_sha256(path)}",
            "contract_version": CONTRACT_VERSION,
            "environment_seed": int(seed),
            "environment_adapter": environment_adapter,
            "training": bool(training),
            "joint_ticks": len(actions),
            "members": [member.identity.member_id for member in self.members],
            "provenance": {
                "environment_source_sha256": f"sha256:{file_sha256(Path(__file__).with_name('escape_chamber.py'))}",
                "team_source_sha256": f"sha256:{file_sha256(Path(__file__))}",
            },
        }

    def consolidate_rewarded_episode(
        self,
        episode_id: str,
        *,
        epochs: int = 25,
    ) -> dict[str, Any]:
        """Rehearse one genuinely rewarded trajectory and persist the result.

        Selection uses only the public scalar outcome already recorded in the
        episode capsule. Training receives member pixels, executed anonymous
        actions, and scalar rewards; it receives no coordinates, mechanics,
        semantic actions, roles, or authored targets.
        """
        if self._running:
            raise EscapeTeamError("cannot consolidate while an episode is running")
        if epochs < 1:
            raise EscapeTeamError("consolidation epochs must be positive")
        capsule = None
        with self.ledger.path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                payload = row.get("payload", {})
                if (
                    row.get("event") == "episode-completed"
                    and payload.get("episode_id") == episode_id
                ):
                    capsule = payload
                    break
        if capsule is None:
            raise EscapeTeamError("episode is not present in the verified ledger")
        if not capsule.get("training"):
            raise EscapeTeamError("evaluation experience cannot be consolidated")
        public_returns = [float(value) for value in capsule.get("returns", [])]
        if not public_returns or max(public_returns) <= 0.0:
            raise EscapeTeamError("consolidation requires genuine positive task reward")
        reference = capsule["archive"]
        archive_path = (self.root / reference["path"]).resolve()
        try:
            archive_path.relative_to(self.root.resolve())
        except ValueError as error:
            raise EscapeTeamError("transition archive path escapes team directory") from error
        if f"sha256:{file_sha256(archive_path)}" != reference.get("sha256"):
            raise EscapeTeamError("transition archive hash differs")
        with np.load(archive_path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}

        updates = []
        for index, member in enumerate(self.members):
            active = arrays["active_before"][:, index].astype(bool)
            actions = arrays["actions"][:, index][active].astype(int).tolist()
            trajectory = {
                "observations": [
                    value for value in arrays["observations"][:, index][active]
                ],
                "actions": actions,
                "rewards": arrays["rewards"][:, index][active].astype(float).tolist(),
                "action_count": ACTION_COUNT,
            }
            if not actions or any(action < 0 for action in actions):
                raise EscapeTeamError("rewarded member trajectory is incomplete")
            updates.append(member.learner.fit_self_imitation(
                [trajectory], epochs=epochs, batch_episodes=1
            ))
        consolidation_id = str(uuid.uuid4())
        record = {
            "consolidation_id": consolidation_id,
            "episode_id": episode_id,
            "archive": reference,
            "selection": "positive-public-scalar-task-return",
            "epochs": int(epochs),
            "updates": updates,
            "semantic_labels_used": False,
            "scripted_targets_used": False,
            "sharing_used": False,
        }
        self.ledger.append(
            "reward-selected-consolidation", record, run_id=consolidation_id
        )
        self._checkpoint(f"consolidation-{consolidation_id}")
        return record

    def _checkpoint(self, label: str) -> Path:
        snapshot = self.root / "snapshots" / label
        if snapshot.exists():
            raise FileExistsError(f"escape-team checkpoint already exists: {snapshot}")
        snapshot.mkdir(parents=True)
        rows = []
        for member in self.members:
            brain = snapshot / f"{member.identity.member_id}.pt"
            member.learner.save(brain)
            rows.append({
                "identity": member.identity.to_json(),
                "episodes": member.episodes,
                "escapes": member.escapes,
                "returns": member.returns,
                "brain": {
                    "path": brain.relative_to(self.root).as_posix(),
                    "sha256": f"sha256:{file_sha256(brain)}",
                },
            })
        payload = {
            "format": self.format,
            "checkpoint": label,
            "completed_episodes": self.completed_episodes,
            "training_episodes": self.training_episodes,
            "evaluation_episodes": self.evaluation_episodes,
            "sharing_mode": self.sharing_mode,
            "reward_table": self.reward_table.to_json(),
            "initialization": self.initialization,
            "members": rows,
            "episode_ledger": self.ledger.verify(),
        }
        manifest = snapshot / MANIFEST_FILENAME
        atomic_write_json(manifest, payload, backup=False, sort_keys=True)
        pointer = {
            "format": self.pointer_format,
            "checkpoint": label,
            "manifest": manifest.relative_to(self.root).as_posix(),
            "manifest_sha256": f"sha256:{file_sha256(manifest)}",
        }
        atomic_write_json(self.root / MANIFEST_FILENAME, pointer, backup=False, sort_keys=True)
        return manifest

    @staticmethod
    def _completed_record_count(path: Path) -> int:
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip() and json.loads(line).get("event") == "episode-completed":
                    count += 1
        return count

    @classmethod
    def _manifest_errors(
        cls,
        root: Path,
        path: Path,
        value: dict[str, Any],
        ledger_state: dict[str, Any],
        completed_records: int,
    ) -> list[str]:
        errors = []
        if value.get("format") != cls.format:
            errors.append("unsupported checkpoint format")
        if value.get("checkpoint") != path.parent.name:
            errors.append("checkpoint label differs from directory")
        embedded = value.get("episode_ledger", {})
        if not isinstance(embedded, dict):
            embedded = {}
        if embedded.get("records") != ledger_state.get("records"):
            errors.append("episode ledger record count differs")
        if embedded.get("head") != ledger_state.get("head"):
            errors.append("episode ledger head differs")
        completed = value.get("completed_episodes")
        if completed != completed_records:
            errors.append("completed episode count differs from ledger")
        members = value.get("members", [])
        if not isinstance(members, list) or len(members) != TEAM_SIZE:
            return errors + ["checkpoint does not contain four members"]
        for row in members:
            if row.get("episodes") != completed:
                errors.append("member episode count differs from team count")
            brain = row.get("brain", {})
            brain_path = (root / brain.get("path", "")).resolve()
            try:
                brain_path.relative_to(root)
            except ValueError:
                errors.append("brain path escapes team directory")
                continue
            if not brain_path.is_file():
                errors.append("member brain is missing")
            elif f"sha256:{file_sha256(brain_path)}" != brain.get("sha256"):
                errors.append("member brain hash differs")
        return errors

    @classmethod
    def _resolve_checkpoint(cls, root: Path) -> dict[str, Any]:
        root = root.resolve()
        ledger_state = HashLedger(root / LEDGER_FILENAME).verify()
        if not ledger_state["valid"]:
            raise EscapeTeamError("episode ledger verification failed")
        completed = cls._completed_record_count(root / LEDGER_FILENAME)
        pointer_errors = []
        try:
            pointer = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
            if pointer.get("format") != cls.pointer_format:
                raise EscapeTeamError("unsupported escape-team pointer")
            manifest = (root / pointer["manifest"]).resolve()
            manifest.relative_to(root)
            if f"sha256:{file_sha256(manifest)}" != pointer.get("manifest_sha256"):
                raise EscapeTeamError("escape-team manifest hash differs")
            value = json.loads(manifest.read_text(encoding="utf-8"))
            pointer_errors = cls._manifest_errors(
                root, manifest, value, ledger_state, completed
            )
            if not pointer_errors:
                return value
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            pointer_errors = [str(error)]

        matches = []
        for manifest in sorted((root / "snapshots").glob(f"*/{MANIFEST_FILENAME}")):
            try:
                value = json.loads(manifest.read_text(encoding="utf-8"))
                if not cls._manifest_errors(
                    root, manifest.resolve(), value, ledger_state, completed
                ):
                    matches.append((manifest.resolve(), value))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        if len(matches) != 1:
            raise EscapeTeamError(
                "checkpoint/ledger inconsistency detected; "
                f"found {len(matches)} recoverable checkpoints "
                f"({'; '.join(pointer_errors)})"
            )
        manifest, value = matches[0]
        atomic_write_json(
            root / MANIFEST_FILENAME,
            {
                "format": cls.pointer_format,
                "checkpoint": value["checkpoint"],
                "manifest": manifest.relative_to(root).as_posix(),
                "manifest_sha256": f"sha256:{file_sha256(manifest)}",
                "recovered_after_incomplete_pointer_update": True,
            },
            backup=False,
            sort_keys=True,
        )
        return value

    @classmethod
    def load(cls, root: Path, *, device: str = "cpu") -> "EscapeTeam":
        root = Path(root).resolve()
        value = cls._resolve_checkpoint(root)
        members = []
        for row in value["members"]:
            identity_value = row["identity"]
            identity = EscapeMemberIdentity(**{
                key: identity_value[key] for key in EscapeMemberIdentity.__dataclass_fields__
            })
            identity_path = root / "members" / identity.member_id / "IDENTITY.json"
            saved_identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if saved_identity != identity.to_json():
                raise EscapeTeamError("member identity record differs")
            fingerprint_body = {
                key: item for key, item in saved_identity.items() if key != "fingerprint"
            }
            if hashlib.sha256(canonical(fingerprint_body)).hexdigest() != identity.fingerprint:
                raise EscapeTeamError("member identity fingerprint differs")
            brain = root / row["brain"]["path"]
            method = value["initialization"].get("learning_method")
            learner_type = RecurrentCausalLearner
            if method in {"independent-ppo", "gum-elapsed-reward-reference"}:
                from .independent_ppo import IndependentPPOLearner, ElapsedRewardGUMLearner
                learner_type = (IndependentPPOLearner if method == "independent-ppo"
                                else ElapsedRewardGUMLearner)
            members.append(_EscapeMember(
                identity,
                learner_type.load(brain, device=device),
                episodes=int(row["episodes"]),
                escapes=int(row["escapes"]),
                returns=float(row["returns"]),
            ))
        result = cls(
            root,
            members,
            reward_table=RewardTable(**value["reward_table"]),
            initialization=value["initialization"],
        )
        result.completed_episodes = int(value["completed_episodes"])
        result.training_episodes = int(value["training_episodes"])
        result.evaluation_episodes = int(value["evaluation_episodes"])
        if value.get("sharing_mode") != "off":
            raise EscapeTeamError("protocol v1 only supports enforced sharing-off mode")
        archive = result.verify_archives()
        if not archive["valid"]:
            raise EscapeTeamError("episode archive verification failed: " + "; ".join(archive["errors"]))
        return result

    def verify_archives(self) -> dict[str, Any]:
        errors = []
        records = 0
        if self.ledger.path.exists():
            with self.ledger.path.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("event") != "episode-completed":
                        continue
                    records += 1
                    reference = row.get("payload", {}).get("archive", {})
                    path = self.root / reference.get("path", "")
                    if reference.get("format") != "gum-cooperative-escape-transitions-v1":
                        errors.append(f"line {number}: unsupported transition archive")
                        continue
                    if not path.is_file():
                        errors.append(f"line {number}: transition archive is missing")
                        continue
                    if f"sha256:{file_sha256(path)}" != reference.get("sha256"):
                        errors.append(f"line {number}: transition archive hash differs")
                        continue
                    try:
                        with np.load(path, allow_pickle=False) as archive:
                            required = {
                                "observations", "next_observations", "actions", "rewards",
                                "terminated", "truncated", "active_before", "active_after",
                                "state_hashes",
                            }
                            missing = required.difference(archive.files)
                            if missing:
                                errors.append(f"line {number}: archive fields are incomplete")
                            else:
                                lengths = {len(archive[name]) for name in required}
                                if lengths != {int(reference["joint_ticks"])}:
                                    errors.append(f"line {number}: archive lengths differ")
                                if archive["observations"].shape != archive["next_observations"].shape:
                                    errors.append(f"line {number}: observation shapes differ")
                                if archive["actions"].shape[1:] != (TEAM_SIZE,):
                                    errors.append(f"line {number}: joint action shape differs")
                    except (OSError, ValueError) as error:
                        errors.append(f"line {number}: archive unreadable: {error}")
        return {"valid": not errors, "records": records, "errors": errors}

    def status(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "team_size": TEAM_SIZE,
            "independent_bodies": True,
            "sharing_mode": self.sharing_mode,
            "completed_episodes": self.completed_episodes,
            "training_episodes": self.training_episodes,
            "evaluation_episodes": self.evaluation_episodes,
            "initialization": self.initialization,
            "reward_table": self.reward_table.to_json(),
            "members": [
                {
                    "identity": member.identity.to_json(),
                    "episodes": member.episodes,
                    "escapes": member.escapes,
                    "return": member.returns,
                    "learner": member.learner.status(),
                }
                for member in self.members
            ],
            "episode_ledger": self.ledger.verify(),
            "archives": self.verify_archives(),
        }
