"""Read-only diagnostics for preserved cooperative escape episodes.

Diagnostics may inspect authoritative replay state, but none of the semantic
measurements produced here are returned to a learner or used as training
reward.  This keeps measurement separate from intervention.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gum.lineage import HashLedger, file_sha256

from .escape_chamber import (
    ACTION_COUNT,
    ROOM_A_ADAPTER,
    TEAM_SIZE,
    WAIT,
    EscapeChamberRoomA,
    RewardTable,
    make_escape_chamber,
)
from .escape_team import LEDGER_FILENAME, EscapeTeam, EscapeTeamError
from .recurrent_meta import RecurrentCausalLearner


DIAGNOSTIC_FORMAT = "gum-cooperative-escape-diagnostics-v1"
PERCEPTION_FORMAT = "gum-cooperative-escape-perception-probe-v1"


def _episode_records(root: Path) -> list[dict[str, Any]]:
    ledger = root / LEDGER_FILENAME
    state = HashLedger(ledger).verify()
    if not state["valid"]:
        raise EscapeTeamError("cannot diagnose an invalid episode ledger")
    records = []
    with ledger.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "episode-completed":
                records.append(row["payload"])
    return records


def _load_archive(root: Path, capsule: dict[str, Any]) -> dict[str, np.ndarray]:
    reference = capsule.get("archive", {})
    path = (root / reference.get("path", "")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise EscapeTeamError("transition archive path escapes team directory") from error
    if not path.is_file():
        raise EscapeTeamError("transition archive is missing")
    if f"sha256:{file_sha256(path)}" != reference.get("sha256"):
        raise EscapeTeamError("transition archive hash differs")
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def _new_member_metrics() -> dict[str, Any]:
    return {
        "plate_entries": 0,
        "plate_ticks": 0,
        "longest_plate_dwell": 0,
        "gate_inside_entries": 0,
        "gate_inside_ticks": 0,
        "gate_proposals": 0,
        "gate_crossings": 0,
        "exit_proposals": 0,
        "escapes": 0,
        "wait_actions": 0,
        "longest_wait_streak": 0,
        "cancelled_moves": 0,
        "minimum_plate_distance": None,
        "minimum_gate_distance": None,
        "anonymous_action_histogram": [0] * ACTION_COUNT,
        "semantic_action_histogram": {
            "wait": 0, "north": 0, "east": 0, "south": 0, "west": 0,
        },
    }


def _minimum_distance(current: int | None, position, target) -> int | None:
    if position is None:
        return current
    distance = abs(position[0] - target[0]) + abs(position[1] - target[1])
    return distance if current is None else min(current, distance)


def _replay_metrics(
    capsule: dict[str, Any],
    arrays: dict[str, np.ndarray],
    reward_table: RewardTable,
) -> dict[str, Any]:
    actions = arrays["actions"]
    horizon = len(actions)
    adapter = capsule["archive"].get("environment_adapter", ROOM_A_ADAPTER)
    world = make_escape_chamber(
        adapter,
        seed=int(capsule["archive"]["environment_seed"]),
        horizon=horizon,
        reward_table=reward_table,
    )
    observations = world.reset()
    members = [_new_member_metrics() for _ in range(TEAM_SIZE)]
    plate_streaks = [0] * TEAM_SIZE
    wait_streaks = [0] * TEAM_SIZE
    gate_inside_before = [position == world.gate_inside for position in world.positions]
    previous_gate_open = False
    gate_open_streak = 0
    gate_open_events = 0
    gate_open_ticks = 0
    longest_gate_open = 0
    blocked_gate_proposals = 0
    gate_proposals_while_open = 0
    gate_proposals_while_closed = 0
    blocked_open_gate_proposals = 0
    open_gate_with_waiting_body_ticks = 0

    for index, position in enumerate(world.positions):
        members[index]["minimum_plate_distance"] = _minimum_distance(
            None, position, world.plate
        )
        members[index]["minimum_gate_distance"] = _minimum_distance(
            None, position, world.gate_inside
        )

    for tick, row in enumerate(actions):
        if not np.array_equal(np.stack(observations), arrays["observations"][tick]):
            raise EscapeTeamError(f"observation replay mismatch at tick {tick + 1}")
        decoded = [None if int(value) < 0 else int(value) for value in row]
        semantic = [
            None if slot is None else world._action_map[slot]
            for slot in decoded
        ]
        positions_before = list(world.positions)
        for member, (slot, meaning) in enumerate(zip(decoded, semantic, strict=True)):
            if slot is None:
                wait_streaks[member] = 0
                continue
            members[member]["anonymous_action_histogram"][slot] += 1
            members[member]["semantic_action_histogram"][meaning] += 1
            if meaning == WAIT:
                members[member]["wait_actions"] += 1
                wait_streaks[member] += 1
                members[member]["longest_wait_streak"] = max(
                    members[member]["longest_wait_streak"], wait_streaks[member]
                )
            else:
                wait_streaks[member] = 0

        step = world.step(decoded)
        actual_hash = world.audit_state()["state_sha256"].removeprefix("sha256:")
        expected_hash = arrays["state_hashes"][tick].decode("ascii")
        checks = (
            (np.array_equal(np.stack(step.observations), arrays["next_observations"][tick]), "next observation"),
            (np.allclose(step.rewards, arrays["rewards"][tick], atol=1e-7), "reward"),
            (bool(step.terminated) == bool(arrays["terminated"][tick]), "termination"),
            (bool(step.truncated) == bool(arrays["truncated"][tick]), "truncation"),
            (actual_hash == expected_hash, "state hash"),
        )
        for valid, label in checks:
            if not valid:
                raise EscapeTeamError(f"{label} replay mismatch at tick {tick + 1}")

        resolution = step.resolution
        gate_crossers = set(resolution["gate_crossers"])
        for member in resolution["ordinary_cancelled"]:
            members[member]["cancelled_moves"] += 1
        for member in resolution["gate_proposals"]:
            members[member]["gate_proposals"] += 1
            blocked_gate_proposals += int(member not in gate_crossers)
            if resolution["gate_open"]:
                gate_proposals_while_open += 1
                blocked_open_gate_proposals += int(member not in gate_crossers)
            else:
                gate_proposals_while_closed += 1
        for member in gate_crossers:
            members[member]["gate_crossings"] += 1
        for member in resolution["exit_proposals"]:
            members[member]["exit_proposals"] += 1
        for member in resolution["escaped"]:
            members[member]["escapes"] += 1

        if world.gate_open:
            gate_open_ticks += 1
            gate_open_streak += 1
            longest_gate_open = max(longest_gate_open, gate_open_streak)
            if not previous_gate_open:
                gate_open_events += 1
            if any(position == world.gate_inside for position in world.positions):
                open_gate_with_waiting_body_ticks += 1
        else:
            gate_open_streak = 0
        previous_gate_open = world.gate_open

        for member, position in enumerate(world.positions):
            on_plate = position == world.plate
            if on_plate:
                members[member]["plate_ticks"] += 1
                plate_streaks[member] += 1
                members[member]["longest_plate_dwell"] = max(
                    members[member]["longest_plate_dwell"], plate_streaks[member]
                )
                if positions_before[member] != world.plate:
                    members[member]["plate_entries"] += 1
            else:
                plate_streaks[member] = 0
            at_gate = position == world.gate_inside
            if at_gate:
                members[member]["gate_inside_ticks"] += 1
                if not gate_inside_before[member]:
                    members[member]["gate_inside_entries"] += 1
            gate_inside_before[member] = at_gate
            members[member]["minimum_plate_distance"] = _minimum_distance(
                members[member]["minimum_plate_distance"], position, world.plate
            )
            members[member]["minimum_gate_distance"] = _minimum_distance(
                members[member]["minimum_gate_distance"], position, world.gate_inside
            )
        observations = step.observations

    external_returns = np.asarray(arrays["rewards"], dtype=np.float64).sum(axis=0)
    updates = capsule.get("updates", [None] * TEAM_SIZE)
    shaped_returns = [
        None if not update else float(update.get("mean_episode_return", float("nan")))
        for update in updates
    ]
    intrinsic_returns = [
        None if shaped is None else shaped - float(external_returns[index])
        for index, shaped in enumerate(shaped_returns)
    ]
    return {
        "episode_id": capsule["episode_id"],
        "environment_seed": int(capsule["seed"]),
        "environment_adapter": adapter,
        "training": bool(capsule["training"]),
        "steps": horizon,
        "escaped_count": int(capsule["escaped_count"]),
        "replay_verified": True,
        "unique_joint_states": len(set(arrays["state_hashes"].tolist())),
        "gate_open_events": gate_open_events,
        "gate_open_ticks": gate_open_ticks,
        "longest_gate_open_streak": longest_gate_open,
        "open_gate_with_waiting_body_ticks": open_gate_with_waiting_body_ticks,
        "blocked_gate_proposals": blocked_gate_proposals,
        "gate_proposals_while_open": gate_proposals_while_open,
        "gate_proposals_while_closed": gate_proposals_while_closed,
        "blocked_open_gate_proposals": blocked_open_gate_proposals,
        "external_returns": external_returns.tolist(),
        "shaped_training_returns": shaped_returns,
        "estimated_intrinsic_returns": intrinsic_returns,
        "update_entropy": [
            None if not update else float(update.get("entropy", float("nan")))
            for update in updates
        ],
        "members": members,
    }


def _parameter_changes(root: Path, latest: dict[str, Any]) -> list[dict[str, Any]]:
    birth_path = root / "snapshots" / "birth" / "ESCAPE_TEAM.json"
    birth = json.loads(birth_path.read_text(encoding="utf-8"))
    birth_by_id = {row["identity"]["member_id"]: row for row in birth["members"]}
    results = []
    for row in latest["members"]:
        identity = row["identity"]
        member_id = identity["member_id"]
        earlier = birth_by_id[member_id]
        old = torch.load(root / earlier["brain"]["path"], map_location="cpu", weights_only=True)
        new = torch.load(root / row["brain"]["path"], map_location="cpu", weights_only=True)
        squared_change = 0.0
        squared_initial = 0.0
        changed_tensors = 0
        for name, tensor in old["policy"].items():
            before = tensor.detach().to(torch.float64)
            after = new["policy"][name].detach().to(torch.float64)
            delta = after - before
            squared_change += float(torch.sum(delta * delta))
            squared_initial += float(torch.sum(before * before))
            changed_tensors += int(not torch.equal(before, after))
        l2 = math.sqrt(squared_change)
        results.append({
            "member_id": member_id,
            "name": identity["name"],
            "policy_l2_change_from_birth": l2,
            "policy_relative_l2_change_from_birth": (
                l2 / math.sqrt(squared_initial) if squared_initial else None
            ),
            "changed_policy_tensors": changed_tensors,
            "total_policy_tensors": len(old["policy"]),
            "brain_file_hash_changed": earlier["brain"]["sha256"] != row["brain"]["sha256"],
        })
    return results


def diagnose_team(team_root: Path) -> dict[str, Any]:
    """Replay and measure every preserved episode without modifying the team."""
    root = Path(team_root).resolve()
    latest = EscapeTeam._resolve_checkpoint(root)
    first_brain = torch.load(
        root / latest["members"][0]["brain"]["path"],
        map_location="cpu",
        weights_only=True,
    )
    learner_config = first_brain["config"]
    reward_table = RewardTable(**latest["reward_table"])
    records = _episode_records(root)
    episodes = [
        _replay_metrics(capsule, _load_archive(root, capsule), reward_table)
        for capsule in records
    ]
    max_entropy = math.log(ACTION_COUNT)
    entropies = [
        value for episode in episodes for value in episode["update_entropy"]
        if value is not None and math.isfinite(value)
    ]
    external = [
        value for episode in episodes for value in episode["external_returns"]
    ]
    intrinsic = [
        value for episode in episodes for value in episode["estimated_intrinsic_returns"]
        if value is not None and math.isfinite(value)
    ]
    aggregate = {
        "episodes": len(episodes),
        "independent_replications": 1 if episodes else 0,
        "total_joint_ticks": sum(item["steps"] for item in episodes),
        "total_escapes": sum(item["escaped_count"] for item in episodes),
        "total_plate_entries": sum(
            member["plate_entries"] for item in episodes for member in item["members"]
        ),
        "total_plate_ticks": sum(
            member["plate_ticks"] for item in episodes for member in item["members"]
        ),
        "longest_plate_dwell": max(
            (member["longest_plate_dwell"] for item in episodes for member in item["members"]),
            default=0,
        ),
        "total_gate_open_events": sum(item["gate_open_events"] for item in episodes),
        "total_gate_open_ticks": sum(item["gate_open_ticks"] for item in episodes),
        "total_gate_proposals": sum(
            member["gate_proposals"] for item in episodes for member in item["members"]
        ),
        "total_gate_crossings": sum(
            member["gate_crossings"] for item in episodes for member in item["members"]
        ),
        "total_blocked_gate_proposals": sum(
            item["blocked_gate_proposals"] for item in episodes
        ),
        "total_gate_proposals_while_open": sum(
            item["gate_proposals_while_open"] for item in episodes
        ),
        "total_gate_proposals_while_closed": sum(
            item["gate_proposals_while_closed"] for item in episodes
        ),
        "total_blocked_open_gate_proposals": sum(
            item["blocked_open_gate_proposals"] for item in episodes
        ),
        "total_open_gate_with_waiting_body_ticks": sum(
            item["open_gate_with_waiting_body_ticks"] for item in episodes
        ),
        "mean_external_return": float(np.mean(external)) if external else None,
        "mean_estimated_intrinsic_return": float(np.mean(intrinsic)) if intrinsic else None,
        "mean_action_entropy": float(np.mean(entropies)) if entropies else None,
        "maximum_action_entropy": max_entropy,
        "mean_entropy_fraction_of_maximum": (
            float(np.mean(entropies)) / max_entropy if entropies else None
        ),
    }
    return {
        "format": DIAGNOSTIC_FORMAT,
        "measurement_only": True,
        "semantic_metrics_exposed_to_learners": False,
        "team_root": str(root),
        "checkpoint": latest["checkpoint"],
        "sharing_mode": latest["sharing_mode"],
        "exploration_configuration": {
            key: learner_config[key] for key in (
                "training_exploration_mix",
                "episodic_action_exploration_mix",
                "episodic_novelty_coefficient",
                "entropy_coefficient",
            )
        },
        "parameter_changes": _parameter_changes(root, latest),
        "aggregate": aggregate,
        "episodes": episodes,
    }


def _probe_accuracy(features: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    split = max(2, int(len(features) * 0.75))
    split = min(split, len(features) - 1)
    train_pairs = features[:split]
    test_pairs = features[split:]
    x_train = np.stack([value for pair in train_pairs for value in pair])
    y_train = np.tile([0, 1], len(train_pairs))
    x_test = np.stack([value for pair in test_pairs for value in pair])
    y_test = np.tile([0, 1], len(test_pairs))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=0, max_iter=2_000),
    )
    model.fit(x_train, y_train)
    distances = [float(np.linalg.norm(second - first)) for first, second in features]
    return {
        "held_out_accuracy": float(model.score(x_test, y_test)),
        "mean_paired_embedding_l2": float(np.mean(distances)),
        "minimum_paired_embedding_l2": float(np.min(distances)),
    }


def probe_visual_mechanism(
    team_root: Path,
    *,
    configurations: int = 48,
    seed: int = 7_701,
) -> dict[str, Any]:
    """Probe frozen encoders; diagnostic labels never enter policy training."""
    if configurations < 8:
        raise ValueError("at least eight held-out configurations are required")
    team = EscapeTeam.load(Path(team_root), device="cpu")
    root = Path(team_root).resolve()
    birth_manifest = json.loads(
        (root / "snapshots" / "birth" / "ESCAPE_TEAM.json").read_text(encoding="utf-8")
    )
    birth_by_id = {
        row["identity"]["member_id"]: RecurrentCausalLearner.load(
            root / row["brain"]["path"], device="cpu"
        )
        for row in birth_manifest["members"]
    }
    rng = np.random.default_rng(seed)
    gate_pairs: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [] for _ in range(TEAM_SIZE)
    ]
    mechanism_pairs: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [] for _ in range(TEAM_SIZE)
    ]
    birth_gate_pairs: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [] for _ in range(TEAM_SIZE)
    ]
    birth_mechanism_pairs: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [] for _ in range(TEAM_SIZE)
    ]
    gate_raw_changes = []
    for config_index in range(configurations):
        world = EscapeChamberRoomA(seed=seed + config_index, reward_table=team.reward_table)
        world.reset()
        reserved = {world.plate, (2, 3)}
        pool = sorted(world._floor.difference(reserved))
        choices = rng.choice(len(pool), size=TEAM_SIZE - 1, replace=False)
        others = [pool[int(value)] for value in choices]

        world.positions = [world.plate, *others]
        world.gate_open = False
        gate_closed = world.observations()
        world.gate_open = True
        gate_open = world.observations()
        gate_raw_changes.append(float(np.mean(np.abs(
            gate_open[0].astype(np.int16) - gate_closed[0].astype(np.int16)
        ))))

        world.positions = [(2, 3), *others]
        world.gate_open = False
        inactive = world.observations()
        world.positions = [world.plate, *others]
        world.gate_open = True
        active = world.observations()

        for member, escape_member in enumerate(team.members):
            policy = escape_member.learner.policy
            birth_policy = birth_by_id[escape_member.identity.member_id].policy
            policy.eval()
            birth_policy.eval()
            with torch.no_grad():
                gate_a = policy.encode_pixels(
                    torch.from_numpy(gate_closed[member].copy())
                ).cpu().numpy()[0]
                gate_b = policy.encode_pixels(
                    torch.from_numpy(gate_open[member].copy())
                ).cpu().numpy()[0]
                mechanism_a = policy.encode_pixels(
                    torch.from_numpy(inactive[member].copy())
                ).cpu().numpy()[0]
                mechanism_b = policy.encode_pixels(
                    torch.from_numpy(active[member].copy())
                ).cpu().numpy()[0]
                birth_gate_a = birth_policy.encode_pixels(
                    torch.from_numpy(gate_closed[member].copy())
                ).cpu().numpy()[0]
                birth_gate_b = birth_policy.encode_pixels(
                    torch.from_numpy(gate_open[member].copy())
                ).cpu().numpy()[0]
                birth_mechanism_a = birth_policy.encode_pixels(
                    torch.from_numpy(inactive[member].copy())
                ).cpu().numpy()[0]
                birth_mechanism_b = birth_policy.encode_pixels(
                    torch.from_numpy(active[member].copy())
                ).cpu().numpy()[0]
            gate_pairs[member].append((gate_a, gate_b))
            mechanism_pairs[member].append((mechanism_a, mechanism_b))
            birth_gate_pairs[member].append((birth_gate_a, birth_gate_b))
            birth_mechanism_pairs[member].append((birth_mechanism_a, birth_mechanism_b))

    results = []
    for index, member in enumerate(team.members):
        current_gate = _probe_accuracy(gate_pairs[index])
        current_mechanism = _probe_accuracy(mechanism_pairs[index])
        birth_gate = _probe_accuracy(birth_gate_pairs[index])
        birth_mechanism = _probe_accuracy(birth_mechanism_pairs[index])
        results.append({
            "member_id": member.identity.member_id,
            "name": member.identity.name,
            "current_gate_visual_only": current_gate,
            "birth_gate_visual_only": birth_gate,
            "gate_accuracy_change_from_birth": (
                current_gate["held_out_accuracy"] - birth_gate["held_out_accuracy"]
            ),
            "current_plate_occupancy_and_gate_condition": current_mechanism,
            "birth_plate_occupancy_and_gate_condition": birth_mechanism,
            "mechanism_accuracy_change_from_birth": (
                current_mechanism["held_out_accuracy"]
                - birth_mechanism["held_out_accuracy"]
            ),
        })
    return {
        "format": PERCEPTION_FORMAT,
        "measurement_only": True,
        "labels_used_for_policy_training": False,
        "frozen_encoder": True,
        "matched_birth_encoder_control": True,
        "configurations": configurations,
        "split": "first 75% train / final 25% held out by configuration",
        "gate_probe_note": (
            "Counterfactual rendering changes only the gate while holding body positions fixed."
        ),
        "mechanism_probe_note": (
            "Valid condition pairs move one body from beside the plate onto it and open the gate."
        ),
        "mean_gate_raw_pixel_absolute_change": float(np.mean(gate_raw_changes)),
        "members": results,
    }
