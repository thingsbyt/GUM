"""Causal bookkeeping and controlled-update audit for rewarded escape experience."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gum.lineage import HashLedger, file_sha256

from .escape_chamber import (
    ACTION_COUNT,
    ROOM_A_ADAPTER,
    TEAM_SIZE,
    RewardTable,
    make_escape_chamber,
)
from .escape_team import LEDGER_FILENAME, EscapeTeamError
from .recurrent_meta import RecurrentCausalLearner


CREDIT_AUDIT_FORMAT = "gum-cooperative-escape-credit-audit-v1"


def _records(root: Path) -> list[dict[str, Any]]:
    state = HashLedger(root / LEDGER_FILENAME).verify()
    if not state["valid"]:
        raise EscapeTeamError("cannot audit an invalid episode ledger")
    rows = []
    with (root / LEDGER_FILENAME).open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if value.get("event") == "episode-completed":
                rows.append(value["payload"])
    return rows


def _snapshot(root: Path, label: str) -> dict[str, Any]:
    path = root / "snapshots" / label / "ESCAPE_TEAM.json"
    if not path.is_file():
        raise EscapeTeamError(f"checkpoint is missing: {label}")
    return json.loads(path.read_text(encoding="utf-8"))


def _archive(root: Path, capsule: dict[str, Any]) -> dict[str, np.ndarray]:
    reference = capsule["archive"]
    path = (root / reference["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise EscapeTeamError("transition archive path escapes team directory") from error
    if f"sha256:{file_sha256(path)}" != reference["sha256"]:
        raise EscapeTeamError("transition archive hash differs")
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def _member_trajectories(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Reproduce active rewards plus the learner's delayed-reward attachment."""
    trajectories = []
    for member in range(TEAM_SIZE):
        observations = []
        actions = []
        rewards = []
        ticks = []
        delayed = []
        for tick in range(len(arrays["actions"])):
            reward = float(arrays["rewards"][tick, member])
            if bool(arrays["active_before"][tick, member]):
                action = int(arrays["actions"][tick, member])
                if action < 0:
                    raise EscapeTeamError("active member is missing an executed action")
                observations.append(arrays["observations"][tick, member].copy())
                actions.append(action)
                rewards.append(reward)
                ticks.append(tick + 1)
            elif reward != 0.0:
                if not rewards:
                    raise EscapeTeamError("delayed reward has no prior executed action")
                rewards[-1] += reward
                delayed.append({
                    "source_tick": tick + 1,
                    "credited_to_action_tick": ticks[-1],
                    "reward": reward,
                })
        trajectories.append({
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "ticks": ticks,
            "delayed_rewards": delayed,
            "action_count": ACTION_COUNT,
        })
    return trajectories


def _discounted_returns(rewards: list[float], discount: float) -> np.ndarray:
    result = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + discount * running
        result[index] = running
    return result


def _policy_trace(
    learner: RecurrentCausalLearner,
    trajectory: dict[str, Any],
) -> dict[str, np.ndarray]:
    action_count = int(trajectory["action_count"])
    hidden = torch.zeros(1, learner.config.hidden_size, device=learner.device)
    action_memory = torch.zeros(
        1,
        learner.config.action_count,
        learner.config.action_memory_size,
        device=learner.device,
    )
    previous_action = learner.config.action_count
    previous_reward = 0.0
    values = []
    raw_chosen = []
    training_chosen = []
    entropies = []
    learner.policy.eval()
    with torch.no_grad():
        for step, (observation, action, reward) in enumerate(zip(
            trajectory["observations"],
            trajectory["actions"],
            trajectory["rewards"],
            strict=True,
        )):
            logits, value, hidden, action_memory = learner.policy(
                torch.from_numpy(np.asarray(observation).copy()).to(learner.device),
                torch.tensor([previous_action], device=learner.device),
                torch.tensor(
                    [previous_reward], dtype=torch.float32, device=learner.device
                ),
                torch.tensor([float(step == 0)], dtype=torch.float32, device=learner.device),
                hidden,
                action_memory,
            )
            raw = torch.softmax(logits[:, :action_count], dim=-1)
            mixed = raw
            if learner.config.training_exploration_mix > 0.0:
                mix = learner.config.training_exploration_mix
                mixed = (1.0 - mix) * raw + mix / action_count
            raw_chosen.append(float(raw[0, action].item()))
            training_chosen.append(float(mixed[0, action].item()))
            entropies.append(float((-(mixed * torch.log(mixed)).sum()).item()))
            values.append(float(value.item()))
            previous_action = action
            previous_reward = reward
    returns = _discounted_returns(trajectory["rewards"], learner.config.discount)
    advantages = returns - np.asarray(values)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    return {
        "values": np.asarray(values),
        "discounted_returns": returns,
        "normalized_advantages": advantages,
        "raw_chosen_probabilities": np.asarray(raw_chosen),
        "training_chosen_probabilities": np.asarray(training_chosen),
        "training_entropies": np.asarray(entropies),
    }


def _replay_semantics(
    capsule: dict[str, Any],
    arrays: dict[str, np.ndarray],
    reward_table,
) -> dict[str, Any]:
    adapter = capsule["archive"].get("environment_adapter", ROOM_A_ADAPTER)
    world = make_escape_chamber(
        adapter,
        seed=int(capsule["seed"]),
        horizon=len(arrays["actions"]),
        reward_table=reward_table,
    )
    observations = world.reset()
    events = []
    replayed_rewards = []
    plate_entry_tick: list[int | None] = [None] * TEAM_SIZE
    plate_intervals: list[list[dict[str, int]]] = [[] for _ in range(TEAM_SIZE)]
    for tick, row in enumerate(arrays["actions"], 1):
        if not np.array_equal(np.stack(observations), arrays["observations"][tick - 1]):
            raise EscapeTeamError(f"observation replay mismatch at tick {tick}")
        actions = [None if int(value) < 0 else int(value) for value in row]
        semantic = [
            None if action is None else world._action_map[action]
            for action in actions
        ]
        before = list(world.positions)
        step = world.step(actions)
        expected_hash = arrays["state_hashes"][tick - 1].decode("ascii")
        actual_hash = world.audit_state()["state_sha256"].removeprefix("sha256:")
        checks = (
            np.array_equal(
                np.stack(step.observations), arrays["next_observations"][tick - 1]
            ),
            np.allclose(step.rewards, arrays["rewards"][tick - 1], atol=1e-7),
            bool(step.terminated) == bool(arrays["terminated"][tick - 1]),
            bool(step.truncated) == bool(arrays["truncated"][tick - 1]),
            actual_hash == expected_hash,
        )
        if not all(checks):
            raise EscapeTeamError(f"transition replay mismatch at tick {tick}")
        replayed_rewards.append(list(step.rewards))
        holders = [
            member for member, position in enumerate(world.positions)
            if position == world.plate and not world.escaped[member]
        ]
        for member in range(TEAM_SIZE):
            on_plate = world.positions[member] == world.plate
            if on_plate and before[member] != world.plate:
                plate_entry_tick[member] = tick
            if not on_plate and before[member] == world.plate:
                start = plate_entry_tick[member]
                if start is not None:
                    plate_intervals[member].append({"start_tick": start, "end_tick": tick - 1})
                plate_entry_tick[member] = None
        if step.resolution["gate_crossers"] or step.resolution["escaped"]:
            events.append({
                "tick": tick,
                "holders": holders,
                "gate_crossers": list(step.resolution["gate_crossers"]),
                "escaped": list(step.resolution["escaped"]),
                "executed_actions": actions,
                "semantic_actions_diagnostic_only": semantic,
                "rewards": list(step.rewards),
                "positions_before": [None if value is None else list(value) for value in before],
                "positions_after": [
                    None if value is None else list(value) for value in world.positions
                ],
            })
        observations = step.observations
    for member, start in enumerate(plate_entry_tick):
        if start is not None:
            plate_intervals[member].append({
                "start_tick": start,
                "end_tick": len(arrays["actions"]),
            })
    return {
        "replay_verified": True,
        "environment_adapter": adapter,
        "events": events,
        "plate_intervals": plate_intervals,
        "replayed_rewards": replayed_rewards,
        "action_map_diagnostic_only": list(world._action_map),
    }


def _manifest_learners(
    root: Path,
    manifest: dict[str, Any],
    *,
    device: str,
) -> list[RecurrentCausalLearner]:
    return [
        RecurrentCausalLearner.load(root / row["brain"]["path"], device=device)
        for row in manifest["members"]
    ]


def _cooperation_roles(
    semantics: dict[str, Any],
    identities: list[dict[str, Any]],
) -> dict[str, Any]:
    """Name observed roles after replay; never feed these labels to a learner."""
    escape_events = [event for event in semantics["events"] if event["escaped"]]
    if not escape_events:
        raise EscapeTeamError("rewarded episode replay contains no escape event")
    escape_event = escape_events[0]
    crosser = int(escape_event["escaped"][0])
    gate_events = [
        event for event in semantics["events"]
        if crosser in event["gate_crossers"] and event["tick"] <= escape_event["tick"]
    ]
    if not gate_events:
        raise EscapeTeamError("escaped member has no replayed gate crossing")
    gate_event = gate_events[-1]
    holders = [int(index) for index in gate_event["holders"] if int(index) != crosser]
    if not holders:
        raise EscapeTeamError("gate crossing has no observed plate holder")

    def named(index: int) -> dict[str, Any]:
        identity = identities[index]
        return {
            "member_index": index,
            "member_id": identity["member_id"],
            "name": identity["name"],
        }

    return {
        "diagnostic_only": True,
        "crosser": named(crosser),
        "holders_at_gate_crossing": [named(index) for index in holders],
        "gate_crossing_tick": gate_event["tick"],
        "escape_tick": escape_event["tick"],
        "rewards_at_gate_crossing": gate_event["rewards"],
        "rewards_at_escape": escape_event["rewards"],
    }


def _weight_difference(
    left: RecurrentCausalLearner,
    right: RecurrentCausalLearner,
) -> dict[str, Any]:
    maximum = 0.0
    squared = 0.0
    exact = True
    for name, tensor in left.policy.state_dict().items():
        delta = tensor.detach().to("cpu", torch.float64) - right.policy.state_dict()[
            name
        ].detach().to("cpu", torch.float64)
        exact = exact and bool(torch.equal(delta, torch.zeros_like(delta)))
        maximum = max(maximum, float(torch.max(torch.abs(delta))))
        squared += float(torch.sum(delta * delta))
    return {
        "exact_tensor_equality": exact,
        "maximum_absolute_difference": maximum,
        "l2_difference": math.sqrt(squared),
    }


def _probability_change(
    before: np.ndarray,
    after: np.ndarray,
) -> dict[str, Any]:
    delta = np.asarray(after, dtype=np.float64) - np.asarray(before, dtype=np.float64)
    return {
        "contexts": len(delta),
        "mean_signed_change": float(delta.mean()),
        "mean_absolute_change": float(np.abs(delta).mean()),
        "maximum_absolute_change": float(np.abs(delta).max()),
        "increased_contexts": int(np.sum(delta > 0.0)),
        "decreased_contexts": int(np.sum(delta < 0.0)),
        "unchanged_contexts": int(np.sum(delta == 0.0)),
    }


def _evaluate_group(
    learners: list[RecurrentCausalLearner],
    *,
    seeds: list[int],
    horizon: int,
    adapter: str,
    reward_table,
) -> dict[str, Any]:
    episodes = []
    for seed in seeds:
        world = make_escape_chamber(
            adapter, seed=seed, horizon=horizon, reward_table=reward_table
        )
        observations = world.reset()
        spec = world.public_spec()
        for learner, observation in zip(learners, observations, strict=True):
            learner.begin(spec, observation, training=False)
        gate_crossings = 0
        proposals_while_open = 0
        plate_ticks = 0
        while True:
            actions = [
                None if world.escaped[index]
                else learners[index].act(observations[index], training=False)
                for index in range(TEAM_SIZE)
            ]
            step = world.step(actions)
            gate_crossings += len(step.resolution["gate_crossers"])
            proposals_while_open += int(step.resolution["gate_open"]) * len(
                step.resolution["gate_proposals"]
            )
            plate_ticks += sum(position == world.plate for position in world.positions)
            for index, learner in enumerate(learners):
                if step.active_before[index]:
                    learner.observe(
                        actions[index], step.transition_for(index), training=False
                    )
            observations = step.observations
            if step.terminated or step.truncated:
                break
        for learner in learners:
            learner.finish_episode(training=False)
        episodes.append({
            "seed": seed,
            "escapes": len(world.escape_order),
            "gate_crossings": gate_crossings,
            "gate_proposals_while_open": proposals_while_open,
            "plate_ticks": plate_ticks,
        })
    return {
        "episodes": len(episodes),
        "total_escapes": sum(row["escapes"] for row in episodes),
        "episodes_with_escape": sum(row["escapes"] > 0 for row in episodes),
        "mean_escapes": float(np.mean([row["escapes"] for row in episodes])),
        "total_gate_crossings": sum(row["gate_crossings"] for row in episodes),
        "total_gate_proposals_while_open": sum(
            row["gate_proposals_while_open"] for row in episodes
        ),
        "total_plate_ticks": sum(row["plate_ticks"] for row in episodes),
        "per_seed": episodes,
    }


def audit_successful_episode(
    team_root: Path,
    episode_id: str,
    output_root: Path,
    *,
    device: str = "cpu",
    evaluation_seed_start: int = 9_510_000,
    evaluation_episodes: int = 64,
    evaluation_horizon: int = 80,
) -> dict[str, Any]:
    """Run the bounded three-branch credit test from a real checkpoint."""
    root = Path(team_root).resolve()
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"credit-audit output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records = _records(root)
    matches = [index for index, row in enumerate(records) if row["episode_id"] == episode_id]
    if len(matches) != 1:
        raise EscapeTeamError("successful episode must appear exactly once")
    position = matches[0]
    capsule = records[position]
    if not capsule["training"] or max(float(value) for value in capsule["returns"]) <= 0:
        raise EscapeTeamError("credit audit requires a rewarded training episode")
    before_label = "birth" if position == 0 else records[position - 1]["episode_id"]
    before = _snapshot(root, before_label)
    after = _snapshot(root, episode_id)
    arrays = _archive(root, capsule)
    reward_table = RewardTable(**before["reward_table"])
    semantics = _replay_semantics(capsule, arrays, reward_table)
    exact_arrays = dict(arrays)
    exact_arrays["rewards"] = np.asarray(
        semantics["replayed_rewards"], dtype=np.float64
    )
    trajectories = _member_trajectories(exact_arrays)
    identities = [row["identity"] for row in before["members"]]
    roles = _cooperation_roles(semantics, identities)

    existing_seeds = {int(row["seed"]) for row in records}
    evaluation_seeds = list(range(
        evaluation_seed_start,
        evaluation_seed_start + evaluation_episodes,
    ))
    if existing_seeds.intersection(evaluation_seeds):
        raise EscapeTeamError("evaluation seeds overlap recorded team experience")

    branch_learners: dict[str, list[RecurrentCausalLearner]] = {
        "control": _manifest_learners(root, before, device=device),
        "ordinary_actor_critic": _manifest_learners(root, before, device=device),
        "reward_outcome_replay": _manifest_learners(root, before, device=device),
    }
    actual_after = _manifest_learners(root, after, device=device)
    branch_updates = {"control": [None] * TEAM_SIZE}
    branch_updates["ordinary_actor_critic"] = [
        learner.fit_recorded_actor_critic(trajectory)
        for learner, trajectory in zip(
            branch_learners["ordinary_actor_critic"], trajectories, strict=True
        )
    ]
    branch_updates["reward_outcome_replay"] = [
        learner.fit_reward_outcome_replay(
            [trajectory], epochs=1, batch_episodes=1
        )
        for learner, trajectory in zip(
            branch_learners["reward_outcome_replay"], trajectories, strict=True
        )
    ]

    actual_reconstruction = []
    member_credit = []
    event_ticks = {
        event["tick"] for event in semantics["events"]
    }
    for index, (identity, trajectory) in enumerate(zip(
        identities, trajectories, strict=True
    )):
        before_trace = _policy_trace(branch_learners["control"][index], trajectory)
        ordinary_trace = _policy_trace(
            branch_learners["ordinary_actor_critic"][index], trajectory
        )
        outcome_trace = _policy_trace(
            branch_learners["reward_outcome_replay"][index], trajectory
        )
        local_by_tick = {tick: local for local, tick in enumerate(trajectory["ticks"])}
        tick_rows = []
        for tick in trajectory["ticks"]:
            local = local_by_tick[tick]
            on_plate_interval = any(
                interval["start_tick"] <= tick <= interval["end_tick"]
                for interval in semantics["plate_intervals"][index]
            )
            if tick in event_ticks or on_plate_interval or trajectory["rewards"][local] > 0:
                tick_rows.append({
                    "tick": tick,
                    "executed_action": trajectory["actions"][local],
                    "scalar_reward": trajectory["rewards"][local],
                    "discounted_return": float(before_trace["discounted_returns"][local]),
                    "normalized_advantage": float(
                        before_trace["normalized_advantages"][local]
                    ),
                    "ordinary_update_direction": (
                        "reinforce"
                        if before_trace["normalized_advantages"][local] > 1e-12
                        else "suppress"
                        if before_trace["normalized_advantages"][local] < -1e-12
                        else "neutral"
                    ),
                    "on_plate_diagnostic_only": on_plate_interval,
                    "chosen_probability": {
                        "control": float(
                            before_trace["raw_chosen_probabilities"][local]
                        ),
                        "ordinary_actor_critic": float(
                            ordinary_trace["raw_chosen_probabilities"][local]
                        ),
                        "reward_outcome_replay": float(
                            outcome_trace["raw_chosen_probabilities"][local]
                        ),
                    },
                })
        actual_reconstruction.append({
            "member_id": identity["member_id"],
            "name": identity["name"],
            "weight_match": _weight_difference(
                branch_learners["ordinary_actor_critic"][index], actual_after[index]
            ),
            "recorded_update": capsule["updates"][index],
            "reconstructed_update": {
                key: value for key, value in branch_updates[
                    "ordinary_actor_critic"
                ][index].items() if key != "chosen_probabilities_before"
            },
        })
        member_credit.append({
            "member_id": identity["member_id"],
            "name": identity["name"],
            "active_action_ticks": len(trajectory["actions"]),
            "delayed_reward_events": trajectory["delayed_rewards"],
            "positive_discounted_return_ticks": int(np.sum(
                before_trace["discounted_returns"] > 0
            )),
            "positive_normalized_advantage_ticks": int(np.sum(
                before_trace["normalized_advantages"] > 0
            )),
            "recorded_context_probability_change": {
                "ordinary_actor_critic": _probability_change(
                    before_trace["raw_chosen_probabilities"],
                    ordinary_trace["raw_chosen_probabilities"],
                ),
                "reward_outcome_replay": _probability_change(
                    before_trace["raw_chosen_probabilities"],
                    outcome_trace["raw_chosen_probabilities"],
                ),
            },
            "audited_causal_contexts": tick_rows,
        })

    for branch, learners in branch_learners.items():
        for identity, learner in zip(identities, learners, strict=True):
            learner.save(output / "branches" / branch / f"{identity['member_id']}.pt")

    evaluations = {
        branch: _evaluate_group(
            learners,
            seeds=evaluation_seeds,
            horizon=evaluation_horizon,
            adapter=semantics["environment_adapter"],
            reward_table=reward_table,
        )
        for branch, learners in branch_learners.items()
    }
    control_escapes = {
        row["seed"]: row["escapes"] for row in evaluations["control"]["per_seed"]
    }
    paired_evaluation = {}
    for branch in ("ordinary_actor_critic", "reward_outcome_replay"):
        comparisons = [
            int(row["escapes"]) - int(control_escapes[row["seed"]])
            for row in evaluations[branch]["per_seed"]
        ]
        paired_evaluation[branch] = {
            "seeds_better_than_control": sum(value > 0 for value in comparisons),
            "seeds_worse_than_control": sum(value < 0 for value in comparisons),
            "seeds_tied_with_control": sum(value == 0 for value in comparisons),
            "net_escape_difference": sum(comparisons),
        }
    reconstruction_exact = all(
        row["weight_match"]["exact_tensor_equality"]
        for row in actual_reconstruction
    )
    delayed_events = sum(
        len(trajectory["delayed_rewards"]) for trajectory in trajectories
    )
    result = {
        "format": CREDIT_AUDIT_FORMAT,
        "source_team": str(root),
        "episode_id": episode_id,
        "pre_episode_checkpoint": before_label,
        "post_episode_checkpoint": episode_id,
        "source_archive": capsule["archive"],
        "training_reward_source": (
            "deterministically replayed float64 environment rewards; archived "
            "float32 rewards were verified equal within 1e-7"
        ),
        "training_calculation_reconstructed": True,
        "semantic_labels_used_for_training": False,
        "semantic_labels_used_for_diagnostics": True,
        "room_a_modified_or_trained": False,
        "alignment_checks": {
            "observations_actions_rewards_and_terminal_flags_replay_exactly": True,
            "executed_actions_used_as_policy_targets": True,
            "per_member_rewards_attached_to_same_tick_then_discounted": True,
            "post_escape_delayed_rewards_present_in_episode": delayed_events > 0,
            "post_escape_delayed_reward_alignment": (
                "last-executed-action" if delayed_events else "not-exercised"
            ),
            "ordinary_update_matches_saved_post_episode_weights_exactly": (
                reconstruction_exact
            ),
        },
        "branch_definitions": {
            "control": "exact pre-episode checkpoint; no update",
            "ordinary_actor_critic": "one reconstructed ordinary update",
            "reward_outcome_replay": (
                "one update using only recorded scalar reward signs"
            ),
        },
        "episode_semantics_diagnostic_only": semantics,
        "observed_cooperation_roles_diagnostic_only": roles,
        "member_credit": member_credit,
        "ordinary_update_reconstruction": actual_reconstruction,
        "branch_updates": branch_updates,
        "evaluation": {
            "untouched_seed_start": evaluation_seed_start,
            "episodes": evaluation_episodes,
            "horizon": evaluation_horizon,
            "seed_overlap_with_training_history": False,
            "branches": evaluations,
            "paired_against_control": paired_evaluation,
        },
    }
    return result
