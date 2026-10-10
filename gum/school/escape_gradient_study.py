"""Preregistered multi-team test of one cooperative-learning correction."""
from __future__ import annotations

from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from gum.lineage import file_sha256
from gum.storage import atomic_write_bytes, atomic_write_json

from .escape_chamber import RewardTable, make_escape_chamber
from .escape_credit_audit import (
    _archive,
    _cooperation_roles,
    _manifest_learners,
    _member_trajectories,
    _policy_trace,
    _records,
    _replay_semantics,
    _snapshot,
    _weight_difference,
)
from .escape_team import EscapeTeamError
from .recurrent_meta import RecurrentCausalLearner


STUDY_FORMAT = "gum-cooperative-gradient-correction-study-v1"


def _prepare_selected_episode(
    root: Path,
    expected_episode_id: str,
) -> dict[str, Any]:
    records = _records(root)
    selected = next(
        (
            (index, row)
            for index, row in enumerate(records)
            if row["training"] and int(row["escaped_count"]) > 0
        ),
        None,
    )
    if selected is None:
        raise EscapeTeamError("preregistered team has no successful training episode")
    position, capsule = selected
    if capsule["episode_id"] != expected_episode_id:
        raise EscapeTeamError("outcome-only trajectory selection differs from plan")
    before_label = "birth" if position == 0 else records[position - 1]["episode_id"]
    before = _snapshot(root, before_label)
    after = _snapshot(root, capsule["episode_id"])
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
    return {
        "records": records,
        "position": position,
        "capsule": capsule,
        "before_label": before_label,
        "before": before,
        "after": after,
        "reward_table": reward_table,
        "trajectories": trajectories,
        "identities": identities,
        "roles": roles,
    }


def _configure_correction(learner: RecurrentCausalLearner) -> None:
    learner.config = replace(learner.config, value_shared_gradient_scale=0.0)
    learner.policy.config = learner.config


def _save_branch(
    output: Path,
    team_name: str,
    branch: str,
    identities: list[dict[str, Any]],
    learners: list[RecurrentCausalLearner],
    *,
    source_checkpoint: str,
    update: str,
) -> dict[str, Any]:
    rows = []
    for identity, learner in zip(identities, learners, strict=True):
        path = output / "checkpoints" / team_name / branch / f"{identity['member_id']}.pt"
        learner.save(path)
        rows.append({
            "identity": identity,
            "brain": {
                "path": path.relative_to(output).as_posix(),
                "sha256": f"sha256:{file_sha256(path)}",
            },
            "config": asdict(learner.config),
        })
    manifest = {
        "format": "gum-cooperative-gradient-branch-checkpoint-v1",
        "phase": "offline-replay-update",
        "team": team_name,
        "branch": branch,
        "source_checkpoint": source_checkpoint,
        "update": update,
        "members": rows,
    }
    target = output / "checkpoints" / team_name / branch / "BRANCH.json"
    atomic_write_json(target, manifest, backup=False, sort_keys=True)
    return {
        "path": target.relative_to(output).as_posix(),
        "sha256": f"sha256:{file_sha256(target)}",
    }


def _archive_evaluation_episode(
    output: Path,
    team_name: str,
    branch: str,
    seed: int,
    rows: dict[str, list[Any]],
) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        observations=np.asarray(rows["observations"], dtype=np.uint8),
        next_observations=np.asarray(rows["next_observations"], dtype=np.uint8),
        actions=np.asarray(rows["actions"], dtype=np.int64),
        rewards=np.asarray(rows["rewards"], dtype=np.float32),
        terminated=np.asarray(rows["terminated"], dtype=np.bool_),
        truncated=np.asarray(rows["truncated"], dtype=np.bool_),
        active_before=np.asarray(rows["active_before"], dtype=np.bool_),
        active_after=np.asarray(rows["active_after"], dtype=np.bool_),
        state_hashes=np.asarray(rows["state_hashes"], dtype="S64"),
    )
    path = output / "evaluation-transitions" / team_name / branch / f"{seed}.npz"
    atomic_write_bytes(path, buffer.getvalue(), backup=False)
    return {
        "format": "gum-cooperative-escape-transitions-v1",
        "phase": "frozen-evaluation",
        "path": path.relative_to(output).as_posix(),
        "sha256": f"sha256:{file_sha256(path)}",
        "joint_ticks": len(rows["actions"]),
    }


def _evaluate_branch(
    learners: list[RecurrentCausalLearner],
    *,
    seeds: list[int],
    horizon: int,
    adapter: str,
    reward_table: RewardTable,
    output: Path,
    team_name: str,
    branch: str,
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
        gate_open_ticks = 0
        plate_ticks = 0
        plate_runs = [0, 0, 0, 0]
        maximum_plate_run = [0, 0, 0, 0]
        member_action_interactions = 0
        rows: dict[str, list[Any]] = {
            name: [] for name in (
                "observations", "next_observations", "actions", "rewards",
                "terminated", "truncated", "active_before", "active_after",
                "state_hashes",
            )
        }
        while True:
            actions = [
                None if world.escaped[index]
                else learners[index].act(observations[index], training=False)
                for index in range(4)
            ]
            step = world.step(actions)
            rows["observations"].append(np.stack(observations))
            rows["next_observations"].append(np.stack(step.observations))
            rows["actions"].append([-1 if value is None else value for value in actions])
            rows["rewards"].append(step.rewards)
            rows["terminated"].append(step.terminated)
            rows["truncated"].append(step.truncated)
            rows["active_before"].append(step.active_before)
            rows["active_after"].append(step.active_after)
            rows["state_hashes"].append(
                world.audit_state()["state_sha256"].removeprefix("sha256:")
            )
            gate_crossings += len(step.resolution["gate_crossers"])
            gate_open_ticks += int(step.resolution["gate_open"])
            plate_ticks += sum(position == world.plate for position in world.positions)
            member_action_interactions += sum(step.active_before)
            for index, position in enumerate(world.positions):
                if position == world.plate and not world.escaped[index]:
                    plate_runs[index] += 1
                    maximum_plate_run[index] = max(
                        maximum_plate_run[index], plate_runs[index]
                    )
                else:
                    plate_runs[index] = 0
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
        archive = _archive_evaluation_episode(
            output, team_name, branch, seed, rows
        )
        episodes.append({
            "seed": seed,
            "escapes": len(world.escape_order),
            "legal_maximum_completion": len(world.escape_order) == 3,
            "gate_crossings": gate_crossings,
            "successful_crossings_during_openings": gate_crossings,
            "gate_open_ticks": gate_open_ticks,
            "plate_ticks": plate_ticks,
            "maximum_sustained_plate_ticks": max(maximum_plate_run),
            "joint_ticks": world.steps,
            "member_action_interactions": member_action_interactions,
            "transition_archive": archive,
        })
    return {
        "episodes": len(episodes),
        "total_escapes": sum(row["escapes"] for row in episodes),
        "episodes_with_escape": sum(row["escapes"] > 0 for row in episodes),
        "legal_maximum_completions": sum(
            row["legal_maximum_completion"] for row in episodes
        ),
        "total_gate_crossings": sum(row["gate_crossings"] for row in episodes),
        "total_gate_open_ticks": sum(row["gate_open_ticks"] for row in episodes),
        "total_plate_ticks": sum(row["plate_ticks"] for row in episodes),
        "mean_maximum_sustained_plate_ticks": float(np.mean([
            row["maximum_sustained_plate_ticks"] for row in episodes
        ])),
        "joint_ticks": sum(row["joint_ticks"] for row in episodes),
        "member_action_interactions": sum(
            row["member_action_interactions"] for row in episodes
        ),
        "per_seed": episodes,
    }


def _recorded_contexts(
    trajectories: list[dict[str, Any]],
    identities: list[dict[str, Any]],
    roles: dict[str, Any],
    branches: dict[str, list[RecurrentCausalLearner]],
) -> list[dict[str, Any]]:
    selected = []
    crosser = int(roles["crosser"]["member_index"])
    gate_tick = int(roles["gate_crossing_tick"])
    escape_tick = int(roles["escape_tick"])
    selected.extend([
        (crosser, gate_tick, "crosser-at-gate"),
        (crosser, escape_tick, "crosser-at-escape"),
    ])
    selected.extend(
        (int(holder["member_index"]), gate_tick, "holder-at-gate")
        for holder in roles["holders_at_gate_crossing"]
    )
    result = []
    for member_index, tick, label in selected:
        trajectory = trajectories[member_index]
        local_by_tick = {value: index for index, value in enumerate(trajectory["ticks"])}
        if tick not in local_by_tick:
            continue
        local = local_by_tick[tick]
        traces = {
            name: _policy_trace(learners[member_index], trajectory)
            for name, learners in branches.items()
        }
        result.append({
            "diagnostic_label": label,
            "member_id": identities[member_index]["member_id"],
            "name": identities[member_index]["name"],
            "tick": tick,
            "executed_action": trajectory["actions"][local],
            "probability": {
                name: float(trace["raw_chosen_probabilities"][local])
                for name, trace in traces.items()
            },
        })
    return result


def _branch_summary(teams: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    rows = [team["evaluation"][branch] for team in teams]
    return {
        "teams": len(rows),
        "episodes": sum(row["episodes"] for row in rows),
        "total_escapes": sum(row["total_escapes"] for row in rows),
        "episodes_with_escape": sum(row["episodes_with_escape"] for row in rows),
        "legal_maximum_completions": sum(
            row["legal_maximum_completions"] for row in rows
        ),
        "total_gate_crossings": sum(row["total_gate_crossings"] for row in rows),
        "total_gate_open_ticks": sum(row["total_gate_open_ticks"] for row in rows),
        "total_plate_ticks": sum(row["total_plate_ticks"] for row in rows),
        "joint_ticks": sum(row["joint_ticks"] for row in rows),
        "member_action_interactions": sum(
            row["member_action_interactions"] for row in rows
        ),
    }


def _bootstrap_team_difference(
    values: list[float], *, seed: int = 314159, samples: int = 10000
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    draws = generator.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return {
        "team_differences": values,
        "mean_difference": float(array.mean()),
        "bootstrap_samples": samples,
        "bootstrap_95_percent_interval": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "note": "Exploratory team-level bootstrap with only four independent teams.",
    }


def run_correction_study(
    repository_root: Path,
    output_root: Path,
    *,
    device: str = "cpu",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    repo = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"study output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    plan_path = (
        repo / "evidence" / "gum-school" / "studies"
        / "cooperative-gradient-attribution-v1" / "STUDY_PLAN.json"
    )
    correction_path = plan_path.with_name("CORRECTION_PLAN.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    attribution_path = repo / correction["attribution_source"]["artifact"]
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))

    existing_seeds = {
        int(row["seed"])
        for replication in correction["replications"]
        for row in _records(repo / replication["team_root"])
    }
    study_seeds = {
        seed
        for replication in correction["replications"]
        for seed in range(
            int(replication["fresh_evaluation_seed_range"][0]),
            int(replication["fresh_evaluation_seed_range"][1]) + 1,
        )
    }
    prohibited = set(range(
        int(plan["evaluation"]["previously_revealed_seed_block_prohibited"][0]),
        int(plan["evaluation"]["previously_revealed_seed_block_prohibited"][1]) + 1,
    ))
    if study_seeds & existing_seeds or study_seeds & prohibited:
        raise EscapeTeamError("preregistered evaluation seeds are not fresh")

    teams = []
    seen_members: set[str] = set()
    recovery = {
        "format": "gum-cooperative-gradient-study-recovery-v1",
        "status": "running",
        "completed_teams": [],
    }
    recovery_path = output / "RECOVERY.json"
    atomic_write_json(recovery_path, recovery, backup=False, sort_keys=True)
    for replication in correction["replications"]:
        team_started = time.perf_counter()
        team_name = replication["team"]
        if progress:
            progress(f"{team_name}: verifying selected trajectory and checkpoints")
        root = repo / replication["team_root"]
        source = _prepare_selected_episode(root, replication["selected_episode_id"])
        identities = source["identities"]
        member_ids = {identity["member_id"] for identity in identities}
        if seen_members & member_ids:
            raise EscapeTeamError("team replications share durable member identities")
        seen_members.update(member_ids)

        control = _manifest_learners(root, source["before"], device=device)
        original = _manifest_learners(root, source["before"], device=device)
        corrected = _manifest_learners(root, source["before"], device=device)
        original_updates = [
            learner.fit_recorded_actor_critic(trajectory)
            for learner, trajectory in zip(
                original, source["trajectories"], strict=True
            )
        ]
        saved_after = _manifest_learners(root, source["after"], device=device)
        reconstruction = [
            _weight_difference(learner, saved)
            for learner, saved in zip(original, saved_after, strict=True)
        ]
        if not all(row["exact_tensor_equality"] for row in reconstruction):
            raise EscapeTeamError(f"{team_name} original update did not reproduce")
        for learner in corrected:
            _configure_correction(learner)
        corrected_updates = [
            learner.fit_recorded_actor_critic(trajectory)
            for learner, trajectory in zip(
                corrected, source["trajectories"], strict=True
            )
        ]
        branches = {"control": control, "original": original, "corrected": corrected}
        recorded_contexts = _recorded_contexts(
            source["trajectories"], identities, source["roles"], branches
        )
        branch_manifests = {
            "control": _save_branch(
                output, team_name, "control", identities, control,
                source_checkpoint=source["before_label"], update="none",
            ),
            "original": _save_branch(
                output, team_name, "original", identities, original,
                source_checkpoint=source["before_label"], update="original-full",
            ),
            "corrected": _save_branch(
                output, team_name, "corrected", identities, corrected,
                source_checkpoint=source["before_label"],
                update="full-with-value-shared-gradient-scale-zero",
            ),
        }
        seeds = list(range(
            int(replication["fresh_evaluation_seed_range"][0]),
            int(replication["fresh_evaluation_seed_range"][1]) + 1,
        ))
        evaluation = {}
        for branch, learners in branches.items():
            if progress:
                progress(f"{team_name}: frozen evaluation branch={branch}")
            evaluation[branch] = _evaluate_branch(
                learners,
                seeds=seeds,
                horizon=int(plan["budgets"]["evaluation_horizon"]),
                adapter=source["capsule"]["environment_adapter"],
                reward_table=source["reward_table"],
                output=output,
                team_name=team_name,
                branch=branch,
            )
        history = source["records"][: source["position"]]
        team_result = {
            "team": team_name,
            "team_root": replication["team_root"],
            "member_ids": sorted(member_ids),
            "source_history_before_selected_episode": {
                "episodes": len(history),
                "training_episodes": sum(row["training"] for row in history),
                "training_failures": sum(
                    row["training"] and int(row["escaped_count"]) == 0
                    for row in history
                ),
                "training_successes": sum(
                    row["training"] and int(row["escaped_count"]) > 0
                    for row in history
                ),
            },
            "selected_episode": {
                "episode_id": source["capsule"]["episode_id"],
                "seed": source["capsule"]["seed"],
                "escaped_count": source["capsule"]["escaped_count"],
                "archive": source["capsule"]["archive"],
                "selection": "earliest successful training episode",
                "pre_update_checkpoint": source["before_label"],
                "diagnostic_roles": source["roles"],
            },
            "original_reconstruction": reconstruction,
            "updates": {
                "original": original_updates,
                "corrected": corrected_updates,
            },
            "recorded_contexts": recorded_contexts,
            "branch_manifests": branch_manifests,
            "evaluation": evaluation,
            "wall_time_seconds": time.perf_counter() - team_started,
        }
        teams.append(team_result)
        recovery["completed_teams"].append({
            "team": team_name,
            "branch_manifests": branch_manifests,
            "evaluation_complete": True,
        })
        atomic_write_json(recovery_path, recovery, backup=False, sort_keys=True)

    summaries = {
        branch: _branch_summary(teams, branch)
        for branch in ("control", "original", "corrected")
    }
    team_differences = [
        (
            team["evaluation"]["corrected"]["total_escapes"]
            - team["evaluation"]["original"]["total_escapes"]
        ) / team["evaluation"]["original"]["episodes"]
        for team in teams
    ]
    corrected_wins = sum(value > 0 for value in team_differences)
    held_out_pass = (
        corrected_wins >= 3
        and summaries["corrected"]["total_escapes"]
        > summaries["original"]["total_escapes"]
    )

    audited_actor = {
        (member["name"], context["diagnostic_label"]): context[
            "raw_action_probability"
        ]["actor"]
        for member in attribution["members"]
        for context in member["diagnostic_contexts"]
    }
    audited_rows = teams[0]["recorded_contexts"]
    closeness = []
    for row in audited_rows:
        key = (row["name"], row["diagnostic_label"])
        actor_probability = audited_actor[key]
        original_distance = abs(row["probability"]["original"] - actor_probability)
        corrected_distance = abs(row["probability"]["corrected"] - actor_probability)
        closeness.append({
            "name": row["name"],
            "diagnostic_label": row["diagnostic_label"],
            "actor_only_probability": actor_probability,
            "original_distance_to_actor_only": original_distance,
            "corrected_distance_to_actor_only": corrected_distance,
            "corrected_is_closer": corrected_distance < original_distance,
        })
    recorded_pass = sum(row["corrected_is_closer"] for row in closeness) >= 2
    if recorded_pass and held_out_pass:
        decision = "continue-to-sealed-evaluation"
    elif recorded_pass:
        decision = "recorded-context-only-retention-generalization-gap"
    else:
        decision = "stop-isolated-patches-prepare-established-baseline"

    recovery["status"] = "complete"
    atomic_write_json(recovery_path, recovery, backup=False, sort_keys=True)
    return {
        "format": STUDY_FORMAT,
        "study_plan": {
            "path": plan_path.relative_to(repo).as_posix(),
            "sha256": f"sha256:{file_sha256(plan_path)}",
        },
        "correction_plan": {
            "path": correction_path.relative_to(repo).as_posix(),
            "sha256": f"sha256:{file_sha256(correction_path)}",
        },
        "attribution": {
            "path": correction["attribution_source"]["artifact"],
            "sha256": f"sha256:{file_sha256(attribution_path)}",
        },
        "device": device,
        "semantic_labels_used_for_training": False,
        "room_a_modified_or_trained": False,
        "new_training_environment_interactions": 0,
        "replay_optimizer_steps": 32,
        "teams": teams,
        "aggregate": summaries,
        "uncertainty": {
            "corrected_minus_original_escape_rate": _bootstrap_team_difference(
                team_differences
            )
        },
        "mechanism_check": {
            "audited_contexts": closeness,
            "required_closer_contexts": 2,
            "passed": recorded_pass,
        },
        "held_out_check": {
            "team_escape_rate_differences_corrected_minus_original": team_differences,
            "teams_with_strict_improvement": corrected_wins,
            "required_teams_with_strict_improvement": 3,
            "aggregate_corrected_exceeds_original": (
                summaries["corrected"]["total_escapes"]
                > summaries["original"]["total_escapes"]
            ),
            "passed": held_out_pass,
        },
        "decision": decision,
        "recovery_record": {
            "path": recovery_path.relative_to(output).as_posix(),
            "sha256": f"sha256:{file_sha256(recovery_path)}",
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
