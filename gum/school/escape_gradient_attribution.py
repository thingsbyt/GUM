"""Loss-component attribution for one preserved cooperative escape episode."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gum.lineage import file_sha256

from .escape_chamber import RewardTable
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


ATTRIBUTION_FORMAT = "gum-cooperative-gradient-attribution-v1"
COMPONENTS = ("actor", "value", "entropy", "full")
PARAMETER_GROUPS = ("visual_encoder", "recurrent_layers", "policy_head", "value_head")


def _parameter_group(name: str) -> str:
    if name.startswith("visual."):
        return "visual_encoder"
    if name.startswith("memory.") or name.startswith("action_memory."):
        return "recurrent_layers"
    if name.startswith("actor."):
        return "policy_head"
    if name.startswith("critic."):
        return "value_head"
    raise ValueError(f"unassigned policy parameter: {name}")


def _recorded_graph(
    learner: RecurrentCausalLearner,
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    """Build the exact one-episode graph used by the ordinary update."""
    action_count = int(trajectory["action_count"])
    observations = trajectory["observations"]
    actions = [int(value) for value in trajectory["actions"]]
    rewards = [float(value) for value in trajectory["rewards"]]
    if not observations or not (len(observations) == len(actions) == len(rewards)):
        raise ValueError("recorded attribution trajectory is incomplete")
    if learner.config.episodic_action_exploration_mix != 0.0:
        raise ValueError("gradient attribution requires the episodic action mixer off")

    hidden = torch.zeros(1, learner.config.hidden_size, device=learner.device)
    action_memory = torch.zeros(
        1,
        learner.config.action_count,
        learner.config.action_memory_size,
        device=learner.device,
    )
    previous_action = learner.config.action_count
    previous_reward = 0.0
    log_probs = []
    values = []
    entropies = []
    raw_probabilities = []
    for step, (observation, action, reward) in enumerate(
        zip(observations, actions, rewards, strict=True)
    ):
        logits, value, hidden, action_memory = learner.policy(
            torch.from_numpy(np.asarray(observation).copy()).to(learner.device),
            torch.tensor([previous_action], device=learner.device),
            torch.tensor([previous_reward], dtype=torch.float32, device=learner.device),
            torch.tensor([float(step == 0)], dtype=torch.float32, device=learner.device),
            hidden,
            action_memory,
        )
        raw = torch.softmax(logits[:, :action_count], dim=-1)
        probabilities = raw
        if learner.config.training_exploration_mix > 0.0:
            mix = learner.config.training_exploration_mix
            probabilities = (1.0 - mix) * raw + mix / action_count
        log_all = torch.log(
            probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
        )
        target = torch.tensor([[action]], device=learner.device)
        log_probs.append(log_all.gather(1, target).squeeze())
        values.append(value.squeeze())
        entropies.append(-(probabilities * log_all).sum())
        raw_probabilities.append(raw[0, action])
        previous_action = action
        previous_reward = reward

    log_prob_tensor = torch.stack(log_probs)
    value_tensor = torch.stack(values)
    entropy_tensor = torch.stack(entropies)
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + learner.config.discount * running
        returns.append(running)
    returns_tensor = torch.tensor(
        list(reversed(returns)), dtype=torch.float32, device=learner.device
    )
    raw_advantages = returns_tensor - value_tensor.detach()
    normalized_advantages = raw_advantages
    if len(normalized_advantages) > 1:
        normalized_advantages = (
            normalized_advantages - normalized_advantages.mean()
        ) / (normalized_advantages.std(unbiased=False) + 1e-6)
    actor_tokens = -(log_prob_tensor * normalized_advantages)
    actor_loss = actor_tokens.mean()
    value_loss = F.mse_loss(value_tensor, returns_tensor)
    entropy = entropy_tensor.mean()
    losses = {
        "actor": actor_loss,
        "value": learner.config.value_coefficient * value_loss,
        "entropy": -learner.config.entropy_coefficient * entropy,
    }
    losses["full"] = losses["actor"] + losses["value"] + losses["entropy"]
    return {
        "losses": losses,
        "actor_tokens": actor_tokens,
        "returns": returns_tensor,
        "values": value_tensor,
        "raw_advantages": raw_advantages,
        "normalized_advantages": normalized_advantages,
        "raw_chosen_probabilities": torch.stack(raw_probabilities),
        "entropy": entropy,
        "unscaled_value_loss": value_loss,
    }


def _gradients(
    learner: RecurrentCausalLearner,
    loss: torch.Tensor,
) -> dict[str, torch.Tensor]:
    learner.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    result: dict[str, list[torch.Tensor]] = {name: [] for name in PARAMETER_GROUPS}
    result["all"] = []
    for name, parameter in learner.policy.named_parameters():
        gradient = (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach()
        )
        flat = gradient.to("cpu", torch.float64).reshape(-1)
        result[_parameter_group(name)].append(flat)
        result["all"].append(flat)
    return {
        name: torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.float64)
        for name, chunks in result.items()
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    if denominator == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def _norms(vectors: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        name: float(torch.linalg.vector_norm(vector))
        for name, vector in vectors.items()
    }


def _state_copy(learner: RecurrentCausalLearner) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to("cpu", torch.float64).clone()
        for name, value in learner.policy.state_dict().items()
    }


def _parameter_deltas(
    before: dict[str, torch.Tensor],
    learner: RecurrentCausalLearner,
) -> dict[str, torch.Tensor]:
    grouped: dict[str, list[torch.Tensor]] = {name: [] for name in PARAMETER_GROUPS}
    grouped["all"] = []
    for name, parameter in learner.policy.named_parameters():
        delta = parameter.detach().to("cpu", torch.float64) - before[name]
        flat = delta.reshape(-1)
        grouped[_parameter_group(name)].append(flat)
        grouped["all"].append(flat)
    return {name: torch.cat(chunks) for name, chunks in grouped.items()}


def _apply_isolated_update(
    learner: RecurrentCausalLearner,
    trajectory: dict[str, Any],
    component: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if component not in ("actor", "value", "entropy"):
        raise ValueError("isolated component must be actor, value, or entropy")
    before = _state_copy(learner)
    graph = _recorded_graph(learner, trajectory)
    vectors = _gradients(learner, graph["losses"][component])
    raw_norm = float(torch.linalg.vector_norm(vectors["all"]))
    clip_scale = min(1.0, learner.config.gradient_clip / max(raw_norm, 1e-30))
    nn.utils.clip_grad_norm_(learner.policy.parameters(), learner.config.gradient_clip)
    learner.optimizer.step()
    deltas = _parameter_deltas(before, learner)
    optimizer_alignment = {
        group: _cosine(deltas[group], -vectors[group])
        for group in (*PARAMETER_GROUPS, "all")
    }
    return ({
        "component": component,
        "loss": float(graph["losses"][component].detach()),
        "raw_gradient_norm": raw_norm,
        "gradient_clip_scale": clip_scale,
        "gradient_norms": _norms(vectors),
        "parameter_delta_norms": _norms(deltas),
        "adam_step_cosine_with_negative_gradient": optimizer_alignment,
    }, vectors)


def _component_gradients(
    checkpoint: Path,
    trajectory: dict[str, Any],
    *,
    device: str,
) -> dict[str, dict[str, torch.Tensor]]:
    result = {}
    for component in COMPONENTS:
        learner = RecurrentCausalLearner.load(checkpoint, device=device)
        graph = _recorded_graph(learner, trajectory)
        result[component] = _gradients(learner, graph["losses"][component])
    return result


def _gradient_alignment(
    component_vectors: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    pairs = {}
    for left, right in itertools.combinations(COMPONENTS, 2):
        pairs[f"{left}_vs_{right}"] = {
            group: _cosine(component_vectors[left][group], component_vectors[right][group])
            for group in (*PARAMETER_GROUPS, "all")
        }
    sum_error = {}
    for group in (*PARAMETER_GROUPS, "all"):
        combined = (
            component_vectors["actor"][group]
            + component_vectors["value"][group]
            + component_vectors["entropy"][group]
        )
        difference = component_vectors["full"][group] - combined
        sum_error[group] = float(torch.linalg.vector_norm(difference))
    return {"pairwise_cosines": pairs, "full_minus_component_sum_norm": sum_error}


def _segment_masks(ticks: list[int], gate_tick: int, escape_tick: int) -> dict[str, list[int]]:
    return {
        "before_gate": [index for index, tick in enumerate(ticks) if tick < gate_tick],
        "gate_tick": [index for index, tick in enumerate(ticks) if tick == gate_tick],
        "between_gate_and_escape": [
            index for index, tick in enumerate(ticks) if gate_tick < tick < escape_tick
        ],
        "escape_tick": [index for index, tick in enumerate(ticks) if tick == escape_tick],
        "after_escape": [index for index, tick in enumerate(ticks) if tick > escape_tick],
    }


def _segment_actor_gradients(
    checkpoint: Path,
    trajectory: dict[str, Any],
    *,
    gate_tick: int,
    escape_tick: int,
    device: str,
) -> dict[str, Any]:
    vectors = {}
    token_count = len(trajectory["ticks"])
    for segment, indices in _segment_masks(
        trajectory["ticks"], gate_tick, escape_tick
    ).items():
        if not indices:
            vectors[segment] = None
            continue
        learner = RecurrentCausalLearner.load(checkpoint, device=device)
        graph = _recorded_graph(learner, trajectory)
        index = torch.tensor(indices, dtype=torch.int64, device=learner.device)
        contribution = graph["actor_tokens"].index_select(0, index).sum() / token_count
        vectors[segment] = _gradients(learner, contribution)
    report = {}
    for segment, value in vectors.items():
        report[segment] = {
            "ticks": [trajectory["ticks"][index] for index in _segment_masks(
                trajectory["ticks"], gate_tick, escape_tick
            )[segment]],
            "gradient_norms": None if value is None else _norms(value),
        }
    pairwise = {}
    available = [(name, value) for name, value in vectors.items() if value is not None]
    for (left_name, left), (right_name, right) in itertools.combinations(available, 2):
        pairwise[f"{left_name}_vs_{right_name}"] = {
            group: _cosine(left[group], right[group])
            for group in ("visual_encoder", "recurrent_layers", "policy_head", "all")
        }
    return {"segments": report, "pairwise_cosines": pairwise}


def _torch_hash(value: Any) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return f"sha256:{hashlib.sha256(buffer.getvalue()).hexdigest()}"


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _runtime() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def attribute_successful_episode(
    team_root: Path,
    episode_id: str,
    output_root: Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Reproduce and decompose one successful update without new experience."""
    started = time.perf_counter()
    root = Path(team_root).resolve()
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"attribution output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    records = _records(root)
    positions = [index for index, row in enumerate(records) if row["episode_id"] == episode_id]
    if len(positions) != 1:
        raise EscapeTeamError("attribution episode must appear exactly once")
    record_index = positions[0]
    capsule = records[record_index]
    if not capsule["training"] or int(capsule["escaped_count"]) < 1:
        raise EscapeTeamError("attribution requires a successful training episode")
    before_label = "birth" if record_index == 0 else records[record_index - 1]["episode_id"]
    before = _snapshot(root, before_label)
    after = _snapshot(root, episode_id)
    arrays = _archive(root, capsule)
    reward_table = RewardTable(**before["reward_table"])
    semantics = _replay_semantics(capsule, arrays, reward_table)
    exact_arrays = dict(arrays)
    exact_arrays["rewards"] = np.asarray(semantics["replayed_rewards"], dtype=np.float64)
    trajectories = _member_trajectories(exact_arrays)
    identities = [row["identity"] for row in before["members"]]
    roles = _cooperation_roles(semantics, identities)

    pre_learners = _manifest_learners(root, before, device=device)
    actual_after = _manifest_learners(root, after, device=device)
    starting_state = []
    for row, learner in zip(before["members"], pre_learners, strict=True):
        brain_path = root / row["brain"]["path"]
        starting_state.append({
            "member_id": row["identity"]["member_id"],
            "name": row["identity"]["name"],
            "brain_path": row["brain"]["path"],
            "brain_sha256": f"sha256:{file_sha256(brain_path)}",
            "manifest_brain_sha256": row["brain"]["sha256"],
            "policy_state_sha256": _torch_hash(learner.policy.state_dict()),
            "optimizer_state_sha256": _torch_hash(learner.optimizer.state_dict()),
            "training_random_state_sha256": _torch_hash(
                learner._training_generator.get_state()
            ),
            "evaluation_random_state_sha256": _torch_hash(
                learner._evaluation_generator.get_state()
            ),
            "config": asdict(learner.config),
        })

    # Gate the whole study on exact reproduction before isolated branches run.
    full_learners = _manifest_learners(root, before, device=device)
    full_updates = [
        learner.fit_recorded_actor_critic(trajectory)
        for learner, trajectory in zip(full_learners, trajectories, strict=True)
    ]
    reproduction = []
    for identity, reconstructed, saved in zip(
        identities, full_learners, actual_after, strict=True
    ):
        comparison = _weight_difference(reconstructed, saved)
        reproduction.append({
            "member_id": identity["member_id"],
            "name": identity["name"],
            "weight_match": comparison,
        })
    if not all(row["weight_match"]["exact_tensor_equality"] for row in reproduction):
        raise EscapeTeamError("original complete update did not reproduce exactly")

    branches: dict[str, list[RecurrentCausalLearner]] = {"full": full_learners}
    update_metrics: dict[str, list[dict[str, Any]]] = {"full": []}
    component_vectors_by_member = []
    for member_index, trajectory in enumerate(trajectories):
        checkpoint = root / before["members"][member_index]["brain"]["path"]
        component_vectors = _component_gradients(
            checkpoint, trajectory, device=device
        )
        component_vectors_by_member.append(component_vectors)
        full_norm = float(torch.linalg.vector_norm(component_vectors["full"]["all"]))
        update_metrics["full"].append({
            "component": "full",
            "loss": float(full_updates[member_index]["loss"]),
            "raw_gradient_norm": full_norm,
            "gradient_clip_scale": min(
                1.0,
                pre_learners[member_index].config.gradient_clip / max(full_norm, 1e-30),
            ),
            "gradient_norms": _norms(component_vectors["full"]),
        })

    for component in ("actor", "value", "entropy"):
        learners = _manifest_learners(root, before, device=device)
        metrics = []
        for learner, trajectory in zip(learners, trajectories, strict=True):
            row, _ = _apply_isolated_update(learner, trajectory, component)
            metrics.append(row)
        branches[component] = learners
        update_metrics[component] = metrics

    member_reports = []
    gate_tick = int(roles["gate_crossing_tick"])
    escape_tick = int(roles["escape_tick"])
    inspected = set()
    crosser_index = int(roles["crosser"]["member_index"])
    inspected.add((crosser_index, gate_tick, "crosser-at-gate"))
    inspected.add((crosser_index, escape_tick, "crosser-at-escape"))
    for holder in roles["holders_at_gate_crossing"]:
        inspected.add((int(holder["member_index"]), gate_tick, "holder-at-gate"))

    for member_index, (identity, trajectory) in enumerate(zip(
        identities, trajectories, strict=True
    )):
        checkpoint = root / before["members"][member_index]["brain"]["path"]
        base_learner = RecurrentCausalLearner.load(checkpoint, device=device)
        graph = _recorded_graph(base_learner, trajectory)
        traces = {
            "before": _policy_trace(base_learner, trajectory),
            **{
                component: _policy_trace(branches[component][member_index], trajectory)
                for component in COMPONENTS
            },
        }
        local_by_tick = {tick: index for index, tick in enumerate(trajectory["ticks"])}
        contexts = []
        for selected_member, tick, label in sorted(inspected):
            if selected_member != member_index or tick not in local_by_tick:
                continue
            local = local_by_tick[tick]
            contexts.append({
                "diagnostic_label": label,
                "tick": tick,
                "executed_action": trajectory["actions"][local],
                "scalar_reward": trajectory["rewards"][local],
                "return": float(graph["returns"][local].detach()),
                "value": float(graph["values"][local].detach()),
                "raw_advantage": float(graph["raw_advantages"][local].detach()),
                "normalized_advantage": float(
                    graph["normalized_advantages"][local].detach()
                ),
                "raw_action_probability": {
                    name: float(trace["raw_chosen_probabilities"][local])
                    for name, trace in traces.items()
                },
            })
        full_before = _state_copy(
            RecurrentCausalLearner.load(checkpoint, device=device)
        )
        full_deltas = _parameter_deltas(full_before, branches["full"][member_index])
        full_optimizer_alignment = {
            group: _cosine(
                full_deltas[group],
                -component_vectors_by_member[member_index]["full"][group],
            )
            for group in (*PARAMETER_GROUPS, "all")
        }
        update_metrics["full"][member_index]["parameter_delta_norms"] = _norms(
            full_deltas
        )
        update_metrics["full"][member_index][
            "adam_step_cosine_with_negative_gradient"
        ] = full_optimizer_alignment
        member_reports.append({
            "member_id": identity["member_id"],
            "name": identity["name"],
            "trajectory": {
                "active_ticks": len(trajectory["ticks"]),
                "ticks": trajectory["ticks"],
                "returns": graph["returns"].detach().cpu().tolist(),
                "values": graph["values"].detach().cpu().tolist(),
                "raw_advantages": graph["raw_advantages"].detach().cpu().tolist(),
                "normalized_advantages": (
                    graph["normalized_advantages"].detach().cpu().tolist()
                ),
            },
            "diagnostic_contexts": contexts,
            "component_gradient_norms": {
                component: _norms(vectors)
                for component, vectors in component_vectors_by_member[member_index].items()
            },
            "component_gradient_alignment": _gradient_alignment(
                component_vectors_by_member[member_index]
            ),
            "actor_temporal_attribution": _segment_actor_gradients(
                checkpoint,
                trajectory,
                gate_tick=gate_tick,
                escape_tick=escape_tick,
                device=device,
            ),
            "updates": {
                component: update_metrics[component][member_index]
                for component in COMPONENTS
            },
        })

    for component, learners in branches.items():
        for identity, learner in zip(identities, learners, strict=True):
            learner.save(
                output / "branches" / component / f"{identity['member_id']}.pt"
            )

    plan_path = (
        Path(__file__).resolve().parents[2]
        / "evidence" / "gum-school" / "studies"
        / "cooperative-gradient-attribution-v1" / "STUDY_PLAN.json"
    )
    result = {
        "format": ATTRIBUTION_FORMAT,
        "source_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "study_plan": {
            "path": str(plan_path.relative_to(Path(__file__).resolve().parents[2])),
            "sha256": f"sha256:{file_sha256(plan_path)}",
        },
        "runtime": _runtime(),
        "source_team": str(root),
        "episode_id": episode_id,
        "pre_update_checkpoint": before_label,
        "post_update_checkpoint": episode_id,
        "source_archive": capsule["archive"],
        "starting_state": starting_state,
        "reproduction_gate": {
            "passed": True,
            "members": reproduction,
            "recorded_updates": capsule["updates"],
            "reconstructed_updates": full_updates,
        },
        "semantic_labels_used_for_training": False,
        "semantic_labels_used_for_diagnostic_measurement_only": True,
        "room_a_modified_or_trained": False,
        "observed_roles_diagnostic_only": roles,
        "members": member_reports,
        "environment_interactions": 0,
        "optimizer_steps": 16,
        "wall_time_seconds": time.perf_counter() - started,
    }
    return result
