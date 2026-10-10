"""Locked, outcome-neutral development pilot of GUM and independent PPO."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np
import psutil
import torch

from gum.lineage import file_sha256
from gum.storage import atomic_write_json
from .escape_chamber import DEVELOPMENT_ADAPTER, RewardTable
from .escape_team import EscapeTeam, EscapeMemberIdentity, _EscapeMember
from .independent_ppo import PPOConfig, IndependentPPOLearner, ElapsedRewardGUMLearner
from .recurrent_meta import RecurrentMetaConfig


PROTOCOL_PATH = Path("evidence/gum-school/studies/independent-ppo-v1/PROTOCOL.json")


def create_baseline_team(root, method, seeds, *, device="cpu"):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("team directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    members = []
    for i, seed in enumerate(seeds):
        if method == "ppo":
            learner = IndependentPPOLearner(seed, device=device)
        elif method == "gum":
            learner = ElapsedRewardGUMLearner(seed, device=device, config=RecurrentMetaConfig(
                batch_episodes=1, training_exploration_mix=.2, episodic_action_exploration_mix=0.,
                episodic_novelty_coefficient=0., evaluation_temperature=1., value_shared_gradient_scale=1.))
        else:
            raise ValueError("unknown method")
        identity = EscapeMemberIdentity.create(i, seed)
        atomic_write_json(root / "members" / identity.member_id / "IDENTITY.json", identity.to_json(), backup=False)
        members.append(_EscapeMember(identity, learner))
    initialization = {"kind": "independent-random-weights", "learning_method": (
        "independent-ppo" if method == "ppo" else "gum-elapsed-reward-reference"),
        "config": asdict(members[0].learner.config), "parameter_sharing": False,
        "elapsed_inactive_reward_discounting": True, "source_policy_sha256": None}
    team = EscapeTeam(root, members, reward_table=RewardTable(), initialization=initialization)
    team.ledger.append("team-born", {"initialization": initialization,
                                   "reward_table": team.reward_table.to_json(), "sharing_mode": "off"})
    team._checkpoint("birth")
    return team


def policy_digest(team):
    digest = __import__("hashlib").sha256()
    for member in team.members:
        for name, tensor in member.learner.policy.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def learning_digest(team):
    """Weights, optimizer, pending on-policy data and training/update RNGs."""
    digest = __import__("hashlib").sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode()); visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value: visit(item)
        else: digest.update(repr(value).encode())
    for member in team.members:
        learner = member.learner
        visit(learner.policy.state_dict()); visit(learner.optimizer.state_dict())
        visit(learner.pending if isinstance(learner, IndependentPPOLearner) else learner._pending)
        visit(learner.training_generator.get_state() if isinstance(learner, IndependentPPOLearner)
              else learner._training_generator.get_state())
        if isinstance(learner, IndependentPPOLearner): visit(learner.update_generator.get_state())
    return digest.hexdigest()


def episode_metrics(team, seed, *, training, horizon, on_step=None):
    counters = {"gate_crossings": 0, "gate_open_ticks": 0, "plate_ticks": 0,
                "maximum_sustained_plate_ticks": 0, "individual_actions": 0}
    streak = [0] * 4
    def callback(world, row):
        counters["gate_crossings"] += len(world.last_resolution["gate_crossers"])
        counters["gate_open_ticks"] += int(world.gate_open)
        counters["plate_ticks"] += sum(p == world.plate for p in world.positions)
        counters["individual_actions"] += sum(a >= 0 for a in row["actions"])
        for i, position in enumerate(world.positions):
            streak[i] = streak[i] + 1 if position == world.plate and not world.escaped[i] else 0
        counters["maximum_sustained_plate_ticks"] = max(counters["maximum_sustained_plate_ticks"], *streak)
        if on_step: on_step(world, row, counters)
    capsule = team.run_episode(seed=seed, training=training, horizon=horizon,
                               environment_adapter=DEVELOPMENT_ADAPTER, on_step=callback)
    learning_records = []
    if training:
        for index, member in enumerate(team.members):
            learner = member.learner
            record = (learner.last_rollout if isinstance(learner, IndependentPPOLearner) else {
                "old_logp": torch.stack(learner._log_probs).detach().cpu(),
                "old_value": torch.stack(learner._values).detach().cpu(),
                "entropy": torch.stack(learner._entropies).detach().cpu(),
                "reward_with_discounted_inactive_tail": torch.tensor(learner._rewards)})
            path = team.root / "learning-records" / f"{capsule['episode_id']}-{index}.pt"
            path.parent.mkdir(exist_ok=True)
            torch.save(record, path)
            learning_records.append({"path":path.relative_to(team.root).as_posix(), "sha256":file_sha256(path)})
    return {"episode_id": capsule["episode_id"], "seed": seed, "training": training,
            "escapes": capsule["escaped_count"], "legal_maximum": capsule["legal_maximum_reached"],
            "joint_ticks": capsule["steps"], "completion_tick": capsule["steps"] if capsule["legal_maximum_reached"] else None,
            "returns": capsule["returns"], "archive": capsule["archive"], "updates": capsule["updates"],
            "learning_records": learning_records, **counters}


def outcome_summary(rows):
    return {"episodes": len(rows), "completions": sum(r["legal_maximum"] for r in rows),
            "completion_rate": float(np.mean([r["legal_maximum"] for r in rows])),
            "escapes": sum(r["escapes"] for r in rows), "gate_crossings": sum(r["gate_crossings"] for r in rows),
            "plate_ticks": sum(r["plate_ticks"] for r in rows),
            "mean_maximum_sustained_plate_ticks": float(np.mean([r["maximum_sustained_plate_ticks"] for r in rows])),
            "mean_completion_tick_when_complete": (float(np.mean([r["completion_tick"] for r in rows if r["legal_maximum"]]))
                                                    if any(r["legal_maximum"] for r in rows) else None),
            "joint_ticks": sum(r["joint_ticks"] for r in rows), "individual_actions": sum(r["individual_actions"] for r in rows)}


def analyze(rows, protocol):
    checkpoints = protocol["evaluation_checkpoints"]
    curves = {method: [] for method in ("gum", "ppo")}
    final = {}
    for method in curves:
        selected = [row for row in rows if row["method"] == method]
        for checkpoint in checkpoints:
            rates = [row["evaluations"][str(checkpoint)]["summary"]["completion_rate"] for row in selected]
            curves[method].append({"training_episodes": checkpoint, "team_completion_rates": rates,
                                   "mean_completion_rate": float(np.mean(rates))})
        before = np.array([r["evaluations"]["0"]["summary"]["completion_rate"] for r in selected])
        after = np.array([r["evaluations"][str(checkpoints[-1])]["summary"]["completion_rate"] for r in selected])
        changes = after - before
        rng = np.random.default_rng(protocol["analysis_seed"])
        bootstrap = changes[rng.integers(0, len(changes), (10000, len(changes)))].mean(1)
        ci = np.quantile(bootstrap, [.025, .975]).tolist()
        # Also report a t interval: all-zero bootstrap does not bound unknown
        # population success probability. These are exploratory, n=4 intervals.
        from scipy.stats import t
        half = float(t.ppf(.975, len(changes)-1) * changes.std(ddof=1) / np.sqrt(len(changes)))
        meets = int(sum((after >= protocol["success_rule"]["minimum_final_rate"]) &
                        (changes >= protocol["success_rule"]["minimum_absolute_gain"])))
        final[method] = {"team_gains": changes.tolist(), "mean_gain": float(changes.mean()),
                         "exploratory_team_bootstrap_95_interval": ci,
                         "exploratory_team_t_95_interval": [float(changes.mean()-half), float(changes.mean()+half)],
                         "teams_meeting_threshold": meets,
                         "development_success": meets >= protocol["success_rule"]["minimum_teams"] and ci[0] > 0}
    gum_rates = np.array(curves["gum"][-1]["team_completion_rates"])
    ppo_rates = np.array(curves["ppo"][-1]["team_completion_rates"])
    differences = ppo_rates - gum_rates
    rng = np.random.default_rng(protocol["analysis_seed"]+1)
    comparative_ci = np.quantile(differences[rng.integers(0,len(differences),(10000,len(differences)))].mean(1),[.025,.975]).tolist()
    return {"curves": curves, "methods": final, "paired_team_ppo_minus_gum": differences.tolist(),
            "paired_exploratory_bootstrap_95_interval": comparative_ci,
            "decision": ("freeze-successful-candidate-and-preregister-room-a-transfer" if any(r["development_success"] for r in final.values())
                         else "both-failed-under-pilot-budget-no-room-a-examination"),
            "uncertainty_limit": "Four team replications are exploratory. Degenerate zero intervals describe observed ties; they do not prove a population effect or impossibility."}


def run_study(output, *, device="cpu", viewer=None):
    repo = Path(__file__).resolve().parents[2]
    protocol = json.loads((repo / PROTOCOL_PATH).read_text())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()): raise FileExistsError("study output must be empty")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    validation_path = repo / "evidence/gum-school/studies/independent-ppo-v1/VALIDATION.json"
    validation = json.loads(validation_path.read_text())
    if not validation["passed"]: raise RuntimeError("PPO implementation validation failed")
    # Abort if the immutable chamber source differs from the registered hash.
    if file_sha256(repo / "gum/school/escape_chamber.py") != protocol["chamber_source_sha256"]:
        raise RuntimeError("frozen environment source changed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    provenance = {"source_commit": commit, "protocol_sha256": file_sha256(repo / PROTOCOL_PATH),
                  "validation_sha256": file_sha256(validation_path), "python": platform.python_version(),
                  "torch": torch.__version__, "numpy": np.__version__, "device": device,
                  "cpu": platform.processor(), "torch_threads": 1,
                  "dependencies": subprocess.check_output([__import__("sys").executable,"-m","pip","freeze"],text=True).splitlines(),
                  "started_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_write_json(output / "PROVENANCE.json", provenance, backup=False)
    atomic_write_json(output / "PROTOCOL.json", protocol, backup=False)
    started = time.perf_counter()
    rows = []
    for index, team_seed in enumerate(protocol["team_seeds"]):
        for method in ("gum", "ppo"):
            root = output / f"team-{index+1}" / method
            seeds = tuple(team_seed + offset for offset in (101,211,307,401))
            team = create_baseline_team(root, method, seeds, device=device)
            actual_config = asdict(team.members[0].learner.config)
            for key, value in protocol[method+"_config"].items():
                if key != "adam_epsilon" and actual_config[key] != value:
                    raise RuntimeError(f"configuration differs from protocol: {method} {key}")
            row = {"team": index+1, "team_seed": team_seed, "member_seeds": seeds, "method": method,
                   "root": root.relative_to(output).as_posix(), "configuration": team.initialization,
                   "parameters_per_member": [m.learner.status()["parameter_count"] for m in team.members],
                   "training": [], "evaluations": {}, "resources": {"training_wall_seconds": 0., "evaluation_wall_seconds": 0.,
                                                                       "peak_process_rss_bytes": 0}}
            phase_resources = {"training_joint_ticks": 0, "evaluation_joint_ticks": 0,
                               "training_individual_actions": 0, "evaluation_individual_actions": 0}
            def perform(seed, training, episode):
                phase = "training" if training else "frozen evaluation"
                before = learning_digest(team) if not training else None
                wall_start = time.perf_counter()
                def show(world, step, counts):
                    row["resources"]["peak_process_rss_bytes"] = max(row["resources"]["peak_process_rss_bytes"],psutil.Process().memory_info().rss)
                    if viewer:
                        viewer.update(world, step, {"method": method.upper(), "phase": phase, "team_seed": team_seed,
                            "team": index+1, "episode": episode, "checkpoint": len(row["training"]),
                            "cumulative": {**phase_resources, "current_episode_ticks": world.steps},
                            "wall_seconds": time.perf_counter()-started, "parameters": row["parameters_per_member"],
                            "optimizer_steps": sum((m.learner.optimizer_steps if method=="ppo" else m.learner.training_episodes) for m in team.members)})
                result = episode_metrics(team,seed,training=training,horizon=protocol["horizon"],on_step=show)
                row["resources"]["training_wall_seconds" if training else "evaluation_wall_seconds"] += time.perf_counter()-wall_start
                prefix = "training" if training else "evaluation"
                phase_resources[prefix+"_joint_ticks"] += result["joint_ticks"]
                phase_resources[prefix+"_individual_actions"] += result["individual_actions"]
                if not training and learning_digest(team) != before:
                    raise RuntimeError("frozen evaluation changed learning state")
                return result
            def evaluate(checkpoint):
                seeds = protocol["evaluation_seed_bases"][index]
                results = [perform(seeds+i,False,i+1) for i in range(protocol["evaluation_episodes"])]
                row["evaluations"][str(checkpoint)] = {"summary": outcome_summary(results), "episodes": results,
                                                       "learning_state_frozen_verified": True}
                row["resources"].update(phase_resources)
                atomic_write_json(output / "RESULTS_IN_PROGRESS.json", {"completed": rows, "current": row},backup=False)
                print(json.dumps({"team":index+1,"method":method,"checkpoint":checkpoint,
                                  "evaluation":outcome_summary(results)}),flush=True)
            evaluate(0)
            for episode in range(protocol["training_episodes"]):
                row["training"].append(perform(protocol["training_seed_bases"][index]+episode,True,episode+1))
                if episode+1 in protocol["evaluation_checkpoints"]: evaluate(episode+1)
            row["training_summary"] = outcome_summary(row["training"])
            row["resources"].update(phase_resources)
            row["resources"]["optimizer_steps"] = sum((m.learner.optimizer_steps if method=="ppo" else m.learner.training_episodes) for m in team.members)
            row["resources"]["optimization_epochs"] = sum((m.learner.optimization_epochs if method=="ppo" else m.learner.training_episodes) for m in team.members)
            row["resources"]["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
            restored = EscapeTeam.load(root,device=device)
            if learning_digest(restored) != learning_digest(team): raise RuntimeError("checkpoint restoration differs")
            row["recovery_verified"] = True
            rows.append(row)
    result = {"format":"gum-independent-ppo-development-study-v1", "protocol":protocol,"provenance":provenance,
              "teams":rows,"analysis":analyze(rows,protocol), "wall_seconds":time.perf_counter()-started,
              "hard_room_a_used":False,"implementation_checks_passed":True,"tuning_environment_interactions":0}
    atomic_write_json(output / "STUDY_RESULTS.json",result,backup=False)
    return result
