"""Exercise the persistent four-member team on distinct development missions.

This is an architecture smoke run, not a promotion or a controlled efficacy
claim.  It runs no baseline and never opens audit-only world state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import torch

from gum.lineage import file_sha256
from gum.school.mission_swarm import MissionSwarm
from gum.school.worlds import (
    CAUSAL_WORKSHOP_ADAPTER,
    CHANGING_MAZE_ADAPTER,
    FOUNDATIONAL_LOADERS,
    create_foundational_world,
)
from gum.storage import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
PROMOTED = (
    ROOT / "evidence" / "gum-school"
    / "sealed-recurrent-composition-official-v1" / "snapshots"
    / "sha256-059137c51bcf5e977d4e56116590cadf1c64d63e9532b930cde60a2758d866a1"
    / "state" / "CAUSAL_META_POLICY.pt"
)
DEFAULT_OUTPUT = (
    ROOT / "evidence" / "gum-school" / "research"
    / "cooperative-mission-swarm-v2"
)
DEFAULT_WORK = ROOT / ".test-temp" / "cooperative-mission-swarm-v2"
MISSIONS = (
    (CAUSAL_WORKSHOP_ADAPTER, "composition", 21_501),
    (CHANGING_MAZE_ADAPTER, "memory", 31_801),
    (CHANGING_MAZE_ADAPTER, "memory", 31_802),
    (CHANGING_MAZE_ADAPTER, "memory", 31_803),
    (CHANGING_MAZE_ADAPTER, "memory", 31_804),
)


def _run_mission(swarm: MissionSwarm, world, *, seed: int) -> dict[str, Any]:
    observation = world.reset(seed)
    started = swarm.begin_mission(
        world.public_spec(),
        observation,
        training=True,
        objective="explore, adapt, reach positive public reward, and preserve the result",
    )
    success = False
    final_reward = 0.0
    for _ in range(world.public_spec().horizon):
        action = swarm.act(observation)
        transition = world.step(action)
        swarm.observe(action, transition)
        observation = transition.observation
        final_reward = float(transition.reward)
        if transition.terminated or transition.truncated:
            success = bool(transition.terminated and final_reward > 0.0)
            break
    capsule = swarm.finish_mission(success=success)
    return {
        "mission_id": capsule["mission_id"],
        "world_id": world.public_spec().world_id,
        "family": world.public_spec().family,
        "seed": int(seed),
        "captain_id": started["captain_id"],
        "attempt": started["attempt"],
        "success": capsule["success"],
        "interactions": capsule["interactions"],
        "return": capsule["return"],
        "proposal_disagreements": capsule["proposal_disagreements"],
        "shared_imports": len(capsule["shared_imports"]),
        "experience": capsule["experience"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--policy", type=Path, default=PROMOTED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    work = args.work.resolve()
    if output.exists() or work.exists():
        raise SystemExit("refusing to overwrite an existing mission archive")
    output.mkdir(parents=True)
    work.mkdir(parents=True)
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    started = time.perf_counter()
    swarm = MissionSwarm.create(
        output / "swarm",
        source_policy=args.policy.resolve(),
        device=device,
    )
    rows = []
    for index, (adapter, mechanism, seed) in enumerate(MISSIONS):
        package = create_foundational_world(
            work / f"mission-{index:02d}",
            adapter,
            seed=seed,
            mechanism=mechanism,
        )
        world = FOUNDATIONAL_LOADERS[adapter](package)
        row = _run_mission(swarm, world, seed=seed)
        rows.append(row)
        print(
            f"mission {index + 1}/{len(MISSIONS)} "
            f"family={row['family']} success={row['success']} "
            f"steps={row['interactions']}"
        )

    reloaded = MissionSwarm.load(output / "swarm", device=device)
    status = reloaded.status()
    report = {
        "format": "gum-school-cooperative-mission-swarm-smoke-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "architecture smoke run; no baseline and no promotion claim",
        "source_policy": {
            "path": args.policy.resolve().relative_to(ROOT).as_posix(),
            "sha256": f"sha256:{file_sha256(args.policy.resolve())}",
        },
        "device": device,
        "wall_clock_seconds": time.perf_counter() - started,
        "missions": rows,
        "status_after_reload": status,
        "design_checks": {
            "four_durable_identities": len(status["members"]) == 4,
            "captaincy_distributed": all(
                member["captain_missions"] >= 1 for member in status["members"]
            ),
            "all_missions_preserved": status["experience_archive"]["valid"]
            and status["experience_archive"]["records"] == len(rows),
            "mission_ledger_valid": status["mission_ledger"]["valid"],
            "knowledge_ledger_valid": status["knowledge_ledger"]["valid"],
            "distinct_worlds_not_blind_retries": len(
                {row["world_id"] for row in rows}
            ) == len(rows),
            "hidden_world_state_given_to_learners": False,
        },
    }
    atomic_write_json(
        output / "MISSION_SWARM_SMOKE_REPORT.json",
        report,
        backup=False,
        sort_keys=True,
    )
    print(json.dumps(report["design_checks"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
