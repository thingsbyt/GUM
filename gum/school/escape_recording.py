"""Verified, non-curated recordings from saved escape-chamber transitions."""
from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from gum.lineage import HashLedger, file_sha256
from gum.storage import atomic_write_bytes, atomic_write_json

from .escape_chamber import (
    CONTRACT_VERSION,
    ROOM_A_ADAPTER,
    TEAM_SIZE,
    RewardTable,
    make_escape_chamber,
)
from .escape_team import LEDGER_FILENAME, EscapeTeam, EscapeTeamError


RECORDING_FORMAT = "gum-cooperative-escape-recording-v1"


def _episode_record(root: Path, episode_id: str | None) -> dict[str, Any]:
    ledger = root / LEDGER_FILENAME
    verification = HashLedger(ledger).verify()
    if not verification["valid"]:
        raise EscapeTeamError("cannot record from an invalid episode ledger")
    matches = []
    with ledger.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") != "episode-completed":
                continue
            payload = row.get("payload", {})
            if episode_id is None or payload.get("episode_id") == episode_id:
                matches.append(payload)
    if not matches:
        raise EscapeTeamError("the requested completed episode is not in the ledger")
    return matches[-1]


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def record_episode_gif(
    team_root: Path,
    output: Path,
    *,
    episode_id: str | None = None,
    fps: int = 15,
    width: int = 960,
) -> dict[str, Any]:
    """Replay every saved joint tick, verify it, and atomically write a GIF.

    This does not ask a policy for new actions. It reconstructs the
    authoritative world from the saved seed and every executed joint action,
    then refuses to publish if pixels, rewards, boundaries, or state hashes do
    not exactly agree with the transition archive.
    """
    root = Path(team_root).resolve()
    output = Path(output)
    if not 1 <= int(fps) <= 60:
        raise ValueError("fps must be in [1, 60]")
    if not 320 <= int(width) <= 1920:
        raise ValueError("width must be in [320, 1920]")
    capsule = _episode_record(root, episode_id)
    archive_ref = capsule.get("archive", {})
    archive_path = (root / archive_ref.get("path", "")).resolve()
    try:
        archive_path.relative_to(root)
    except ValueError as error:
        raise EscapeTeamError("transition archive path escapes the team directory") from error
    if not archive_path.is_file():
        raise EscapeTeamError("transition archive is missing")
    if f"sha256:{file_sha256(archive_path)}" != archive_ref.get("sha256"):
        raise EscapeTeamError("transition archive hash differs")

    manifest = EscapeTeam._resolve_checkpoint(root)
    method = manifest.get("initialization", {}).get("learning_method", "gum")
    method_label = "PPO" if method == "independent-ppo" else "GUM"
    phase_label = "TRAINING" if capsule["training"] else "FROZEN EVALUATION"
    table = RewardTable(**manifest["reward_table"])
    with np.load(archive_path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    actions = arrays["actions"]
    if actions.ndim != 2 or actions.shape[1] != TEAM_SIZE:
        raise EscapeTeamError("joint action archive has the wrong shape")
    horizon = len(actions)
    adapter = archive_ref.get("environment_adapter", ROOM_A_ADAPTER)
    world = make_escape_chamber(
        adapter,
        seed=int(archive_ref["environment_seed"]),
        horizon=horizon,
        reward_table=table,
    )
    observations = world.reset()
    frames = [world.spectator_frame()]
    for tick, action_row in enumerate(actions):
        if not np.array_equal(np.stack(observations), arrays["observations"][tick]):
            raise EscapeTeamError(f"observation replay mismatch at joint tick {tick + 1}")
        decoded = [None if int(value) < 0 else int(value) for value in action_row]
        step = world.step(decoded)
        expected_hash = arrays["state_hashes"][tick].decode("ascii")
        actual_hash = world.audit_state()["state_sha256"].removeprefix("sha256:")
        checks = (
            (np.array_equal(np.stack(step.observations), arrays["next_observations"][tick]), "next observation"),
            (np.allclose(step.rewards, arrays["rewards"][tick], atol=1e-7), "reward"),
            (bool(step.terminated) == bool(arrays["terminated"][tick]), "termination"),
            (bool(step.truncated) == bool(arrays["truncated"][tick]), "truncation"),
            (actual_hash == expected_hash, "state hash"),
        )
        for valid, label in checks:
            if not valid:
                raise EscapeTeamError(f"{label} replay mismatch at joint tick {tick + 1}")
        frames.append(world.spectator_frame())
        observations = step.observations

    resized = []
    for frame in frames:
        image = Image.fromarray(frame)
        height = round(image.height * width / image.width)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 24), fill=(9, 13, 19))
        draw.text((8, 6), f"REPLAY / {phase_label} / {method_label} / ENVIRONMENT SEED {archive_ref['environment_seed']} / EPISODE {capsule['episode_id']}", fill=(238, 231, 210))
        resized.append(image)
    buffer = io.BytesIO()
    resized[0].save(
        buffer, format="GIF", save_all=True, append_images=resized[1:],
        duration=round(1000 / fps), loop=0, optimize=True, disposal=1,
    )
    atomic_write_bytes(output, buffer.getvalue(), backup=False)
    sidecar = output.with_suffix(output.suffix + ".json")
    metadata = {
        "format": RECORDING_FORMAT,
        "episode_id": capsule["episode_id"],
        "environment_seed": int(archive_ref["environment_seed"]),
        "environment_adapter": adapter,
        "contract_version": archive_ref.get("contract_version", CONTRACT_VERSION),
        "training": bool(capsule["training"]),
        "learning_method": method_label,
        "display_phase": f"replay of {phase_label.lower()}",
        "treatment": capsule["treatment"],
        "sharing_mode": capsule["sharing_mode"],
        "joint_ticks": horizon,
        "recorded_frames": len(frames),
        "every_joint_tick_included": True,
        "fps": int(fps),
        "escaped_count": int(capsule["escaped_count"]),
        "legal_maximum": TEAM_SIZE - 1,
        "legal_maximum_reached": bool(capsule["legal_maximum_reached"]),
        "archive": archive_ref,
        "archive_replay_verified": True,
        "recording_sha256": f"sha256:{file_sha256(output)}",
        "source_commit": _git_commit(Path(__file__).resolve().parents[2]),
    }
    atomic_write_json(sidecar, metadata, backup=False, sort_keys=True)
    return {**metadata, "recording": str(output), "sidecar": str(sidecar)}
