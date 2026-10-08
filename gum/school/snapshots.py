"""Immutable, content-addressed learner snapshots for GUM School."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from gum.lineage import canonical, file_sha256
from gum.storage import atomic_write_json


class SnapshotError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _regular_files(root: Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise SnapshotError(f"learner state must be a directory: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SnapshotError(f"learner state may not contain symbolic links: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise SnapshotError(f"learner state contains a non-regular entry: {path}")
    if not files:
        raise SnapshotError("learner state directory must contain at least one file")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _file_inventory(root: Path) -> tuple[dict[str, str], int]:
    files: dict[str, str] = {}
    total = 0
    for path in _regular_files(root):
        relative = path.relative_to(root).as_posix()
        files[relative] = file_sha256(path)
        total += path.stat().st_size
    return files, total


class SnapshotStore:
    """Store immutable generations and atomically point at the promoted one."""

    format = "gum-school-snapshot-v1"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.snapshots = self.root / "snapshots"
        self.pointer_path = self.root / "PROMOTED.json"
        self.snapshots.mkdir(parents=True, exist_ok=True)

    def snapshot_path(self, snapshot_id: str) -> Path:
        if not isinstance(snapshot_id, str) or not snapshot_id.startswith("sha256-"):
            raise SnapshotError(f"invalid snapshot identifier {snapshot_id!r}")
        suffix = snapshot_id.removeprefix("sha256-")
        if len(suffix) != 64 or any(character not in "0123456789abcdef" for character in suffix):
            raise SnapshotError(f"invalid snapshot identifier {snapshot_id!r}")
        return self.snapshots / snapshot_id

    def create(
        self,
        source: Path,
    ) -> dict:
        source = Path(source)
        files, total = _file_inventory(source)
        content_digest = hashlib.sha256(canonical({"files": files})).hexdigest()
        snapshot_id = f"sha256-{content_digest}"
        destination = self.snapshot_path(snapshot_id)
        if destination.exists():
            result = self.verify(snapshot_id)
            if not result["valid"]:
                raise SnapshotError(
                    f"existing snapshot {snapshot_id} failed verification: {result['errors']}"
                )
            return json.loads((destination / "SNAPSHOT.json").read_text(encoding="utf-8"))

        temporary = self.snapshots / f".{snapshot_id}.{uuid.uuid4().hex}.tmp"
        state = temporary / "state"
        state.mkdir(parents=True)
        try:
            for relative in files:
                source_path = source / Path(relative)
                target_path = state / Path(relative)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
            manifest = {
                "format": self.format,
                "snapshot_id": snapshot_id,
                "created_at_utc": _utc_now(),
                "files": files,
                "total_bytes": total,
                "lineage_location": "school ledger and decision records",
            }
            atomic_write_json(temporary / "SNAPSHOT.json", manifest, backup=False, sort_keys=True)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                resolved = temporary.resolve()
                if self.snapshots.resolve() not in resolved.parents:
                    raise SnapshotError("refusing to clean a temporary path outside snapshot storage")
                shutil.rmtree(temporary)
        result = self.verify(snapshot_id)
        if not result["valid"]:
            raise SnapshotError(f"created snapshot {snapshot_id} failed verification: {result['errors']}")
        return manifest

    def verify(self, snapshot_id: str) -> dict:
        folder = self.snapshot_path(snapshot_id)
        errors: list[str] = []
        manifest_path = folder / "SNAPSHOT.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return {"valid": False, "snapshot_id": snapshot_id, "errors": [f"manifest unreadable: {error}"]}
        if manifest.get("format") != self.format:
            errors.append("unsupported snapshot format")
        if manifest.get("snapshot_id") != snapshot_id:
            errors.append("snapshot identifier differs from manifest")
        try:
            actual_files, actual_total = _file_inventory(folder / "state")
        except SnapshotError as error:
            errors.append(str(error))
            actual_files, actual_total = {}, 0
        if actual_files != manifest.get("files"):
            errors.append("snapshot file inventory or hashes differ from manifest")
        if actual_total != manifest.get("total_bytes"):
            errors.append("snapshot byte count differs from manifest")
        digest = hashlib.sha256(canonical({"files": actual_files})).hexdigest()
        if snapshot_id != f"sha256-{digest}":
            errors.append("snapshot identifier differs from content hash")
        return {"valid": not errors, "snapshot_id": snapshot_id, "errors": errors,
                "files": len(actual_files), "total_bytes": actual_total}

    def materialize(self, snapshot_id: str, destination: Path) -> Path:
        verification = self.verify(snapshot_id)
        if not verification["valid"]:
            raise SnapshotError(f"cannot materialize invalid snapshot: {verification['errors']}")
        destination = Path(destination)
        if destination.exists():
            if any(destination.iterdir()):
                raise SnapshotError(f"candidate directory is not empty: {destination}")
        else:
            destination.mkdir(parents=True)
        source = self.snapshot_path(snapshot_id) / "state"
        for path in _regular_files(source):
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        return destination

    def verify_materialization(self, snapshot_id: str, directory: Path) -> dict:
        verification = self.verify(snapshot_id)
        errors = list(verification["errors"])
        try:
            manifest = json.loads(
                (self.snapshot_path(snapshot_id) / "SNAPSHOT.json").read_text(encoding="utf-8")
            )
            files, total = _file_inventory(Path(directory))
            if files != manifest.get("files"):
                errors.append("materialized file inventory or hashes differ from snapshot")
            if total != manifest.get("total_bytes"):
                errors.append("materialized byte count differs from snapshot")
        except (OSError, json.JSONDecodeError, SnapshotError) as error:
            errors.append(f"materialization unreadable: {error}")
        return {"valid": not errors, "snapshot_id": snapshot_id, "errors": errors}

    def promoted(self) -> dict | None:
        if not self.pointer_path.exists():
            return None
        try:
            pointer = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SnapshotError(f"promoted pointer unreadable: {error}") from error
        if pointer.get("format") != "gum-school-promoted-pointer-v1":
            raise SnapshotError("unsupported promoted pointer format")
        verification = self.verify(pointer.get("snapshot_id"))
        if not verification["valid"]:
            raise SnapshotError(f"promoted snapshot failed verification: {verification['errors']}")
        return pointer

    def promote(self, snapshot_id: str, *, expected_current: str | None, run_id: str) -> dict:
        verification = self.verify(snapshot_id)
        if not verification["valid"]:
            raise SnapshotError(f"refusing to promote invalid snapshot: {verification['errors']}")
        current = self.promoted()
        current_id = None if current is None else current["snapshot_id"]
        if current_id == snapshot_id:
            # A resumed transaction, or a passing no-change candidate, already
            # has the requested content installed.
            return current
        if current_id != expected_current:
            raise SnapshotError(
                f"promoted snapshot changed concurrently: expected {expected_current!r}, found {current_id!r}"
            )
        pointer = {
            "format": "gum-school-promoted-pointer-v1",
            "snapshot_id": snapshot_id,
            "previous_snapshot_id": current_id,
            "generation": 1 if current is None else int(current["generation"]) + 1,
            "promoted_at_utc": _utc_now(),
            "run_id": run_id,
        }
        atomic_write_json(self.pointer_path, pointer, sort_keys=True)
        return pointer
