"""Crash-resistant local persistence helpers for GUM state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any


def _flush_file(path: Path, data: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _flush_directory(path: Path) -> None:
    """Persist directory-entry changes where the platform supports it."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes, *, backup: bool = True) -> None:
    """Atomically replace *path* and retain one recoverable previous version."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = path.with_name(f".{path.name}.{token}.tmp")
    backup_path = path.with_name(path.name + ".bak")
    backup_temporary = path.with_name(f".{path.name}.{token}.bak.tmp")
    try:
        _flush_file(temporary, data)
        if backup and path.exists():
            shutil.copyfile(path, backup_temporary)
            # Windows requires a write-capable descriptor for fsync.
            with backup_temporary.open("rb+") as handle:
                os.fsync(handle.fileno())
            os.replace(backup_temporary, backup_path)
        os.replace(temporary, path)
        _flush_directory(path.parent)
    finally:
        for leftover in (temporary, backup_temporary):
            try:
                leftover.unlink()
            except FileNotFoundError:
                pass


def atomic_write_text(path: Path, text: str, *, backup: bool = True) -> None:
    atomic_write_bytes(Path(path), text.encode("utf-8"), backup=backup)


def atomic_write_json(
    path: Path,
    value: Any,
    *,
    backup: bool = True,
    indent: int | None = 2,
    sort_keys: bool = False,
) -> None:
    atomic_write_text(
        Path(path),
        json.dumps(value, indent=indent, sort_keys=sort_keys, ensure_ascii=False),
        backup=backup,
    )
