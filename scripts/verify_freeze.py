"""Verify every file recorded by scripts/freeze_repo.py."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "FROZEN_RELEASE.json").read_text(encoding="utf-8"))
binary_suffixes = {".gif", ".png", ".zip", ".bundle", ".pt", ".pth"}


def release_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    return data if path.suffix.lower() in binary_suffixes or b"\0" in data else data.replace(b"\r\n", b"\n")


errors = []
for name, expected in manifest["files"].items():
    path = root / name
    if not path.is_file(): errors.append(f"missing: {name}"); continue
    actual = hashlib.sha256(release_bytes(path)).hexdigest()
    if actual != expected: errors.append(f"changed: {name}")
print(json.dumps({"valid": not errors, "checked": len(manifest["files"]), "errors": errors}, indent=2))
raise SystemExit(1 if errors else 0)
