"""Verify every file recorded by scripts/freeze_repo.py."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
binary_suffixes = {".gif", ".png", ".zip", ".bundle", ".pt", ".pth"}


def release_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    return data if path.suffix.lower() in binary_suffixes or b"\0" in data else data.replace(b"\r\n", b"\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("FROZEN_RELEASE.json"))
    args = parser.parse_args(argv)
    manifest_path = (root / args.manifest).resolve()
    try: manifest_path.relative_to(root)
    except ValueError: raise SystemExit("manifest must stay inside the repository")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file(): errors.append(f"missing: {name}"); continue
        actual = hashlib.sha256(release_bytes(path)).hexdigest()
        if actual != expected: errors.append(f"changed: {name}")
    print(json.dumps({"valid": not errors, "manifest": manifest_path.name,
                      "version": manifest.get("version"), "checked": len(manifest["files"]),
                      "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
