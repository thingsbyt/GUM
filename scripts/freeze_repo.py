"""Create a deterministic SHA-256 manifest for the repository contents."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
excluded = {"FROZEN_RELEASE.json"}
excluded_dirs = {".git", ".pytest_cache", ".scratch", ".test-temp", "__pycache__"}
binary_suffixes = {".gif", ".png", ".zip", ".bundle", ".pt", ".pth"}


def release_bytes(path: Path) -> bytes:
    """Hash text with Git's declared LF form so verification is cross-platform."""
    data = path.read_bytes()
    return data if path.suffix.lower() in binary_suffixes or b"\0" in data else data.replace(b"\r\n", b"\n")


files = {}
for path in sorted(root.rglob("*")):
    if not path.is_file() or excluded_dirs.intersection(path.parts) or path.name in excluded:
        continue
    files[path.relative_to(root).as_posix()] = hashlib.sha256(release_bytes(path)).hexdigest()
value = {"format": "gum-frozen-release-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
         "name": "GUM — Growing Understanding Machine", "version": "0.2.0-public-preview",
         "files": files, "file_count": len(files)}
(root / "FROZEN_RELEASE.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Frozen {len(files)} files")

