"""Create a deterministic SHA-256 manifest for the repository contents."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
excluded_dirs = {".git", ".venv", ".gum-workspace", ".pytest_cache", ".scratch",
                 ".test-temp", "__pycache__", "htmlcov", "local-results",
                 "model_cache", "runs", "work"}
binary_suffixes = {".gif", ".png", ".zip", ".bundle", ".pt", ".pth"}


def release_bytes(path: Path) -> bytes:
    """Hash text with Git's declared LF form so verification is cross-platform."""
    data = path.read_bytes()
    return data if path.suffix.lower() in binary_suffixes or b"\0" in data else data.replace(b"\r\n", b"\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, default=Path("RELEASE_MANIFEST.json"))
    args = parser.parse_args(argv)
    output = (root / args.output).resolve()
    try: output.relative_to(root)
    except ValueError: raise SystemExit("manifest output must stay inside the repository")

    files = {}
    for path in sorted(root.rglob("*")):
        if (not path.is_file() or excluded_dirs.intersection(path.parts)
                or any(part.endswith(".egg-info") for part in path.parts)
                or any(part.endswith("-results") for part in path.parts)
                or path.name == ".coverage" or path.suffix in {".tmp", ".log"}
                or path.resolve() == output):
            continue
        files[path.relative_to(root).as_posix()] = hashlib.sha256(release_bytes(path)).hexdigest()
    value = {"format": "gum-frozen-release-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "name": "GUM — Growing Understanding Machine", "version": str(args.version),
             "files": files, "file_count": len(files)}
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Frozen {len(files)} files in {output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

