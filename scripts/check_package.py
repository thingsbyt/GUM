"""Check local documentation links and Studio evidence declarations."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


root = Path(__file__).resolve().parents[1]
errors = []

for markdown in (root / "README.md", *sorted((root / "docs").glob("*.md"))):
    text = markdown.read_text(encoding="utf-8")
    for target in re.findall(r"!?\[[^]]*\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        clean = target.split("#", 1)[0].replace("%20", " ")
        if clean and not (markdown.parent / clean).resolve().is_file():
            errors.append(f"broken link in {markdown.relative_to(root)}: {target}")

index = json.loads((root / "EVIDENCE_INDEX.json").read_text(encoding="utf-8"))
for collection in ("replays", "documents"):
    for item in index.get(collection, []):
        if not (root / item["file"]).is_file():
            errors.append(f"missing {collection} file: {item['file']}")
for claim in index.get("claims", []):
    if not (root / claim["evidence"]).is_file():
        errors.append(f"missing claim evidence: {claim['evidence']}")

node = shutil.which("node")
if node:
    sys.path.insert(0, str(root))
    from gum.interface import HTML
    match = re.search(r"<script>(.*?)</script>", HTML, flags=re.DOTALL)
    if not match:
        errors.append("GUM Studio has no embedded script")
    else:
        temp_root = root / ".test-temp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            script = Path(folder) / "studio.js"
            script.write_text(match.group(1), encoding="utf-8")
            result = subprocess.run([node, "--check", str(script)], capture_output=True, text=True)
            if result.returncode:
                errors.append("GUM Studio JavaScript syntax: " + result.stderr.strip())

print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
raise SystemExit(1 if errors else 0)
