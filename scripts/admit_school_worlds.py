"""Run and save the deterministic GUM School foundational-world admission audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gum.school.admission import run_foundational_admission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence" / "gum-school" / "admissions" / "foundational-v1.json",
    )
    parser.add_argument("--trials", type=int, default=24)
    arguments = parser.parse_args(argv)
    report = run_foundational_admission(arguments.output, trials=arguments.trials)
    digest = hashlib.sha256(arguments.output.read_bytes()).hexdigest()
    print(json.dumps({
        "passed": report["passed"],
        "adapters": len(report["adapters"]),
        "evidence": str(arguments.output),
        "sha256": digest,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
