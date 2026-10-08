"""Run the precommitted frozen collective-learning replication."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .collective_learning_benchmark import run_collective_audit


FROZEN = {
    "collective_mind.py": "E3B00668ED8A9857BD243897C82483FF057188643CD51DC28B0FD30145CC6F2A",
    "collective_learning_benchmark.py": "F055A17C4949C119A3476CDC794491A3B6C1846E0614A41AF3DB9A8FDCA2C4C9",
    "semantic_composer.py": "C4F2694BC05B3DD5F36BF2A7DFAEB757871320E30F1D807D4EC9D831B6F3E05B",
    "semantic_world_model.py": "B36F93744E38AACB16A6AA458161C028C4B8A191E38E0F3C842208153B70EE86",
}
LANGUAGE_SEED = 24_440_057
HELDOUT_SEED = 97_220_069


def _hashes():
    folder = Path(__file__).parent
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest().upper() for name in FROZEN}


def run_frozen_audit(output, tasks=120, horizon=42):
    before = _hashes()
    if before != FROZEN:
        raise RuntimeError(f"frozen collective hash mismatch: {before}")
    report = run_collective_audit(output, tasks=tasks, horizon=horizon,
                                  language_seed=LANGUAGE_SEED, heldout_seed=HELDOUT_SEED)
    after = _hashes(); hashes_match = before == after == FROZEN
    report["format"] = "wailah-frozen-collective-v18-transfer-audit-v1"
    report["classification"] = "precommitted frozen multi-agent semantic knowledge integration"
    report["frozen_implementation"] = {"expected": FROZEN, "before": before,
                                       "after": after, "unchanged": hashes_match}
    report["precommitted_pass"] = bool(report["precommitted_pass"] and hashes_match)
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv); report = run_frozen_audit(args.output)
    print(json.dumps({"aggregate": {name: {key: value for key, value in row.items() if key != "rows"}
        for name, row in report["aggregate"].items()}, "effect": report["effect"],
        "library": report["exchange"]["status_after_target"],
        "frozen_unchanged": report["frozen_implementation"]["unchanged"],
        "pass": report["precommitted_pass"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
