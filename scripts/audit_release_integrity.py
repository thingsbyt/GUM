"""Audit public-package leakage and the strongest learning-evidence gates."""
from __future__ import annotations

import json
from pathlib import Path
import re
import zipfile


root = Path(__file__).resolve().parents[1]
excluded_dirs = {".git", ".test-temp", ".venv", "__pycache__"}
patterns = {
    "windows_home_path": re.compile(rb"C:" + rb"\\\\Users\\\\|C:/" + rb"Users/"),
    "aws_access_key": re.compile((rb"AKIA" + rb"[0-9A-Z]{16}")),
    "github_token": re.compile((rb"gh" + rb"[pousr]_[A-Za-z0-9_]{20,}")),
    "openai_key": re.compile((rb"s" + rb"k-[A-Za-z0-9]{20,}")),
    "private_key": re.compile(rb"BEGIN (RSA|OPENSSH|EC) PRIVATE KEY"),
}


def scan(name: str, data: bytes, hits: list[dict]) -> None:
    for kind, pattern in patterns.items():
        if pattern.search(data):
            hits.append({"kind": kind, "file": name})


leak_hits: list[dict] = []
for path in root.rglob("*"):
    if (not path.is_file() or excluded_dirs.intersection(path.parts)
            or any(part.endswith(".egg-info") for part in path.parts)):
        continue
    relative = path.relative_to(root).as_posix()
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                scan(f"{relative}!{member}", archive.read(member), leak_hits)
    else:
        scan(relative, path.read_bytes(), leak_hits)


def load(relative: str) -> dict:
    return json.loads((root / relative).read_text(encoding="utf-8"))


growth = load("evidence/growth/GROWTH_CHALLENGE_AUDIT.json")
genesis = load("evidence/concept-genesis/CONCEPT_GENESIS_AUDIT.json")
rescue = load("evidence/data-rescue/DATA_RESCUE_AUDIT.json")
meta = load("evidence/learning-to-learn/LEARNING_TO_LEARN_AUDIT.json")
language = load("evidence/clarification-test/CLARIFICATION_AUDIT.json")

gates = {
    "growth_passed": growth["passed"],
    "growth_hidden_state_absent": growth["checks"]["hidden_state_absent_from_learner_traces"],
    "growth_source_frozen": growth["checks"]["source_frozen"],
    "genesis_passed": genesis["passed"],
    "genesis_hidden_state_absent": genesis["checks"]["hidden_state_absent_from_learner_traces"],
    "genesis_source_frozen": genesis["checks"]["source_frozen"],
    "data_rescue_passed": rescue["passed"],
    "data_rescue_hidden_state_absent": rescue["checks"]["hidden_state_absent_from_learner_traces"],
    "data_rescue_sources_unchanged": rescue["checks"]["sources_unchanged"],
    "meta_learning_passed": meta["passed"],
    "meta_learning_zero_solutions_retained": meta["checks"]["zero_solutions_retained"],
    "clarification_passed": language["passed"],
    "clarification_no_preinstalled_dictionary": not language["protocol"]["preinstalled_word_dictionary"],
    "clarification_no_language_model": language["protocol"]["language_model_used"] == "none",
}
report = {
    "format": "gum-public-release-integrity-v1",
    "publication_leak_scan": {"passed": not leak_hits, "hits": leak_hits},
    "learning_evidence_gates": {
        "source": "saved release evidence; experiments are not rerun by this audit",
        "saved_evidence_only": True,
        "gates": gates,
    },
    "all_learning_evidence_gates_passed": all(gates.values()),
    "interpretation": (
        "This command validates stored gate fields and scans the package; it does not rerun experiments. "
        "The saved gates support experience-dependent learning in the documented bounded tasks. "
        "They do not prove that no bug or indirect leakage exists, nor do they establish general intelligence."
    ),
}
report["passed"] = report["publication_leak_scan"]["passed"] and report["all_learning_evidence_gates_passed"]
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 1)
