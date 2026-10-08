"""Build a compact, auditable GUM research release from recorded artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


SOURCE_DIRS = ("gum", "jepa_asteroids", "tests", "configs")
SOURCE_FILES = (
    "README.md", "GUM.md", "requirements.txt", "NOTICE.txt", "SOURCES.md",
    "AUTONOMOUS_LEARNING.md", "LIFELONG_AGENT.md", "RUN_WINDOWS.bat",
)

EVIDENCE_TREES = (
    "gum_v1",
    "gum_reveal_gate_v2",
    "gum_reveal_gate_v3",
    "problem_solving_proof",
    "useful_tasks_10_final",
    "two_rocket_v20",
    "cooperative_immune_v19",
    "grounded_dialogue_v23",
)

EVIDENCE_FILES = (
    "WAILAH_PROBLEM_SOLVING_AND_UTILITY_RESULT.md",
    "WAILAH_TEN_USEFUL_TASKS_RESULT.md",
    "TWO_ROCKET_ASTEROIDS_V20_RESULT.md",
    "WAILAH_COOPERATIVE_IMMUNE_V19_RESULT.md",
    "WAILAH_INTELLIGENCE_AUDIT.md",
    "WAILAH_Living_Evidence.md",
    "WAILAH_Lifelong_Proof.json",
)

CONCEPT_FILES = (
    "CONCEPT_BRIDGE_AUDIT.json", "RESULTS.md", "CONFUSION.csv",
    "APPLE_CONCEPT_REPLAY.gif", "UNSEEN_APPLE_SAMPLE.png",
)

REPLAY_SOURCES = {
    "two_rocket_asteroids.gif": "two_rocket_v20/TWO_ROCKET_V20_REPLAY.gif",
    "cooperative_immune_game.gif": "cooperative_immune_v19/COOPERATIVE_IMMUNE_V19_REPLAY.gif",
    "apple_concept.gif": "concept_bridge_v22/APPLE_CONCEPT_REPLAY.gif",
    "logic_chain.gif": "WAILAH_Logic_Chain.gif",
    "maze_escape.gif": "WAILAH_Maze_Escape.gif",
    "lifelong_showcase.gif": "WAILAH_Lifelong_Showcase.gif",
    "json_normalizer.gif": "useful_tasks_10_final/runs/json_normalizer/first/ACTIONS.gif",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def _copy_if_present(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def _curated_claims(outputs: Path) -> list[dict]:
    dual = _read_json(outputs / "gum_reveal_gate_v3" / "DUAL_REVEAL_GATE_REPORT.json")
    problem = _read_json(outputs / "problem_solving_proof" / "WAILAH_SEALED_PROBLEM_SOLVING_AUDIT.json")
    utility = _read_json(outputs / "useful_tasks_10_final" / "UTILITY_SUITE_AUDIT.json")
    asteroids = _read_json(outputs / "two_rocket_v20" / "TWO_ROCKET_V20_FINAL_FROZEN_AUDIT.json")
    immune = _read_json(outputs / "cooperative_immune_v19" / "COOPERATIVE_IMMUNE_V19_FROZEN_AUDIT.json")
    concept = _read_json(outputs / "concept_bridge_v22" / "CONCEPT_BRIDGE_AUDIT.json")
    language = _read_json(outputs / "grounded_dialogue_v23" / "GROUNDED_DIALOGUE_AUDIT.json")
    first = problem["aggregate"]["semantic_composer"]
    revisit = problem["aggregate"]["revisit_after_all_tasks"]
    return [
        {
            "id": "dual-gate",
            "status": "passed-internal-sealed-test",
            "title": "Persistent single and collective learning",
            "headline": "20/20 held-out worlds solved safely by both conditions",
            "metrics": [
                f"Single revisit: {dual['aggregate']['single_revisit_speedup']:.2f}x faster",
                f"Cross-owner transfer: {dual['aggregate']['collective_cross_owner_speedup_over_fresh']:.2f}x faster than fresh",
                f"Random control: {dual['aggregate']['random']['successes']}/20",
            ],
            "scope": "Two related visual-composition world families; internally designed and evaluated.",
            "evidence": "evidence/gum_reveal_gate_v3/DUAL_REVEAL_GATE_REPORT.json",
        },
        {
            "id": "problem-solving",
            "status": "passed-internal-sealed-test",
            "title": "Compositional problem solving",
            "headline": f"{first['successes']}/{first['tasks']} held-out problems solved",
            "metrics": [
                f"First encounter: {first['mean_steps']:.2f} actions",
                f"Revisit: {revisit['mean_steps']:.2f} actions",
                "Corrupted meanings: 0/256",
            ],
            "scope": "A structured four-stage event world, with causal controls and shuffled actions.",
            "evidence": "evidence/problem_solving_proof/WAILAH_SEALED_PROBLEM_SOLVING_AUDIT.json",
        },
        {
            "id": "useful-work",
            "status": "passed-internal-functional-test",
            "title": "Useful file workflows",
            "headline": "10/10 workflows verified on first encounter and revisit",
            "metrics": [
                f"Mean first run: {utility['first_mean_actions']:.1f} actions",
                f"Mean revisit: {utility['revisit_mean_actions']:.1f} actions",
                f"Recorded demonstrations: {utility['videos_created']}",
            ],
            "scope": "The domain tools were provided; GUM learned shuffled control effects and execution order.",
            "evidence": "evidence/useful_tasks_10_final/UTILITY_SUITE_AUDIT.json",
        },
        {
            "id": "asteroids",
            "status": "passed-internal-sealed-test",
            "title": "Two-body visual coordination",
            "headline": f"{asteroids['aggregate']['coordinated']['total_hits']} hits across 8 held-out worlds",
            "metrics": [
                "Both rockets contributed in 8/8 worlds",
                "Zero coordinated friendly fire",
                "All 16 shuffled controllers grounded",
            ],
            "scope": "Engineered visual tracking and safety machinery; control meanings learned online.",
            "evidence": "evidence/two_rocket_v20/TWO_ROCKET_V20_FINAL_FROZEN_AUDIT.json",
        },
        {
            "id": "cooperation",
            "status": "passed-internal-sealed-test",
            "title": "Inferred cooperation in a fictional game",
            "headline": "12/12 successes when communication was available; 0/12 without it",
            "metrics": [
                f"Mean coordinated rounds: {immune['aggregate']['communicating_pair']['mean_joint_rounds']:.1f}",
                "Zero protected-cell damage",
                "Every pair switched from independent to coordinated mode",
            ],
            "scope": "A designed fictional environment, not a biomedical model or medical result.",
            "evidence": "evidence/cooperative_immune_v19/COOPERATIVE_IMMUNE_V19_FROZEN_AUDIT.json",
        },
        {
            "id": "language",
            "status": "passed-bounded-test",
            "title": "Grounded command interface",
            "headline": f"{language['results']['correct_and_executed']}/{language['results']['trials']} novel bounded commands",
            "metrics": [
                "3/3 ambiguous commands triggered clarification",
                f"Learned lexicon: {len(language['learned_lexicon'])} words",
                "No pretrained language model used",
            ],
            "scope": "A bounded grammar and four engineered action kinds, not open-ended language understanding.",
            "evidence": "evidence/grounded_dialogue_v23/GROUNDED_DIALOGUE_AUDIT.json",
        },
        {
            "id": "concept-bridge",
            "status": "mixed-result",
            "title": "Learned visual-symbol bridge",
            "headline": f"Apple: {concept['sealed']['apple']['correct']}/{concept['sealed']['apple']['trials']} ({concept['sealed']['apple']['accuracy']:.0%})",
            "metrics": [
                f"All concepts, photo to glyph: {concept['sealed']['photo_to_glyph_accuracy']:.2%}",
                f"All concepts, composed photo to word: {concept['sealed']['photo_to_word_accuracy']:.2%}",
                "Overall precommitted suite: failed",
            ],
            "scope": "Closed-set ten-class recognition. Several classes failed badly; that failure is retained.",
            "evidence": "evidence/concept_bridge_v22/CONCEPT_BRIDGE_AUDIT.json",
        },
    ]


def _build_index(outputs: Path, release: Path) -> dict:
    replays = []
    titles = {
        "two_rocket_asteroids.gif": "Two-rocket Asteroids",
        "cooperative_immune_game.gif": "Two-agent fictional rescue game",
        "apple_concept.gif": "Apple concept bridge",
        "logic_chain.gif": "Delayed logic chain",
        "maze_escape.gif": "Maze escape",
        "lifelong_showcase.gif": "Lifelong learning showcase",
        "json_normalizer.gif": "Useful JSON workflow",
    }
    for name in REPLAY_SOURCES:
        path = release / "media" / name
        if path.exists():
            replays.append({"id": path.stem, "title": titles[name], "file": f"media/{name}",
                            "sha256": _sha256(path)})
    docs = [
        ("start", "Start here", "START_HERE.md"),
        ("summary", "Executive summary", "docs/EXECUTIVE_SUMMARY.md"),
        ("paper", "Preliminary paper", "docs/PRELIMINARY_PAPER.md"),
        ("evidence", "Evidence catalog", "docs/EVIDENCE_CATALOG.md"),
        ("methods", "Methods and architecture", "docs/METHODS_AND_ARCHITECTURE.md"),
        ("reproduce", "Reproduction guide", "docs/REPRODUCIBILITY.md"),
        ("history", "Project history", "docs/PROJECT_HISTORY.md"),
        ("claims", "Claims and limitations", "docs/CLAIMS_AND_LIMITATIONS.md"),
        ("studio", "Studio guide", "docs/STUDIO_GUIDE.md"),
    ]
    return {
        "format": "gum-evidence-index-v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "GUM — Growing Understanding Machine",
        "release_type": "preliminary internal research package",
        "claims": _curated_claims(outputs),
        "replays": replays,
        "documents": [{"id": key, "title": title, "file": file} for key, title, file in docs],
        "global_warning": (
            "These are internally produced benchmark results. GUM is a modular collection of engineered "
            "learners behind one auditable harness, not evidence of consciousness or general intelligence."
        ),
    }


def _manifest(release: Path) -> dict:
    rows = []
    for path in sorted(p for p in release.rglob("*") if p.is_file() and p.name != "PACKAGE_MANIFEST.json"):
        rows.append({"path": path.relative_to(release).as_posix(), "bytes": path.stat().st_size,
                     "sha256": _sha256(path)})
    return {
        "format": "gum-package-manifest-v1",
        "files": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "entries": rows,
    }


def build(source: Path, outputs: Path, release: Path, *, zip_release: bool = True) -> Path:
    source = source.resolve(); outputs = outputs.resolve(); release = release.resolve()
    if release.exists():
        shutil.rmtree(release)
    release.mkdir(parents=True)

    docs_source = source / "release_docs"
    _copy_if_present(docs_source / "START_HERE.md", release / "START_HERE.md")
    if (docs_source / "docs").exists():
        _copy_tree(docs_source / "docs", release / "docs")
    for name in ("START_GUM_STUDIO.ps1", "START_GUM_STUDIO.bat"):
        _copy_if_present(docs_source / name, release / name)

    for name in SOURCE_DIRS:
        folder = source / name
        if folder.exists():
            _copy_tree(folder, release / "source" / name)
    for name in SOURCE_FILES:
        _copy_if_present(source / name, release / "source" / name)

    for name in EVIDENCE_TREES:
        folder = outputs / name
        if folder.exists():
            _copy_tree(folder, release / "evidence" / name)
    for name in EVIDENCE_FILES:
        _copy_if_present(outputs / name, release / "evidence" / name)
    for name in CONCEPT_FILES:
        _copy_if_present(outputs / "concept_bridge_v22" / name,
                         release / "evidence" / "concept_bridge_v22" / name)
    for target_name, relative in REPLAY_SOURCES.items():
        _copy_if_present(outputs / Path(relative), release / "media" / target_name)

    index = _build_index(outputs, release)
    (release / "EVIDENCE_INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    manifest = _manifest(release)
    (release / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if zip_release:
        archive = release.with_suffix(".zip")
        if archive.exists():
            archive.unlink()
        shutil.make_archive(str(release), "zip", root_dir=release.parent, base_dir=release.name)
    return release


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the GUM preliminary research package")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args(argv)
    result = build(args.source, args.outputs, args.release, zip_release=not args.no_zip)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
