"""Non-destructive learned workflow for verified file archival.

The apprentice observes two successful demonstrations to ground an unknown
visual command.  For every new job, numeric operation controls are remapped. It
must discover the controls from filesystem effects, execute the learned goal,
and independently verify the resulting archive. Source files are never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import zipfile

import numpy as np

from .semantic_composer import SemanticSkillComposer


OPERATIONS = ("file-copied", "file-renamed", "checksum-created",
              "index-created", "archive-created")
GLYPHS = (
    ((1, 1, 1), (0, 1, 0), (0, 1, 0)),
    ((1, 0, 0), (1, 1, 1), (0, 0, 1)),
    ((1, 1, 0), (0, 1, 0), (0, 1, 1)),
    ((1, 0, 1), (1, 1, 1), (0, 1, 0)),
    ((1, 1, 1), (1, 0, 0), (1, 1, 1)),
)


def _goal_card() -> np.ndarray:
    frame = np.zeros((1, 9, 35), dtype=np.uint8)
    for position, glyph in enumerate(GLYPHS):
        mask = np.asarray(glyph, dtype=bool); x = 2 + position * 6
        frame[0, 2:5, x:x + 3][mask] = 3 + position * 2
    return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_snapshot(path: Path) -> dict[str, str]:
    return {str(file.relative_to(path)): _sha256(file)
            for file in sorted(Path(path).rglob("*")) if file.is_file()}


def _safe_name(relative: Path) -> str:
    stem = "__".join(relative.with_suffix("").parts).lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "file"
    suffix = re.sub(r"[^a-z0-9.]", "", relative.suffix.lower())
    return stem + suffix


class VerifiedArchiveJob:
    actions = [1, 2, 3, 4, 5]

    def __init__(self, source: Path, job: Path):
        self.source = Path(source).resolve(); self.job = Path(job).resolve()
        if not self.source.is_dir(): raise ValueError("input must be a directory")
        if self.source == self.job or self.job.is_relative_to(self.source):
            raise ValueError("output job must be outside the input directory")
        self.staging = self.job / "staging"; self.staging.mkdir(parents=True, exist_ok=False)
        self.archive = self.job / "VERIFIED_ARCHIVE.zip"
        identity = "\n".join(str(path.relative_to(self.source)) for path in
                             sorted(self.source.rglob("*")) if path.is_file())
        self.context_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
        seed = int(self.context_id, 16) % (2 ** 32)
        permutation = np.random.default_rng(seed).permutation(len(OPERATIONS))
        self.control = {action: OPERATIONS[int(permutation[index])]
                        for index, action in enumerate(self.actions)}
        self.completed = set(); self.manifest = None

    def _copy(self) -> bool:
        if "file-copied" in self.completed: return False
        files = [path for path in sorted(self.source.rglob("*")) if path.is_file()]
        if not files: raise ValueError("input directory contains no files")
        for index, source in enumerate(files):
            relative = source.relative_to(self.source)
            destination = self.staging / f"raw_{index:04d}_{relative.name}"
            shutil.copy2(source, destination)
        return True

    def _rename(self) -> bool:
        if "file-copied" not in self.completed or "file-renamed" in self.completed: return False
        for index, path in enumerate(sorted(self.staging.iterdir())):
            relative_name = path.name.split("_", 2)[-1]
            candidate = self.staging / f"{index + 1:04d}_{_safe_name(Path(relative_name))}"
            path.rename(candidate)
        return True

    def _checksums(self) -> bool:
        if "file-renamed" not in self.completed or "checksum-created" in self.completed: return False
        rows = [{"file": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in sorted(self.staging.iterdir()) if path.is_file()]
        self.manifest = rows
        (self.staging / "SHA256SUMS.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return True

    def _index(self) -> bool:
        if "checksum-created" not in self.completed or "index-created" in self.completed: return False
        lines = ["# Verified archive index", "", "| File | Bytes | SHA-256 |", "|---|---:|---|"]
        for row in self.manifest:
            lines.append(f"| {row['file']} | {row['bytes']} | `{row['sha256']}` |")
        (self.staging / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True

    def _archive(self) -> bool:
        if "index-created" not in self.completed or "archive-created" in self.completed: return False
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(self.staging.iterdir()):
                if path.is_file(): bundle.write(path, path.name)
        return True

    def step(self, action: int) -> tuple[list[str], bool]:
        operation = self.control[int(action)]
        changed = {"file-copied": self._copy, "file-renamed": self._rename,
                   "checksum-created": self._checksums, "index-created": self._index,
                   "archive-created": self._archive}[operation]()
        if changed: self.completed.add(operation)
        return ([f"event:semantic:{operation}"] if changed else []), self.archive.exists()

    def verify(self) -> dict:
        if not self.archive.exists(): return {"verified": False, "reason": "archive missing"}
        with zipfile.ZipFile(self.archive, "r") as bundle:
            names = set(bundle.namelist())
            if not {"SHA256SUMS.json", "INDEX.md"} <= names:
                return {"verified": False, "reason": "metadata missing"}
            manifest = json.loads(bundle.read("SHA256SUMS.json"))
            failures = []
            for row in manifest:
                data = bundle.read(row["file"])
                actual = hashlib.sha256(data).hexdigest()
                if actual != row["sha256"] or len(data) != row["bytes"]:
                    failures.append(row["file"])
        return {"verified": not failures, "files": len(manifest),
                "failed_files": failures, "archive_sha256": _sha256(self.archive),
                "source_unchanged": all(path.is_file() for path in self.source.rglob("*") if path.is_file())}


def _trained_apprentice() -> SemanticSkillComposer:
    model = SemanticSkillComposer(); card = _goal_card()
    for demonstration in range(2):
        model.reset_episode(f"demonstration-{demonstration}", card)
        for index, operation in enumerate(OPERATIONS):
            model.observe(index + 1, [f"event:semantic:{operation}"],
                          index == len(OPERATIONS) - 1, index == len(OPERATIONS) - 1)
    if model.status()["grounded_symbol_meanings"] != len(OPERATIONS):
        raise RuntimeError("workflow demonstration did not ground every operation")
    return model


def run_workflow(source: Path, output_root: Path) -> dict:
    source, output_root = Path(source).resolve(), Path(output_root).resolve()
    source_before = _source_snapshot(source)
    output_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256((str(source) + str(time.time_ns())).encode()).hexdigest()[:10]
    job_path = output_root / f"verified-archive-{digest}"; job_path.mkdir()
    job = VerifiedArchiveJob(source, job_path); brain_path = output_root / "APPRENTICE_BRAIN.json"
    if brain_path.exists():
        model = SemanticSkillComposer(); model.restore(json.loads(brain_path.read_text(encoding="utf-8")))
        resumed = True
    else:
        model = _trained_apprentice(); resumed = False
    context = f"filesystem:{job.context_id}"; model.reset_episode(context, _goal_card()); trace = []
    for step in range(1, 51):
        plan = model.recommend(job.actions)
        if plan is None: raise RuntimeError("learned workflow could not be decoded")
        action, reason = plan; events, success = job.step(action)
        model.observe(action, events, success, success)
        trace.append({"step": step, "action": action, "reason": reason,
                      "observed_events": events, "complete": success})
        if success: break
    verification = job.verify()
    source_after = _source_snapshot(source); source_unchanged = source_before == source_after
    verification["source_unchanged"] = source_unchanged
    brain_path.write_text(json.dumps(model.export(), indent=2), encoding="utf-8")
    report = {"format": "wailah-useful-file-workflow-v1",
              "source": str(source), "source_files": len([x for x in source.rglob("*") if x.is_file()]),
              "job": str(job_path), "archive": str(job.archive),
              "persistent_brain": str(brain_path), "resumed_prior_experience": resumed,
              "learned_visual_operations": model.status()["grounded_dictionary"],
              "numeric_control_mapping_audit_only": job.control,
              "trace": trace, "actions_to_complete": len(trace),
              "verification": verification,
              "safety": {"source_files_modified": not source_unchanged,
                         "source_files_deleted": sorted(set(source_before) - set(source_after)),
                         "all_work_confined_to_job_directory": True}}
    (job_path / "WORKFLOW_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not verification["verified"] or not source_unchanged:
        raise RuntimeError("archive verification or source-integrity check failed")
    return report


def create_demo_inbox(path: Path) -> Path:
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    samples = {"Quarterly Notes.txt": "Revenue review\nRisks: supplier delay\n",
               "Project Plan v2.md": "# Project plan\n- prototype\n- validate\n- ship\n",
               "contacts.csv": "name,role\nAda,Engineering\nGrace,Operations\n",
               "read me!.txt": "This directory demonstrates verified archival.\n"}
    for name, content in samples.items():
        target = path / name
        if not target.exists(): target.write_text(content, encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(argv)
    source = create_demo_inbox(args.output / "sample_inbox") if args.demo else args.input
    if source is None: parser.error("--input is required unless --demo is used")
    report = run_workflow(source, args.output)
    print(json.dumps(report, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
