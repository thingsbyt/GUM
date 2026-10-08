"""Ten safe practical workflows driven by the persistent semantic apprentice."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import zipfile

import numpy as np

from .semantic_composer import SemanticSkillComposer


STAGES = ("source-scanned", "data-transformed", "result-validated",
          "report-created", "bundle-created")
GLYPHS = (
    ((1, 1, 1), (0, 1, 0), (0, 1, 0)),
    ((1, 0, 0), (1, 1, 1), (0, 0, 1)),
    ((1, 1, 0), (0, 1, 0), (0, 1, 1)),
    ((1, 0, 1), (1, 1, 1), (0, 1, 0)),
    ((1, 1, 1), (1, 0, 0), (1, 1, 1)),
)

TASKS = {
    "csv_cleaner": "Clean, deduplicate, and sort a CSV table",
    "json_normalizer": "Normalize JSON while preserving its data",
    "log_triage": "Extract and summarize warnings and errors",
    "duplicate_finder": "Find byte-identical duplicate files",
    "markdown_catalog": "Catalog Markdown titles and word counts",
    "file_organizer": "Copy mixed files into type-based folders",
    "verified_backup": "Create a checksum-verified backup",
    "search_index": "Build a local inverted text-search index",
    "sensitive_audit": "Locate likely email, phone, and key exposures",
    "dataset_profiler": "Profile columns, missing values, and numeric ranges",
}


def _goal_card() -> np.ndarray:
    frame = np.zeros((1, 9, 35), dtype=np.uint8)
    for position, glyph in enumerate(GLYPHS):
        mask = np.asarray(glyph, dtype=bool); x = 2 + position * 6
        frame[0, 2:5, x:x + 3][mask] = 3 + position * 2
    return frame


def _hash_bytes(value: bytes) -> str: return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str: return _hash_bytes(Path(path).read_bytes())


def _snapshot(path: Path) -> dict[str, str]:
    return {str(file.relative_to(path)): _hash_file(file)
            for file in sorted(Path(path).rglob("*")) if file.is_file()}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8")


def _prepare_inputs(root: Path) -> dict[str, Path]:
    roots = {name: root / name / "input" for name in TASKS}
    for path in roots.values(): path.mkdir(parents=True, exist_ok=True)
    _write(roots["csv_cleaner"] / "people.csv",
           "id,name,city\n3, Grace Hopper ,Arlington\n1, Ada Lovelace, London \n"
           "2,Alan Turing,Manchester\n2,Alan Turing,Manchester\n4,,Boston\n")
    _write(roots["json_normalizer"] / "settings.json",
           '{"z":3,"service":{"enabled":true,"ports":[8080,8081]},"a":"demo"}')
    _write(roots["log_triage"] / "application.log",
           "2026-10-06 INFO start\n2026-10-06 WARN disk 82%\n2026-10-06 ERROR timeout api\n"
           "2026-10-06 INFO retry\n2026-10-06 ERROR timeout api\n2026-10-06 WARN cache miss\n")
    _write(roots["duplicate_finder"] / "alpha.txt", "identical payload\n")
    _write(roots["duplicate_finder"] / "alpha-copy.txt", "identical payload\n")
    _write(roots["duplicate_finder"] / "unique.txt", "different payload\n")
    _write(roots["markdown_catalog"] / "guide.md", "# User Guide\n\nInstall, configure, and verify.\n")
    _write(roots["markdown_catalog"] / "notes.md", "# Release Notes\n\nFixed two issues and added indexing.\n")
    _write(roots["markdown_catalog"] / "nested" / "api.md", "# API Reference\n\nEndpoints and examples.\n")
    _write(roots["file_organizer"] / "proposal.doc.txt", "proposal\n")
    _write(roots["file_organizer"] / "metrics.csv", "day,value\n1,10\n2,14\n")
    _write(roots["file_organizer"] / "config.json", '{"enabled":true}')
    (roots["file_organizer"] / "pixel.png").write_bytes(b"\x89PNG\r\n\x1a\nDEMO")
    _write(roots["verified_backup"] / "project.txt", "important project state\n")
    _write(roots["verified_backup"] / "config.ini", "mode=safe\nversion=2\n")
    _write(roots["search_index"] / "science.txt", "causal learning builds useful models")
    _write(roots["search_index"] / "engineering.txt", "useful systems require careful verification")
    _write(roots["search_index"] / "memory.txt", "learning and memory support adaptation")
    _write(roots["sensitive_audit"] / "sample.txt",
           "Demo contact: alex@example.test, phone 555-010-2233.\n"
           "Fake key for scanner testing: api_key=DEMO-1234567890-SECRET\n")
    _write(roots["sensitive_audit"] / "clean.txt", "No sensitive-looking content here.\n")
    _write(roots["dataset_profiler"] / "measurements.csv",
           "sample,temperature,pressure,status\nA,21.5,101.2,ok\nB,22.1,,ok\n"
           "C,20.8,100.9,review\nD,,101.4,ok\n")
    return roots


class PracticalTask:
    actions = [1, 2, 3, 4, 5]

    def __init__(self, name: str, source: Path, work: Path):
        self.name = name; self.source = Path(source); self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=False); self.completed = []
        order = np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)).permutation(5)
        self.control = {action: STAGES[int(order[index])] for index, action in enumerate(self.actions)}
        self.scan_rows = []; self.metrics = {}; self.verified = False

    def scan(self) -> bool:
        if self.completed: return False
        self.scan_rows = [{"file": str(path.relative_to(self.source)), "bytes": path.stat().st_size,
                           "sha256": _hash_file(path)}
                          for path in sorted(self.source.rglob("*")) if path.is_file()]
        (self.work / "SCAN.json").write_text(json.dumps(self.scan_rows, indent=2), encoding="utf-8")
        return bool(self.scan_rows)

    def transform(self) -> bool:
        if self.completed != [STAGES[0]]: return False
        getattr(self, f"_transform_{self.name}")(); return True

    def validate(self) -> bool:
        if self.completed != list(STAGES[:2]): return False
        self.verified = bool(getattr(self, f"_validate_{self.name}")())
        (self.work / "VALIDATION.json").write_text(json.dumps(
            {"verified": self.verified, "metrics": self.metrics}, indent=2), encoding="utf-8")
        return self.verified

    def report(self) -> bool:
        if self.completed != list(STAGES[:3]) or not self.verified: return False
        lines = [f"# {self.name.replace('_', ' ').title()}", "", TASKS[self.name], "",
                 "## Result", "", "- Verification: **passed**",
                 f"- Input files: **{len(self.scan_rows)}**", "", "## Metrics", ""]
        lines.extend(f"- {key.replace('_', ' ').title()}: **{value}**"
                     for key, value in sorted(self.metrics.items()))
        (self.work / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True

    def package(self) -> bool:
        if self.completed != list(STAGES[:4]): return False
        bundle = self.work / "RESULTS.zip"
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.work.rglob("*")):
                if path.is_file() and path != bundle: archive.write(path, path.relative_to(self.work))
        self.metrics["bundle_sha256"] = _hash_file(bundle); return True

    def step(self, action: int) -> tuple[list[str], bool]:
        stage = self.control[int(action)]
        expected = STAGES[len(self.completed)] if len(self.completed) < len(STAGES) else None
        if stage != expected: return [], False
        changed = {STAGES[0]: self.scan, STAGES[1]: self.transform, STAGES[2]: self.validate,
                   STAGES[3]: self.report, STAGES[4]: self.package}[stage]()
        if changed: self.completed.append(stage)
        done = len(self.completed) == len(STAGES)
        return ([f"event:semantic:{stage}"] if changed else []), done

    # --- Concrete useful transforms and their independent checks. ---
    def _transform_csv_cleaner(self):
        path = next(self.source.glob("*.csv"))
        with path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
        fields = list(rows[0]); cleaned = [{k: (v or "").strip() for k, v in row.items()} for row in rows]
        unique = {tuple(row[k] for k in fields): row for row in cleaned}
        result = sorted(unique.values(), key=lambda row: tuple(row[k] for k in fields))
        with (self.work / "cleaned.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(result)
        self.metrics.update(input_rows=len(rows), output_rows=len(result), duplicates_removed=len(rows)-len(result))
    def _validate_csv_cleaner(self):
        with (self.work / "cleaned.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return len(rows) == self.metrics["output_rows"] and all(v == v.strip() for row in rows for v in row.values())

    def _transform_json_normalizer(self):
        source = next(self.source.glob("*.json")); value = json.loads(source.read_text())
        (self.work / "normalized.json").write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        self.metrics.update(top_level_keys=len(value), normalized_bytes=(self.work / "normalized.json").stat().st_size)
    def _validate_json_normalizer(self):
        return json.loads(next(self.source.glob("*.json")).read_text()) == json.loads((self.work / "normalized.json").read_text())

    def _transform_log_triage(self):
        issues = []
        for line_number, line in enumerate(next(self.source.glob("*.log")).read_text().splitlines(), 1):
            match = re.search(r"\b(WARN|ERROR)\b\s+(.*)$", line)
            if match: issues.append({"line": line_number, "severity": match.group(1), "message": match.group(2)})
        with (self.work / "issues.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("line", "severity", "message")); writer.writeheader(); writer.writerows(issues)
        counts = Counter(row["severity"] for row in issues)
        self.metrics.update(issues=len(issues), errors=counts["ERROR"], warnings=counts["WARN"])
    def _validate_log_triage(self):
        with (self.work / "issues.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return len(rows) == self.metrics["issues"] and all(row["severity"] in ("WARN", "ERROR") for row in rows)

    def _transform_duplicate_finder(self):
        groups = defaultdict(list)
        for path in self.source.iterdir():
            if path.is_file(): groups[_hash_file(path)].append(path.name)
        duplicates = [{"sha256": key, "files": sorted(value)} for key, value in groups.items() if len(value) > 1]
        (self.work / "duplicates.json").write_text(json.dumps(duplicates, indent=2), encoding="utf-8")
        self.metrics.update(duplicate_groups=len(duplicates), duplicate_files=sum(len(x["files"]) for x in duplicates))
    def _validate_duplicate_finder(self):
        groups = json.loads((self.work / "duplicates.json").read_text())
        return bool(groups) and all(len({_hash_file(self.source / name) for name in row["files"]}) == 1 for row in groups)

    def _transform_markdown_catalog(self):
        rows = []
        for path in sorted(self.source.rglob("*.md")):
            text = path.read_text(); title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
            rows.append({"file": str(path.relative_to(self.source)), "title": title,
                         "words": len(re.findall(r"\b\w+\b", text))})
        (self.work / "catalog.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        lines = ["# Markdown catalog", ""] + [f"- [{r['title']}](../input/{r['file']}) — {r['words']} words" for r in rows]
        (self.work / "CATALOG.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        self.metrics.update(documents=len(rows), total_words=sum(x["words"] for x in rows))
    def _validate_markdown_catalog(self):
        return len(json.loads((self.work / "catalog.json").read_text())) == len(list(self.source.rglob("*.md")))

    def _transform_file_organizer(self):
        categories = {".txt": "documents", ".md": "documents", ".csv": "data", ".json": "data",
                      ".png": "images", ".jpg": "images", ".jpeg": "images"}
        manifest = []
        for path in sorted(self.source.rglob("*")):
            if not path.is_file(): continue
            category = categories.get(path.suffix.lower(), "other"); destination = self.work / "organized" / category / path.name
            destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
            manifest.append({"source": path.name, "category": category, "sha256": _hash_file(path)})
        (self.work / "organization.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.metrics.update(files_organized=len(manifest), categories=len({x["category"] for x in manifest}))
    def _validate_file_organizer(self):
        rows = json.loads((self.work / "organization.json").read_text())
        return all(_hash_file(self.work / "organized" / row["category"] / row["source"]) == row["sha256"] for row in rows)

    def _transform_verified_backup(self):
        backup = self.work / "backup"; backup.mkdir()
        rows = []
        for path in sorted(self.source.rglob("*")):
            if not path.is_file(): continue
            destination = backup / path.relative_to(self.source); destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination); rows.append({"file": str(path.relative_to(self.source)), "sha256": _hash_file(path)})
        (self.work / "backup_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        self.metrics.update(files_backed_up=len(rows), bytes_backed_up=sum(x.stat().st_size for x in backup.rglob("*") if x.is_file()))
    def _validate_verified_backup(self):
        rows = json.loads((self.work / "backup_manifest.json").read_text())
        return all(_hash_file(self.work / "backup" / row["file"]) == row["sha256"] for row in rows)

    def _transform_search_index(self):
        index = defaultdict(lambda: defaultdict(int))
        for path in sorted(self.source.rglob("*.txt")):
            for word in re.findall(r"[a-z0-9]+", path.read_text().lower()): index[word][path.name] += 1
        serial = {word: dict(files) for word, files in sorted(index.items())}
        (self.work / "search_index.json").write_text(json.dumps(serial, indent=2), encoding="utf-8")
        self.metrics.update(indexed_words=len(serial), indexed_documents=len(list(self.source.rglob("*.txt"))))
    def _validate_search_index(self):
        value = json.loads((self.work / "search_index.json").read_text())
        return "learning" in value and "useful" in value and self.metrics["indexed_documents"] == 3

    def _transform_sensitive_audit(self):
        patterns = {"email": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
                    "phone": re.compile(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b"),
                    "api_key": re.compile(r"(?i)api[_-]?key\s*=\s*\S+")}
        findings = []
        for path in sorted(self.source.rglob("*.txt")):
            text = path.read_text()
            for kind, pattern in patterns.items():
                for match in pattern.finditer(text):
                    value = match.group(0); redacted = value[:3] + "…" + value[-3:]
                    findings.append({"file": path.name, "kind": kind, "redacted": redacted})
        (self.work / "findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
        counts = Counter(x["kind"] for x in findings)
        self.metrics.update(findings=len(findings), emails=counts["email"], phones=counts["phone"], api_keys=counts["api_key"])
    def _validate_sensitive_audit(self):
        findings = json.loads((self.work / "findings.json").read_text())
        return len(findings) == 3 and all("example.test" not in row["redacted"] for row in findings)

    def _transform_dataset_profiler(self):
        path = next(self.source.glob("*.csv"))
        with path.open(encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
        profile = {"rows": len(rows), "columns": {}}
        for field in rows[0]:
            values = [(row[field] or "").strip() for row in rows]; present = [v for v in values if v]
            numeric = []
            for value in present:
                try: numeric.append(float(value))
                except ValueError: pass
            column = {"missing": len(values)-len(present), "unique": len(set(present))}
            if len(numeric) == len(present) and numeric:
                column.update(min=min(numeric), max=max(numeric), mean=statistics.fmean(numeric))
            profile["columns"][field] = column
        (self.work / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
        self.metrics.update(rows=len(rows), columns=len(rows[0]), total_missing=sum(x["missing"] for x in profile["columns"].values()))
    def _validate_dataset_profiler(self):
        value = json.loads((self.work / "profile.json").read_text())
        return value["rows"] == 4 and value["columns"]["temperature"]["missing"] == 1


def _trained_model() -> SemanticSkillComposer:
    model = SemanticSkillComposer(); card = _goal_card()
    for repetition in range(2):
        model.reset_episode(f"utility-demonstration-{repetition}", card)
        for index, stage in enumerate(STAGES):
            model.observe(index + 1, [f"event:semantic:{stage}"], index == 4, index == 4)
    return model


def _record_video(task: str, trace: list[dict], output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    frames = []; width, height = 900, 470
    try:
        title_font = ImageFont.truetype("arial.ttf", 22)
        font = ImageFont.truetype("arial.ttf", 17)
        small = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        title_font = font = small = ImageFont.load_default()
    for row in trace:
        image = Image.new("RGB", (width, height), (7, 13, 25)); draw = ImageDraw.Draw(image)
        green, white, muted, orange = (95, 224, 167), (232, 239, 248), (138, 153, 176), (255, 178, 88)
        draw.text((28, 22), "LEARNED UTILITY WORKFLOW", font=title_font, fill=green)
        draw.text((28, 60), TASKS[task], font=font, fill=white)
        draw.text((28, 104), f"Step {row['step']}   unknown control: {row['action']}", font=font, fill=orange)
        draw.text((28, 136), f"Decision: {row['reason']}", font=small, fill=muted)
        event = row["events"][0].replace("event:semantic:", "") if row["events"] else "no effect / keep testing"
        draw.text((28, 170), f"Observed result: {event}", font=font, fill=white)
        draw.text((28, 218), "Goal progress", font=font, fill=muted)
        for index, stage in enumerate(STAGES):
            complete = index < row["completed"]
            y = 254 + index * 36
            draw.rectangle((30, y + 3, 49, y + 22), fill=green if complete else (48, 60, 80))
            draw.text((63, y), stage, font=font, fill=white if complete else muted)
        if row["done"]: draw.text((480, 300), "VERIFIED COMPLETE", font=title_font, fill=green)
        frames.append(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=650, loop=0,
                   optimize=False, disposal=2)


def _run_once(model: SemanticSkillComposer, name: str, source: Path, output: Path,
              run_name: str) -> dict:
    before = _snapshot(source); job_path = output / name / run_name
    task = PracticalTask(name, source, job_path); context = f"utility:{name}"
    model.reset_episode(context, _goal_card()); trace = []
    for step in range(1, 40):
        action, reason = model.recommend(task.actions); events, done = task.step(action)
        model.observe(action, events, done, done)
        trace.append({"step": step, "action": action, "reason": reason, "events": events,
                      "completed": len(task.completed), "done": done})
        if done: break
    after = _snapshot(source); video = job_path / "ACTIONS.gif"; _record_video(name, trace, video)
    audit = {"task": name, "description": TASKS[name], "verified": task.verified and bool(done),
             "actions": len(trace), "source_unchanged": before == after,
             "control_mapping_audit_only": task.control, "trace": trace,
             "metrics": task.metrics, "job": str(job_path), "video": str(video),
             "bundle": str(job_path / "RESULTS.zip")}
    (job_path / "AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def run_suite(output: Path) -> dict:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    sources = _prepare_inputs(output / "tasks"); model = _trained_model()
    first = [_run_once(model, name, sources[name], output / "runs", "first") for name in TASKS]
    revisit = [_run_once(model, name, sources[name], output / "runs", "revisit") for name in TASKS]
    brain = output / "UTILITY_APPRENTICE_BRAIN.json"
    brain.write_text(json.dumps(model.export(), indent=2), encoding="utf-8")
    report = {"format": "wailah-ten-useful-workflows-v1", "tasks": len(TASKS),
              "all_first_runs_verified": all(x["verified"] and x["source_unchanged"] for x in first),
              "all_revisits_verified": all(x["verified"] and x["source_unchanged"] for x in revisit),
              "first_mean_actions": float(np.mean([x["actions"] for x in first])),
              "revisit_mean_actions": float(np.mean([x["actions"] for x in revisit])),
              "optimal_revisits": sum(x["actions"] == len(STAGES) for x in revisit),
              "videos_created": sum(Path(x["video"]).exists() for x in first + revisit),
              "first_runs": first, "revisits": revisit, "persistent_brain": str(brain)}
    (output / "UTILITY_SUITE_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv); report = run_suite(args.output)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("first_runs", "revisits")}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
