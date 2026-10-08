"""A real-file JSON rescue environment for the frozen Concept Genesis mind."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ACTION_COUNT = 5
FIELDS = ("name", "email", "age")
ALIASES = {"Name": "name", "full_name": "name", "e-mail": "email",
           "email_address": "email", "age_years": "age", "Age": "age"}
COLORS = np.asarray(((231, 91, 76), (69, 184, 238), (239, 188, 74),
                     (176, 112, 226), (73, 213, 151), (245, 133, 181)), dtype=np.uint8)


def create_messy_jsonl(path: Path, seed: int, records: int = 10) -> Path:
    """Create varied content with a stable family of independently verifiable defects."""
    rng = np.random.default_rng(seed); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    first = ["Ada", "Grace", "Alan", "Katherine", "Edsger", "Barbara", "Donald", "Margaret"]
    last = ["Lovelace", "Hopper", "Turing", "Johnson", "Dijkstra", "Liskov", "Knuth", "Hamilton"]
    order = rng.permutation(len(first)); rows = []
    for index in range(records):
        person = int(order[index % len(order)]); name = f"  {first[person]} {last[person]}  "
        email = f" {first[person]}.{last[person]}@Example.TEST "; age = str(28 + (person * 7) % 45)
        if index < 6:
            keys = (("Name", "e-mail", "age_years"), ("full_name", "email_address", "Age"))[index % 2]
        else:
            keys = ("name", "email", "age")
        rows.append({keys[0]: name, keys[1]: email, keys[2]: age})
    # Two content duplicates differ only in whitespace/case and therefore survive premature finalization.
    rows[-2] = dict(rows[0]); rows[-1] = dict(rows[1])
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows: handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip(): rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows: handle.write(json.dumps(row, sort_keys=True) + "\n")


class JsonDataRescueWorld:
    """Anonymous tools operate on a real working JSONL file; source is read-only."""

    def __init__(self, source: Path, work: Path, seed: int):
        self.source = Path(source).resolve(); self.work = Path(work).resolve(); self.seed = int(seed)
        self.work.mkdir(parents=True, exist_ok=True); self.active = self.work / "active.jsonl"
        self.original = _read_jsonl(self.source)
        if not self.original: raise ValueError("source JSONL is empty")
        rng = np.random.default_rng(seed); self.control_to_operation = rng.permutation(ACTION_COUNT).tolist()
        self.metric_positions = rng.permutation(6).tolist(); self.background = rng.integers(15, 38, 3, dtype=np.uint8)
        identity = hashlib.sha256(self.source.read_bytes() + str(seed).encode()).hexdigest()[:16]
        self.world_id = identity; self.steps = 0; self.reset()

    @property
    def public_spec(self):
        return {"world_id": self.world_id, "observation": "rgb-file-quality-view",
                "action_count": ACTION_COUNT, "horizon": ACTION_COUNT, "reward": "verified-terminal-scalar"}

    def reset(self):
        self.rows = copy.deepcopy(self.original); self.steps = 0; self.history = []; _write_jsonl(self.active, self.rows)
        return self.render()

    @staticmethod
    def _canonicalize(rows):
        for row in rows:
            for old, new in list(ALIASES.items()):
                if old in row and new not in row: row[new] = row.pop(old)

    @staticmethod
    def _trim(rows):
        for row in rows:
            for key, value in list(row.items()):
                if isinstance(value, str): row[key] = value.strip()

    @staticmethod
    def _coerce(rows):
        for row in rows:
            value = row.get("age")
            if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()): row["age"] = int(value)

    @staticmethod
    def _normalize_email(rows):
        for row in rows:
            value = row.get("email")
            if isinstance(value, str): row["email"] = value.strip().lower()

    @staticmethod
    def _finalize(rows):
        unique = {}
        for row in rows:
            key = row.get("email")
            if isinstance(key, str) and key == key.strip().lower(): unique.setdefault(key, row)
            else: unique[f"unresolved-{len(unique)}"] = row
        result = sorted(unique.values(), key=lambda row: str(row.get("email", "")))
        for index, row in enumerate(result, 1): row["record_id"] = f"R{index:04d}"
        rows[:] = result

    def _issues(self):
        alias = sum(sum(key in ALIASES for key in row) for row in self.rows)
        whitespace = sum(sum(isinstance(value, str) and value != value.strip() for value in row.values()) for row in self.rows)
        non_integer_age = sum(not isinstance(row.get("age"), int) for row in self.rows)
        email_format = sum(not isinstance(row.get("email"), str) or row.get("email") != row.get("email", "").strip().lower()
                           for row in self.rows)
        normalized = [row.get("email") for row in self.rows if isinstance(row.get("email"), str)]
        duplicates = max(0, len(normalized) - len(set(normalized)))
        missing_ids = sum("record_id" not in row for row in self.rows)
        return [alias, whitespace, non_integer_age, email_format, duplicates, missing_ids]

    def verify(self):
        issues = self._issues(); fields = all(set(FIELDS) <= set(row) for row in self.rows)
        ids = [row.get("record_id") for row in self.rows]
        sorted_emails = [row.get("email") for row in self.rows]
        return bool(not any(issues) and fields and len(ids) == len(set(ids)) and sorted_emails == sorted(sorted_emails))

    def render(self):
        frame = np.zeros((48, 48, 3), dtype=np.uint8); frame[:] = self.background
        issues = self._issues()
        for metric, count in enumerate(issues):
            slot = self.metric_positions[metric]; row, col = divmod(slot, 3)
            y0, x0 = 4 + row * 21, 4 + col * 15
            area = min(60, int(count) * 6)
            for offset in range(area):
                y = y0 + offset // 10; x = x0 + offset % 10; frame[y, x] = COLORS[metric]
        return np.moveaxis(frame, -1, 0)

    def step(self, action: int):
        action = int(action); operation = int(self.control_to_operation[action])
        (self._canonicalize, self._trim, self._coerce, self._normalize_email, self._finalize)[operation](self.rows)
        self.history.append(operation); self.steps += 1; _write_jsonl(self.active, self.rows)
        done = self.steps >= ACTION_COUNT; success = bool(done and self.verify())
        return self.render(), (1.0 if success else 0.0), done, {"success": success}

    def audit(self):
        return {"world_id": self.world_id, "source": str(self.source), "work": str(self.work),
                "control_to_operation": list(self.control_to_operation), "operation_history": list(self.history),
                "verified": self.verify(), "issues": self._issues(), "active_file": str(self.active)}


def export_csv(jsonl: Path, output: Path):
    rows = _read_jsonl(jsonl); output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("record_id", "name", "email", "age"))
        writer.writeheader(); writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    return output
