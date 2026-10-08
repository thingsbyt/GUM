"""Append-only, hash-chained evolutionary and experience tracking."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

from .storage import atomic_write_json


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HashLedger:
    format = "gum-hash-ledger-v1"

    def __init__(self, path: Path, *, anchor_path: Path | None = None):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.anchor_path = (Path(anchor_path) if anchor_path is not None
                            else self.path.with_name(self.path.name + ".anchor.json"))
        self._head, self._records = self._read_state()
        self._startup_errors: list[str] = []
        if self.anchor_path.exists():
            try:
                anchor = json.loads(self.anchor_path.read_text(encoding="utf-8"))
                if anchor.get("format") != "gum-ledger-anchor-v1":
                    self._startup_errors.append("unsupported ledger anchor")
                if int(anchor.get("records", -1)) != self._records:
                    self._startup_errors.append("record count differs from startup anchor")
                if anchor.get("head") != self._head:
                    self._startup_errors.append("ledger head differs from startup anchor")
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
                self._startup_errors.append(f"startup anchor unreadable: {error}")

    def _read_state(self):
        if not self.path.exists(): return "0" * 64, 0
        last = None; count = 0
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip(): last = json.loads(line); count += 1
        return ("0" * 64 if last is None else last["record_hash"]), count

    def _last_hash(self): return self._head

    def append(self, event: str, payload: dict, *, run_id: str | None = None) -> dict:
        if self._startup_errors:
            raise RuntimeError("refusing to extend a ledger that disagrees with its anchor: "
                               + "; ".join(self._startup_errors))
        record = {"format": self.format, "event_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": str(event),
            "run_id": run_id, "previous_hash": self._last_hash(), "payload": payload}
        record["record_hash"] = hashlib.sha256(canonical(record)).hexdigest()
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        self._head = record["record_hash"]; self._records += 1
        try:
            self.write_anchor()
        except Exception as error:
            self._startup_errors.append(f"anchor update failed: {error}")
            raise
        return record

    def write_anchor(self, path: Path | None = None) -> dict:
        """Write an independently checkable expected head/count checkpoint.

        A copied or externally published anchor detects suffix deletion while
        it remains trusted.  It is not a signature and cannot stop an attacker
        from rewriting both the ledger and every local anchor.
        """
        target = self.anchor_path if path is None else Path(path)
        value = {"format": "gum-ledger-anchor-v1", "ledger": self.path.name,
                 "records": self._records, "head": self._head}
        atomic_write_json(target, value, sort_keys=True)
        return value

    def verify(self, *, expected_head: str | None = None,
               expected_records: int | None = None,
               anchor_path: Path | None = None) -> dict:
        previous = "0" * 64; count = 0; errors = list(self._startup_errors)
        if self.path.exists():
            try:
                with self.path.open(encoding="utf-8") as handle:
                    for number, line in enumerate(handle, 1):
                        if not line.strip(): continue
                        row = json.loads(line); claimed = row.pop("record_hash", None)
                        actual = hashlib.sha256(canonical(row)).hexdigest()
                        if row.get("previous_hash") != previous: errors.append(f"line {number}: broken parent link")
                        if claimed != actual: errors.append(f"line {number}: invalid record hash")
                        previous = claimed or ""; count += 1
            except (OSError, json.JSONDecodeError, TypeError) as error:
                errors.append(f"ledger unreadable: {error}")

        selected_anchor = self.anchor_path if anchor_path is None else Path(anchor_path)
        anchor = None
        if selected_anchor.exists():
            try:
                anchor = json.loads(selected_anchor.read_text(encoding="utf-8"))
                if anchor.get("format") != "gum-ledger-anchor-v1": errors.append("unsupported ledger anchor")
                if int(anchor.get("records", -1)) != count: errors.append("record count differs from anchor")
                if anchor.get("head") != previous: errors.append("ledger head differs from anchor")
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
                errors.append(f"anchor unreadable: {error}")
        if expected_records is not None and int(expected_records) != count:
            errors.append("record count differs from expected value")
        if expected_head is not None and str(expected_head) != previous:
            errors.append("ledger head differs from expected value")
        return {"valid": not errors, "records": count, "head": previous,
                "anchored": anchor is not None, "anchor": anchor, "errors": errors}


class TransitionTrace(HashLedger):
    format = "gum-transition-trace-v1"
