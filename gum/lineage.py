"""Append-only, hash-chained evolutionary and experience tracking."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HashLedger:
    format = "gum-hash-ledger-v1"

    def __init__(self, path: Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self._head = self._read_last_hash()

    def _read_last_hash(self):
        if not self.path.exists(): return "0" * 64
        last = None
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip(): last = json.loads(line)
        return "0" * 64 if last is None else last["record_hash"]

    def _last_hash(self): return self._head

    def append(self, event: str, payload: dict, *, run_id: str | None = None) -> dict:
        record = {"format": self.format, "event_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": str(event),
            "run_id": run_id, "previous_hash": self._last_hash(), "payload": payload}
        record["record_hash"] = hashlib.sha256(canonical(record)).hexdigest()
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._head = record["record_hash"]
        return record

    def verify(self) -> dict:
        previous = "0" * 64; count = 0; errors = []
        if not self.path.exists(): return {"valid": True, "records": 0, "head": previous, "errors": []}
        with self.path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip(): continue
                row = json.loads(line); claimed = row.pop("record_hash", None)
                actual = hashlib.sha256(canonical(row)).hexdigest()
                if row.get("previous_hash") != previous: errors.append(f"line {number}: broken parent link")
                if claimed != actual: errors.append(f"line {number}: invalid record hash")
                previous = claimed or ""; count += 1
        return {"valid": not errors, "records": count, "head": previous, "errors": errors}


class TransitionTrace(HashLedger):
    format = "gum-transition-trace-v1"
