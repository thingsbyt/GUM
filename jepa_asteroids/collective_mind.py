"""Provenance-aware knowledge exchange between independently learning agents.

The collective never copies numeric controls or whole private memories.  A
member may publish compact semantic claims supported by its own successful
experience.  Conflicting claims remain visible, independent agreement wins
over one agent's confidence, and recipients treat accepted claims as
hypotheses until they succeed with them in a fresh context.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

from .semantic_composer import SemanticSkillComposer


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class ExperiencePacket:
    """A small auditable claim, not a transplanted policy or action trace."""

    origin_agent: str
    subject: str
    assertion: str
    evidence_count: int
    successful_tasks: tuple[str, ...]
    kind: str = "visual-symbol-meaning"
    confidence: float = 0.0
    packet_id: str = field(default="")

    def __post_init__(self):
        if self.evidence_count < 1:
            raise ValueError("experience packet needs positive evidence")
        if not self.origin_agent or not self.subject or not self.assertion:
            raise ValueError("incomplete experience packet")
        if not self.packet_id:
            content = asdict(self); content.pop("packet_id", None)
            object.__setattr__(self, "packet_id", hashlib.sha256(_canonical(content)).hexdigest()[:24])

    def to_dict(self) -> dict:
        value = asdict(self); value["successful_tasks"] = list(self.successful_tasks)
        return value

    @classmethod
    def from_dict(cls, value: dict):
        row = dict(value); row["successful_tasks"] = tuple(row.get("successful_tasks", ()))
        return cls(**row)


def compile_packets(agent_id: str, mind: SemanticSkillComposer) -> list[ExperiencePacket]:
    """Distil grounded meanings while excluding local buttons and contexts."""
    payload = mind.export(); successes = payload.get("successful_compositions_data", [])
    task_hashes = tuple(sorted(hashlib.sha256(_canonical(row["tokens"])).hexdigest()[:16]
                               for row in successes if int(row.get("count", 0)) > 0))
    packets = []
    for subject, evidence in payload.get("token_evidence", {}).items():
        total = sum(int(count) for count in evidence.values())
        if not total:
            continue
        assertion, support = max(evidence.items(), key=lambda row: (int(row[1]), row[0]))
        runner_up = max((int(value) for key, value in evidence.items() if key != assertion), default=0)
        if int(support) < 2 or int(support) <= runner_up:
            continue
        packets.append(ExperiencePacket(
            origin_agent=str(agent_id), subject=str(subject), assertion=str(assertion),
            evidence_count=int(support), successful_tasks=task_hashes,
            confidence=float(int(support) / max(1, total))))
    return packets


class SharedExperienceLibrary:
    """Append-only collective memory with consensus, quarantine, and validation."""

    format = "wailah-shared-experience-library-v1"

    def __init__(self):
        self.packets: dict[str, ExperiencePacket] = {}
        self.validations = defaultdict(set)
        self.failed_validations = defaultdict(set)
        self.assignments = []

    def assign(self, agent: str, task: str) -> None:
        self.assignments.append({"agent": str(agent), "task": str(task), "status": "assigned"})

    def complete(self, agent: str, task: str) -> None:
        for row in reversed(self.assignments):
            if row["agent"] == str(agent) and row["task"] == str(task):
                row["status"] = "complete"; return
        raise KeyError((agent, task))

    def ingest(self, packet: ExperiencePacket) -> bool:
        content = packet.to_dict(); content["packet_id"] = ""
        expected = ExperiencePacket.from_dict(content).packet_id
        if packet.packet_id != expected:
            raise ValueError("packet provenance hash mismatch")
        existed = packet.packet_id in self.packets
        self.packets[packet.packet_id] = packet
        return not existed

    def _claims(self, subject: str):
        rows = defaultdict(list)
        for packet in self.packets.values():
            if packet.subject == subject and packet.kind == "visual-symbol-meaning":
                rows[packet.assertion].append(packet)
        return rows

    def decision(self, subject: str) -> dict | None:
        claims = self._claims(subject)
        if not claims:
            return None
        ranked = []
        for assertion, packets in claims.items():
            origins = {packet.origin_agent for packet in packets}
            evidence = sum(packet.evidence_count for packet in packets)
            validations = set().union(*(self.validations[packet.packet_id] for packet in packets))
            failures = set().union(*(self.failed_validations[packet.packet_id] for packet in packets))
            # Independent corroboration and outside validation dominate a
            # single source's arbitrarily large self-reported count.
            score = (len(validations), len(origins), evidence, -len(failures), assertion)
            ranked.append((score, assertion, packets, origins, validations, failures))
        ranked.sort(reverse=True); _, assertion, packets, origins, validations, failures = ranked[0]
        runner_origins = len(ranked[1][3]) if len(ranked) > 1 else 0
        accepted = len(origins) > runner_origins or len(validations) > 0 or len(ranked) == 1
        status = "validated" if validations else "corroborated" if len(origins) >= 2 else "provisional"
        losers = [packet.packet_id for row in ranked[1:] for packet in row[2]
                  if len(origins) >= 2 and len(origins) > len(row[3])]
        return {"subject": subject, "assertion": assertion, "accepted": accepted,
                "status": status, "origin_agents": sorted(origins),
                "packet_ids": sorted(packet.packet_id for packet in packets),
                "quarantined_packet_ids": sorted(losers),
                "evidence_count": sum(packet.evidence_count for packet in packets),
                "validated_by": sorted(validations), "failed_by": sorted(failures)}

    def accepted_claims(self) -> list[dict]:
        subjects = sorted({packet.subject for packet in self.packets.values()})
        return [row for subject in subjects if (row := self.decision(subject)) and row["accepted"]]

    def digest_into(self, mind: SemanticSkillComposer) -> list[str]:
        """Install accepted claims as revisable evidence, never local controls."""
        used = []
        for decision in self.accepted_claims():
            # Enough support to decode, but bounded so new direct experience
            # can overturn a collective hypothesis.
            support = min(3, max(2, len(decision["origin_agents"]) + 1))
            mind.token_evidence[decision["subject"]][decision["assertion"]] += support
            used.extend(decision["packet_ids"])
        return sorted(set(used))

    def validate(self, packet_ids: list[str], validator: str, success: bool) -> None:
        target = self.validations if success else self.failed_validations
        for packet_id in packet_ids:
            if packet_id not in self.packets:
                raise KeyError(packet_id)
            target[packet_id].add(str(validator))

    def status(self) -> dict:
        decisions = self.accepted_claims()
        quarantined = set(row_id for row in decisions for row_id in row["quarantined_packet_ids"])
        return {"format": self.format, "participants": sorted({p.origin_agent for p in self.packets.values()}),
                "packets": len(self.packets), "subjects": len({p.subject for p in self.packets.values()}),
                "accepted_claims": len(decisions), "corroborated_claims": sum(
                    row["status"] in ("corroborated", "validated") for row in decisions),
                "validated_claims": sum(row["status"] == "validated" for row in decisions),
                "quarantined_packets": len(quarantined),
                "task_assignments": len(self.assignments),
                "completed_assignments": sum(row["status"] == "complete" for row in self.assignments)}

    def export(self) -> dict:
        return self.status() | {"packets_data": [p.to_dict() for p in self.packets.values()],
            "validations": {key: sorted(value) for key, value in self.validations.items()},
            "failed_validations": {key: sorted(value) for key, value in self.failed_validations.items()},
            "assignments": list(self.assignments)}

    def save(self, path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.export(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format") != cls.format:
            raise ValueError("unsupported collective library")
        result = cls()
        for row in value.get("packets_data", []): result.ingest(ExperiencePacket.from_dict(row))
        for key, rows in value.get("validations", {}).items(): result.validations[key].update(rows)
        for key, rows in value.get("failed_validations", {}).items(): result.failed_validations[key].update(rows)
        result.assignments = list(value.get("assignments", [])); return result


def naive_digest(packets: list[ExperiencePacket], mind: SemanticSkillComposer) -> None:
    """Unsafe comparison: trust every source's claimed evidence count."""
    for packet in packets:
        mind.token_evidence[packet.subject][packet.assertion] += packet.evidence_count
