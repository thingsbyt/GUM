"""Two resilient visual minds that exchange learned experience, not world state."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from .resilient_generalist import ResilientGeneralist


class ExperienceCollective:
    format = "gum-experience-collective-v1"

    def __init__(self):
        self.members = [ResilientGeneralist(), ResilientGeneralist()]
        self.world_library = {}; self.role_library = {}; self.exchanges = 0
        self.published_worlds = [set(), set()]

    @staticmethod
    def _cores(member):
        return [member.shared.core] + [branch.core for branch in member.branches.values()]

    def publish(self, member_index: int):
        member_index = int(member_index); member = self.members[member_index]
        for core in self._cores(member):
            for context, memory in core.worlds.items():
                self.world_library[context] = deepcopy(memory); self.published_worlds[member_index].add(context)
            for form, roles in core.form_roles.items():
                current = self.role_library.setdefault(form, {})
                for role, count in roles.items(): current[role] = max(int(count), int(current.get(role, 0)))
        self.exchanges += 1; self.synchronize()

    def synchronize(self):
        for member in self.members:
            core = member.shared.core
            for context, memory in self.world_library.items(): core.worlds[context] = deepcopy(memory)
            for form, roles in self.role_library.items():
                for role, count in roles.items(): core.form_roles[form][role] = max(core.form_roles[form][role], count)

    def status(self):
        overlap = self.published_worlds[0] & self.published_worlds[1]
        return {"format": self.format, "members": 2, "exchanges": self.exchanges,
            "library_worlds": len(self.world_library), "library_visual_forms": len(self.role_library),
            "published_worlds_by_member": [len(row) for row in self.published_worlds],
            "worlds_experienced_by_both": len(overlap),
            "member_status": [member.status() for member in self.members]}

    def save(self, folder: Path):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        for index, member in enumerate(self.members): member.save(folder / f"member-{index}.json")
        value = {"format": self.format, "exchanges": self.exchanges,
            "published_worlds": [sorted(row) for row in self.published_worlds]}
        (folder / "collective.json").write_text(json.dumps(value, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path):
        folder = Path(folder); value = json.loads((folder / "collective.json").read_text(encoding="utf-8"))
        if value.get("format") != cls.format: raise ValueError("unsupported collective format")
        result = cls(); result.members = [ResilientGeneralist.load(folder / f"member-{i}.json") for i in range(2)]
        result.exchanges = int(value["exchanges"])
        result.published_worlds = [set(row) for row in value["published_worlds"]]
        for index in range(2):
            member = result.members[index]
            for core in result._cores(member):
                for context, memory in core.worlds.items(): result.world_library[context] = deepcopy(memory)
                for form, roles in core.form_roles.items():
                    current = result.role_library.setdefault(form, {})
                    for role, count in roles.items(): current[role] = max(int(count), int(current.get(role, 0)))
        result.synchronize(); return result
