"""A conservative world router around the frozen adaptive visual learner.

Shared experience is attempted first.  If it produces an irreversible failed
episode in a context, that context gets an isolated learner branch.  This is a
generic negative-transfer safeguard: it uses only pixels, reward, and done.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from jepa_asteroids.adaptive_visual_agent import AdaptiveUniversalAgent, LearnedLayoutAdapter
from jepa_asteroids.universal_goal_state import perceive_state


class ResilientGeneralist:
    format = "gum-resilient-generalist-v1"

    def __init__(self):
        self.router = LearnedLayoutAdapter(); self.shared = AdaptiveUniversalAgent()
        self.branches = {}; self.failures = Counter(); self.active = self.shared
        self.context = None; self.branch_events = []; self.episodes = 0
        self.episode_open = False; self.episode_steps = 0; self.episode_reward = 0.0

    def _context(self, frame): return perceive_state(self.router.adapt(frame))["context"]

    def begin(self, frame):
        next_context = self._context(frame)
        if (self.episode_open and self.context == next_context and self.active is self.shared
                and self.episode_steps >= 250 and self.episode_reward <= 0
                and self.context not in self.branches):
            self.failures[self.context] += 1; self.branches[self.context] = AdaptiveUniversalAgent()
            self.branch_events.append({"context": self.context, "episode": self.episodes,
                "reason": "shared strategy made no rewarded progress for a full horizon"})
        self.context = next_context; self.active = self.branches.get(self.context, self.shared)
        self.episode_open = True; self.episode_steps = 0; self.episode_reward = 0.0
        self.episodes += 1; return self.active.begin(frame)

    def act(self, frame): return self.active.act(frame)

    def observe(self, before, action, after, reward, done):
        learned = self.active.observe(before, action, after, reward, done)
        self.episode_steps += 1; self.episode_reward += float(reward)
        if done and float(reward) < 0:
            self.failures[self.context] += 1
            if self.active is self.shared and self.context not in self.branches:
                self.branches[self.context] = AdaptiveUniversalAgent()
                self.branch_events.append({"context": self.context, "episode": self.episodes,
                    "reason": "transferred strategy caused irreversible negative terminal"})
        if done: self.episode_open = False
        return learned | {"strategy_branch": "isolated" if self.active is not self.shared else "shared",
                          "context_failures": self.failures[self.context]}

    def status(self):
        return {"format": self.format, "episodes": self.episodes, "contexts_with_failures": len(self.failures),
            "isolated_branches": len(self.branches), "branch_events": list(self.branch_events),
            "shared": self.shared.status(),
            "branches": {key: agent.status() for key, agent in self.branches.items()}}

    @staticmethod
    def _save_agent(agent, path): agent.save(path)

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        shared_path = path.with_name(path.stem + ".shared.json"); self._save_agent(self.shared, shared_path)
        branch_files = {}
        for index, (context, agent) in enumerate(sorted(self.branches.items())):
            branch = path.with_name(path.stem + f".branch-{index}.json"); self._save_agent(agent, branch)
            branch_files[context] = branch.name
        payload = {"format": self.format, "episodes": self.episodes, "failures": dict(self.failures),
            "branch_events": self.branch_events, "router": self.router.status(),
            "shared": shared_path.name, "branches": branch_files}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path); value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("format") != cls.format: raise ValueError("unsupported resilient generalist format")
        result = cls(); result.episodes = int(value["episodes"]); result.failures = Counter(value["failures"])
        result.branch_events = list(value["branch_events"]); result.router = LearnedLayoutAdapter.restore(value["router"])
        result.shared = AdaptiveUniversalAgent.load(path.with_name(value["shared"]))
        result.branches = {context: AdaptiveUniversalAgent.load(path.with_name(name))
                           for context, name in value["branches"].items()}
        result.active = result.shared; result.episode_open = False; return result
