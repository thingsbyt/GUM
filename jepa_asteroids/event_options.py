"""Reward-independent discovery and composition of visual event options.

An option is a short, visually guarded action fragment that reliably produces
an observed event.  Events are learned from pixels (objects appearing,
disappearing, or changing relations), not supplied task labels.  The planner
builds a graph of these fragments and can execute several options in sequence
before any external reward has identified the final goal.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import math


def _salient(events: list[str]) -> list[str]:
    unique = list(dict.fromkeys(events))
    # Object creation/removal is a compact, persistent milestone. Relation
    # changes are useful only when no object event explains the transition;
    # otherwise one movement can manufacture dozens of redundant "skills".
    objects = [event for event in unique
               if event.startswith(("event:appeared:", "event:disappeared:"))]
    if objects:
        return objects[:4]
    relations = [event for event in unique
                 if event.startswith(("event:relation-created:",
                                      "event:relation-destroyed:"))]
    return relations[:2]


class EventOptionPlanner:
    """Mine reusable event-producing skills and plan over their outcomes."""

    format = "wailah-event-options-v1"

    def __init__(self):
        self.options: dict[str, dict] = {}
        self.by_entry = defaultdict(set)
        self.event_edges = defaultdict(Counter)
        self.event_actions = defaultdict(Counter)
        self.state_action_attempts = defaultdict(Counter)
        self.event_achievements = Counter()
        self.event_progress = Counter()
        self.known_states = set()
        self.events_discovered = 0
        self.options_discovered = 0
        self.option_plans = 0
        self.option_steps = 0
        self.option_completions = 0
        self.option_aborts = 0
        self.multi_option_plans = 0
        self.frontier_probes = 0
        self.observations = 0
        self.coherence_streak = 0
        self.reset_episode()

    def reset_episode(self) -> None:
        self.segment = []
        self.previous_event = None
        self.episode_events = []
        self.episode_event_attempts = Counter()
        self.active_steps = []
        self.active_index = 0
        self.active_option_keys = []
        self.last_plan = None

    @staticmethod
    def _normal_data(data) -> dict | None:
        if not data:
            return None
        if "x" not in data or "y" not in data:
            return None
        return {"x": int(data["x"]), "y": int(data["y"])}

    @classmethod
    def _key(cls, entry: str, event: str, result: str) -> str:
        # Multiple exploratory prefixes can reach the same visual milestone.
        # They compete inside one option and the shortest observed trace wins.
        payload = {"entry": entry, "event": event, "result": result}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:24]

    def _record_option(self, event: str, steps: list[dict], novel_state: bool) -> str | None:
        if not steps or len(steps) > 32:
            return None
        entry, result = steps[0]["state"], steps[-1]["next_state"]
        key = self._key(entry, event, result)
        row = self.options.get(key)
        if row is None:
            row = {"entry_state": entry, "result_state": result,
                   "target_event": event, "steps": [dict(step) for step in steps],
                   "occurrences": 0, "replays": 0, "completions": 0,
                   "failures": 0, "novel_state_discoveries": 0}
            self.options[key] = row; self.by_entry[entry].add(key)
            self.options_discovered += 1
        elif len(steps) < len(row["steps"]):
            row["steps"] = [dict(step) for step in steps]
        row["occurrences"] += 1
        row["novel_state_discoveries"] += int(novel_state)
        return key

    @staticmethod
    def _reliability(row: dict) -> float:
        return ((row.get("occurrences", 0) + row.get("completions", 0) + 1.0) /
                (row.get("occurrences", 0) + row.get("completions", 0) +
                 row.get("failures", 0) + 2.0))

    def _event_score(self, row: dict) -> float:
        event = row["target_event"]
        progress = 10.0 * self.event_progress[event]
        unlock = 2.5 * row.get("novel_state_discoveries", 0)
        downstream = 1.5 * len(self.event_edges.get(event, {}))
        frontier = 4.0 if not self.by_entry.get(row["result_state"]) else 0.0
        rarity = 3.0 / math.sqrt(1 + self.event_achievements[event])
        uncertainty = 2.0 / math.sqrt(1 + row.get("replays", 0))
        failure = 5.0 * row.get("failures", 0) / max(1, row.get("replays", 0))
        tried = 8.0 * self.episode_event_attempts[event]
        return progress + unlock + downstream + frontier + rarity + uncertainty - failure - tried

    def _trusted(self) -> bool:
        if self.observations < 64 or self.coherence_streak < 32:
            return False
        forward = [(key, row) for key, row in self.options.items()
                   if row.get("novel_state_discoveries", 0) > 0]
        if len(forward) < 2:
            return False
        entries = {row["entry_state"] for _, row in forward}
        return any(row["result_state"] in entries for _, row in forward)

    def _plan(self, start: str, actions: list[int]) -> tuple[list[dict], list[str]] | None:
        """Search the learned option graph for the most useful event frontier."""
        queue = deque([(start, [])]); best_path_for_state = {start: 0}; candidates = []
        while queue and len(best_path_for_state) <= 512:
            state, path = queue.popleft()
            if len(path) >= 4:
                continue
            for key in sorted(self.by_entry.get(state, ())):
                row = self.options[key]
                if any(int(step["action"]) not in actions for step in row["steps"]):
                    continue
                if self._reliability(row) < .34:
                    continue
                if (row["target_event"].startswith("event:disappeared:") and
                        row.get("novel_state_discoveries", 0) == 0 and
                        self.event_progress[row["target_event"]] <= 0):
                    continue
                proposal = path + [key]
                visited_states = {start}
                for prior_key in path:
                    visited_states.add(self.options[prior_key]["result_state"])
                # Before reward says otherwise, an option that returns to a
                # state already on the plan is regression, not a frontier.
                # This rejects attractive-looking reset loops while retaining
                # rewarded cycles for genuinely repeatable tasks.
                if (row["result_state"] in visited_states and
                        self.event_progress[row["target_event"]] <= 0):
                    continue
                score = self._event_score(row) - .18 * sum(
                    len(self.options[item]["steps"]) for item in proposal)
                candidates.append((score, proposal))
                nxt = row["result_state"]
                if len(proposal) < best_path_for_state.get(nxt, 99):
                    best_path_for_state[nxt] = len(proposal); queue.append((nxt, proposal))
        if not candidates:
            return None
        score, keys = max(candidates, key=lambda item: (item[0], -len(item[1]), item[1]))
        if score <= -2.0:
            return None
        steps = []
        for key in keys:
            for step in self.options[key]["steps"]:
                enriched = dict(step); enriched["option_key"] = key
                enriched["target_event"] = self.options[key]["target_event"]
                steps.append(enriched)
        return steps, keys

    def recommend(self, state: str, actions: list[int]):
        actions = sorted(map(int, actions))
        if not self._trusted():
            return None
        if self.active_index < len(self.active_steps):
            step = self.active_steps[self.active_index]
            if step["state"] == state and int(step["action"]) in actions:
                self.option_steps += 1
                return int(step["action"]), self._normal_data(step.get("data")), "execute-learned-event-option"
            key = step.get("option_key")
            if key in self.options:
                self.options[key]["failures"] += 1
            self.option_aborts += 1; self.active_steps = []; self.active_index = 0
            self.active_option_keys = []
        planned = self._plan(state, actions)
        if planned is None:
            forward_results = {row["result_state"] for row in self.options.values()
                               if (row.get("novel_state_discoveries", 0) > 0 or
                                   self.event_progress[row["target_event"]] > 0)}
            if state not in forward_results:
                return None
            # The end of a learned option chain is an experimental frontier.
            # Probe its least-tested legal control using the position-invariant
            # concept state, so evidence accumulates across shifted layouts.
            action = min(actions, key=lambda candidate: (
                self.state_action_attempts[state][candidate], candidate))
            self.frontier_probes += 1
            return int(action), None, "probe-learned-event-frontier"
        self.active_steps, self.active_option_keys = planned; self.active_index = 0
        for key in self.active_option_keys:
            self.options[key]["replays"] += 1
            self.episode_event_attempts[self.options[key]["target_event"]] += 1
        self.option_plans += 1
        if len(self.active_option_keys) > 1:
            self.multi_option_plans += 1
        self.last_plan = {"events": [self.options[key]["target_event"]
                                      for key in self.active_option_keys],
                          "option_count": len(self.active_option_keys),
                          "action_count": len(self.active_steps)}
        step = self.active_steps[0]; self.option_steps += 1
        return int(step["action"]), self._normal_data(step.get("data")), "execute-learned-event-option"

    def observe(self, state: str, action: int, next_state: str, events: list[str],
                progress: bool, done: bool, data: dict | None = None) -> None:
        token = {"state": str(state), "action": int(action), "next_state": str(next_state),
                 "data": self._normal_data(data)}
        self.state_action_attempts[str(state)][int(action)] += 1
        self.observations += 1
        self.segment.append(token); self.segment = self.segment[-32:]
        was_novel_state = next_state not in self.known_states
        self.known_states.add(state); self.known_states.add(next_state)
        salient = _salient(events)
        for event in salient:
            if self.event_achievements[event] == 0:
                self.events_discovered += 1
            self.event_achievements[event] += 1
            self.event_actions[event][int(action)] += 1
            if self.previous_event is not None and self.previous_event != event:
                self.event_edges[self.previous_event][event] += 1
            self._record_option(event, list(self.segment), was_novel_state)
            self.previous_event = event
            if event not in self.episode_events:
                self.episode_events.append(event)
        if salient:
            self.segment = []

        event_count = max(1, self.events_discovered)
        coherent = (self.events_discovered >= 2 and
                    len(self.options) <= max(8, int(2.0 * event_count)))
        self.coherence_streak = self.coherence_streak + 1 if coherent else 0

        if self.active_index < len(self.active_steps):
            expected = self.active_steps[self.active_index]
            target_seen = expected.get("target_event") in salient
            transition_seen = expected["next_state"] == next_state
            if int(expected["action"]) == int(action) and (target_seen or transition_seen):
                completed_key = expected.get("option_key")
                self.active_index += 1
                next_key = (self.active_steps[self.active_index].get("option_key")
                            if self.active_index < len(self.active_steps) else None)
                if completed_key and completed_key != next_key and completed_key in self.options:
                    self.options[completed_key]["completions"] += 1
                    self.option_completions += 1
                if self.active_index >= len(self.active_steps):
                    self.active_steps = []; self.active_index = 0; self.active_option_keys = []
            else:
                key = expected.get("option_key")
                if key in self.options:
                    self.options[key]["failures"] += 1
                self.option_aborts += 1; self.active_steps = []; self.active_index = 0
                self.active_option_keys = []

        if progress:
            credit = 1.0
            for event in reversed(self.episode_events[-16:]):
                self.event_progress[event] += credit
                credit *= .9
        if done:
            self.segment = []
            if not progress and self.active_index < len(self.active_steps):
                key = self.active_steps[self.active_index].get("option_key")
                if key in self.options:
                    self.options[key]["failures"] += 1
                self.option_aborts += 1
            self.active_steps = []; self.active_index = 0; self.active_option_keys = []

    def status(self) -> dict:
        return {"format": self.format, "events_discovered": self.events_discovered,
                "event_options": len(self.options),
                "event_dependencies": sum(len(rows) for rows in self.event_edges.values()),
                "option_plans": self.option_plans, "option_steps": self.option_steps,
                "option_completions": self.option_completions,
                "option_aborts": self.option_aborts,
                "multi_option_plans": self.multi_option_plans,
                "frontier_probes": self.frontier_probes,
                "observations": self.observations,
                "coherence_streak": self.coherence_streak,
                "trusted_for_control": self._trusted(),
                "progress_credited_events": len(self.event_progress),
                "last_plan": self.last_plan}

    def export(self) -> dict:
        return self.status() | {
            "options_data": self.options,
            "event_edges": {event: dict(rows) for event, rows in self.event_edges.items()},
            "event_actions": {event: dict(rows) for event, rows in self.event_actions.items()},
            "state_action_attempts": {state: dict(rows)
                                      for state, rows in self.state_action_attempts.items()},
            "event_achievements": dict(self.event_achievements),
            "event_progress": dict(self.event_progress),
            "known_states_data": sorted(self.known_states)}

    def restore(self, value: dict) -> None:
        if value.get("format") != self.format:
            raise ValueError("unsupported event option format")
        self.options = {str(key): dict(row) for key, row in value.get("options_data", {}).items()}
        self.by_entry = defaultdict(set)
        for key, row in self.options.items():
            self.by_entry[str(row["entry_state"])].add(key)
        self.event_edges = defaultdict(Counter, {str(event): Counter(rows)
                                                   for event, rows in value.get("event_edges", {}).items()})
        self.event_actions = defaultdict(Counter, {str(event): Counter({int(a): int(n)
            for a, n in rows.items()}) for event, rows in value.get("event_actions", {}).items()})
        self.state_action_attempts = defaultdict(Counter, {str(state): Counter({int(a): int(n)
            for a, n in rows.items()}) for state, rows in value.get("state_action_attempts", {}).items()})
        self.event_achievements = Counter({str(k): int(v)
                                           for k, v in value.get("event_achievements", {}).items()})
        self.event_progress = Counter({str(k): float(v)
                                       for k, v in value.get("event_progress", {}).items()})
        self.known_states = set(map(str, value.get("known_states_data", [])))
        for name in ("events_discovered", "option_plans", "option_steps",
                     "option_completions", "option_aborts", "multi_option_plans",
                     "frontier_probes", "observations", "coherence_streak"):
            setattr(self, name, int(value.get(name, 0)))
        self.options_discovered = len(self.options)
        self.last_plan = value.get("last_plan")
