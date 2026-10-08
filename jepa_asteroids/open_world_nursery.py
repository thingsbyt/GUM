"""Open-world visual causal discovery and self-directed experimentation.

This module deliberately does not contain a catalogue of object meanings or
recipes.  A world is rendered as pixels, the learner segments persistent
objects, proposes untried object-object interventions, records visual changes,
and turns repeatable changes into an executable causal graph.  Desired objects
are shown visually; reward is sparse and arrives only when the desired object
has actually been made.

The benchmark is small enough to audit, but exercises mechanisms that are
useful beyond the synthetic world: novelty seeking, active experiments,
open-ended rule storage, multi-step planning, retention, and cross-world
affordance learning from appearance rather than object identity.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HEIGHT, WIDTH = 112, 176
SLOT_CENTERS = [(20 + 34 * x, 50 + 28 * y) for y in range(3) for x in range(5)]
MAX_SLOTS = len(SLOT_CENTERS)


def _masks() -> list[np.ndarray]:
    """A bank of unlabeled visual forms; indices have no learner-visible meaning."""
    bank = []
    for index in range(24):
        rng = np.random.default_rng(91_000 + index)
        value = np.zeros((11, 11), dtype=np.uint8)
        # Connected random walk gives irregular but segmentable objects.
        y = x = 5; value[y, x] = 1
        for _ in range(33 + index % 8):
            dy, dx = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
            y = int(np.clip(y + dy, 1, 9)); x = int(np.clip(x + dx, 1, 9)); value[y, x] = 1
        # A stable one-pixel dilation keeps components legible.
        grown = value.copy()
        for yy, xx in zip(*np.where(value)):
            grown[max(0, yy - 1):min(11, yy + 2), max(0, xx - 1):min(11, xx + 2)] = 1
        bank.append(grown)
    return bank


MASKS = _masks()
PALETTE = np.asarray([
    (231, 76, 60), (52, 152, 219), (46, 204, 113), (241, 196, 15),
    (155, 89, 182), (230, 126, 34), (26, 188, 156), (236, 240, 241),
    (127, 140, 141), (243, 156, 18), (52, 73, 94), (255, 102, 153),
    (102, 204, 255), (153, 255, 102), (255, 204, 102), (204, 153, 255),
], dtype=np.uint8)


@dataclass(frozen=True)
class SeenObject:
    signature: str
    form: str
    x: int
    y: int


def _component_signature(patch: np.ndarray, mask: np.ndarray) -> tuple[str, str]:
    ys, xs = np.where(mask); crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    rgb = patch[:, mask].mean(1).round().astype(np.uint8)
    form = hashlib.sha256(crop.tobytes() + bytes(crop.shape)).hexdigest()[:12]
    identity = hashlib.sha256(crop.tobytes() + rgb.tobytes()).hexdigest()[:16]
    return identity, form


def perceive(frame: np.ndarray) -> dict:
    """Segment the raw image without access to world state or type ids."""
    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.shape != (3, HEIGHT, WIDTH):
        raise ValueError(f"unexpected nursery frame {rgb.shape}")
    foreground = rgb.max(0) > 30
    seen = np.zeros_like(foreground, dtype=bool); components = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if seen[y, x] or not foreground[y, x]:
                continue
            stack = [(y, x)]; seen[y, x] = True; points = []
            while stack:
                yy, xx = stack.pop(); points.append((yy, xx))
                for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                    ny, nx = yy + dy, xx + dx
                    if (0 <= ny < HEIGHT and 0 <= nx < WIDTH and foreground[ny, nx]
                            and not seen[ny, nx]):
                        seen[ny, nx] = True; stack.append((ny, nx))
            if len(points) < 8:
                continue
            yy = np.asarray([p[0] for p in points]); xx = np.asarray([p[1] for p in points])
            y0, y1, x0, x1 = int(yy.min()), int(yy.max()), int(xx.min()), int(xx.max())
            local = foreground[y0:y1 + 1, x0:x1 + 1]
            signature, form = _component_signature(rgb[:, y0:y1 + 1, x0:x1 + 1], local)
            components.append(SeenObject(signature, form, int(xx.mean()), int(yy.mean())))
    # The goal lives above the separator. The separator itself is filtered as
    # an extremely wide component. Everything below it is manipulable.
    objects = [o for o in components if o.y > 34 and o.x < 171]
    goals = [o for o in components if o.y < 29 and o.x < 45]
    if len(goals) != 1:
        raise RuntimeError(f"expected one visual goal, perceived {len(goals)}")
    return {"goal": goals[0], "objects": sorted(objects, key=lambda o: (o.y, o.x))}


class OpenWorldNursery:
    """A reskinnable visual world with a hidden, generated causal recipe graph."""

    def __init__(self, seed: int, *, synthesis: bool = True, goal: str = "final"):
        self.seed = int(seed); self.synthesis = bool(synthesis); self.goal_name = goal
        rng = np.random.default_rng(seed)
        type_count = 13 if synthesis else 10
        # Worlds reskin every identity and change their recipes, while retaining
        # a discoverable visual regularity: some forms tend to afford being the
        # second object in an intervention, roots tend to be first objects, and
        # manufactured forms tend to participate in later compositions. These
        # roles are not exposed to the learner; it must infer them statistically.
        forms = np.concatenate((rng.choice(np.arange(6, 12), 2, replace=False),
                                rng.choice(np.arange(0, 6), 4, replace=False),
                                rng.choice(np.arange(12, 24), type_count - 6, replace=False)))
        colors = rng.choice(len(PALETTE), type_count, replace=False)
        self.visual = {f"v{i}": (int(forms[i]), int(colors[i])) for i in range(type_count)}
        # The environment uses arbitrary private ids. The learner never sees
        # these names or this graph; it receives pixels and consequences only.
        self.initial = ["v0", "v1", "v2", "v3", "v4", "v5"]
        self.rules = {
            ("v0", "v2"): "v6", ("v6", "v3"): "v7",
            ("v1", "v4"): "v8", ("v1", "v5"): "v9",
        }
        if synthesis:
            self.rules.update({("v8", "v5"): "v10", ("v7", "v10"): "v11",
                               ("v9", "v3"): "v12"})
            self.goals = {"shallow": "v6", "branch": "v10", "final": "v11"}
        else:
            self.rules.update({("v7", "v8"): "v9"})
            self.goals = {"shallow": "v6", "branch": "v8", "final": "v9"}
        self.goal_type = self.goals[goal]
        self.rng = rng; self.episode = 0; self.interactions = 0

    def reset(self, episode_seed: int | None = None) -> np.ndarray:
        self.episode += 1
        rng = np.random.default_rng(self.seed * 1009 + self.episode if episode_seed is None else episode_seed)
        self.inventory = list(self.initial); rng.shuffle(self.inventory)
        slots = rng.choice(MAX_SLOTS, len(self.inventory), replace=False)
        self.locations = {kind: int(slot) for kind, slot in zip(self.inventory, slots)}
        self.done = False; self.interactions = 0
        return self.render()

    def _nearest(self, x: int, y: int) -> str | None:
        if not self.locations: return None
        kind, distance = min(((kind, (SLOT_CENTERS[slot][0] - x) ** 2 +
                                      (SLOT_CENTERS[slot][1] - y) ** 2)
                              for kind, slot in self.locations.items()), key=lambda row: row[1])
        return kind if distance <= 15 ** 2 else None

    def step(self, action: tuple[int, int, int, int]):
        if self.done: raise RuntimeError("episode already ended")
        ax, ay, bx, by = map(int, action); left, right = self._nearest(ax, ay), self._nearest(bx, by)
        created = None
        if left is not None and right is not None:
            outcome = self.rules.get((left, right))
            if outcome is not None and outcome not in self.inventory and len(self.inventory) < MAX_SLOTS:
                empty = sorted(set(range(MAX_SLOTS)) - set(self.locations.values()))
                self.inventory.append(outcome); self.locations[outcome] = empty[0]; created = outcome
        self.interactions += 1
        success = self.goal_type in self.inventory
        self.done = success
        return self.render(), (1.0 if success else 0.0), self.done, {
            "success": success, "created_private_audit_only": created}

    def render(self) -> np.ndarray:
        frame = np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8)
        # Goal is a raw visual sample, not a symbolic task id.
        self._paint(frame, self.goal_type, 20, 16)
        frame[:, 32:34, 4:172] = 24
        for kind, slot in self.locations.items():
            self._paint(frame, kind, *SLOT_CENTERS[slot])
        return frame

    def _paint(self, frame: np.ndarray, kind: str, cx: int, cy: int) -> None:
        mask_index, color_index = self.visual[kind]; mask = MASKS[mask_index]
        color = PALETTE[color_index]
        y0, x0 = cy - 5, cx - 5
        for channel in range(3):
            patch = frame[channel, y0:y0 + 11, x0:x0 + 11]
            patch[mask > 0] = color[channel]


class OpenWorldMind:
    """Persistent causal graph learner with curiosity-generated experiments."""

    format = "wailah-open-world-mind-v1"

    def __init__(self):
        self.worlds: dict[str, dict] = {}
        self.form_roles = defaultdict(lambda: Counter())
        self.total_experiments = 0; self.total_predictions = 0; self.correct_predictions = 0
        self.rules_discovered = 0; self.self_generated_goals = 0
        self.current_key = None; self.pending = None

    @staticmethod
    def _world_key(view: dict) -> str:
        # Initial object identities define a reskinned causal world; layout and
        # requested goal do not. All benchmark tasks begin from reset.
        identities = sorted(o.signature for o in view["objects"])
        return hashlib.sha256("|".join(identities).encode()).hexdigest()[:20]

    def begin(self, frame: np.ndarray) -> str:
        view = perceive(frame); self.current_key = self._world_key(view)
        self.worlds.setdefault(self.current_key, {"rules": {}, "tested": [], "visits": 0})
        self.worlds[self.current_key]["visits"] += 1; self.pending = None
        return self.current_key

    @property
    def memory(self) -> dict:
        if self.current_key is None: raise RuntimeError("call begin first")
        return self.worlds[self.current_key]

    @staticmethod
    def _pair_key(a: str, b: str) -> str: return f"{a}>{b}"

    def _plan(self, visible: set[str], goal: str) -> list[tuple[str, str]] | None:
        rules = {tuple(key.split(">", 1)): value for key, value in self.memory["rules"].items()}
        queue = deque([(frozenset(visible), [])]); visited = {frozenset(visible)}
        while queue:
            state, plan = queue.popleft()
            if goal in state: return plan
            for pair, outcome in sorted(rules.items()):
                if pair[0] in state and pair[1] in state and outcome not in state:
                    nxt = frozenset(set(state) | {outcome})
                    if nxt not in visited:
                        visited.add(nxt); queue.append((nxt, plan + [pair]))
        return None

    def act(self, frame: np.ndarray) -> tuple[tuple[int, int, int, int], dict]:
        view = perceive(frame); by_id = {o.signature: o for o in view["objects"]}
        goal = view["goal"].signature; visible = set(by_id)
        plan = self._plan(visible, goal)
        reason = "execute-discovered-causal-plan"
        if plan:
            pair = plan[0]
        else:
            # Reconstruct known products before probing the unknown frontier.
            # This is essential across short episodes: knowledge survives even
            # though transient objects disappear when the physical world resets.
            frontier = []
            for key, outcome in self.memory["rules"].items():
                a, b = key.split(">", 1)
                if a in visible and b in visible and outcome not in visible:
                    frontier.append((a, b))
            if frontier:
                pair = sorted(frontier)[0]
                reason = "reconstruct-known-causal-frontier"
                left, right = by_id[pair[0]], by_id[pair[1]]
                action = (left.x, left.y, right.x, right.y)
                self.pending = {"pair": pair, "left_form": left.form, "right_form": right.form,
                                "known_prediction": self.memory["rules"].get(self._pair_key(*pair))}
                return action, {"reason": reason, "hypothesis": f"intervene({pair[0]},{pair[1]})",
                                "goal": goal, "known_rules": len(self.memory["rules"])}
            tested = set(self.memory["tested"]); candidates = []
            for left in view["objects"]:
                for right in view["objects"]:
                    if left.signature == right.signature: continue
                    key = self._pair_key(left.signature, right.signature)
                    if key in tested: continue
                    # Transfer only an affordance preference, never a recipe:
                    # visual forms that often caused effects as the second
                    # argument are tested earlier in a new world.
                    prior = self.form_roles[right.form]
                    p_effect = (prior["target_effect"] + 1) / (prior["target_tests"] + 2)
                    source = self.form_roles[left.form]
                    p_source = (source["source_effect"] + 1) / (source["source_tests"] + 2)
                    candidates.append((3 * p_effect + p_source, key, left.signature, right.signature))
            if not candidates:
                # All currently possible interventions are known. Re-test the
                # least-supported rule instead of inventing a fake action.
                if not self.memory["rules"]:
                    pair = (view["objects"][0].signature, view["objects"][-1].signature)
                else:
                    key = sorted(self.memory["rules"])[0]; pair = tuple(key.split(">", 1))
                reason = "repeat-low-evidence-intervention"
            else:
                _, _, a, b = max(candidates, key=lambda row: (row[0], row[1]))
                pair = (a, b); reason = "self-generated-causal-experiment"
                self.self_generated_goals += 1
        left, right = by_id[pair[0]], by_id[pair[1]]
        action = (left.x, left.y, right.x, right.y)
        self.pending = {"pair": pair, "left_form": left.form, "right_form": right.form,
                        "known_prediction": self.memory["rules"].get(self._pair_key(*pair))}
        return action, {"reason": reason, "hypothesis": f"intervene({pair[0]},{pair[1]})",
                        "goal": goal, "known_rules": len(self.memory["rules"])}

    def observe(self, before: np.ndarray, action, after: np.ndarray, reward: float, done: bool) -> dict:
        if self.pending is None: raise RuntimeError("act must precede observe")
        prior = perceive(before); later = perceive(after)
        old = {o.signature for o in prior["objects"]}; new_objects = [o for o in later["objects"] if o.signature not in old]
        outcome = new_objects[0].signature if len(new_objects) == 1 else None
        pair = self.pending["pair"]; key = self._pair_key(*pair)
        if key not in self.memory["tested"]: self.memory["tested"].append(key)
        lf, rf = self.pending["left_form"], self.pending["right_form"]
        self.form_roles[lf]["source_tests"] += 1; self.form_roles[rf]["target_tests"] += 1
        self.total_experiments += 1
        if outcome is not None:
            if key not in self.memory["rules"]: self.rules_discovered += 1
            self.memory["rules"][key] = outcome
            self.form_roles[lf]["source_effect"] += 1; self.form_roles[rf]["target_effect"] += 1
            self.form_roles[new_objects[0].form]["produced"] += 1
        prediction = self.pending["known_prediction"]
        if prediction is not None:
            self.total_predictions += 1; self.correct_predictions += int(prediction == outcome or prediction in old)
        self.pending = None
        return {"new_visual_concept": outcome, "reward": float(reward), "done": bool(done)}

    def invented_concepts(self) -> dict:
        concepts = Counter()
        for rows in self.form_roles.values():
            profile = (int(rows["source_effect"] > 0), int(rows["target_effect"] > 0),
                       int(rows["produced"] > 0))
            concepts["causal-profile-" + "".join(map(str, profile))] += 1
        return dict(concepts)

    def status(self) -> dict:
        return {"format": self.format, "remembered_worlds": len(self.worlds),
                "visual_forms_experienced": len(self.form_roles),
                "causal_rules_discovered": self.rules_discovered,
                "self_generated_experiments": self.self_generated_goals,
                "invented_affordance_concepts": self.invented_concepts(),
                "prediction_accuracy": self.correct_predictions / max(1, self.total_predictions)}

    def export(self) -> dict:
        return self.status() | {"worlds": self.worlds,
            "form_roles": {key: dict(value) for key, value in self.form_roles.items()},
            "total_experiments": self.total_experiments,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "rules_discovered": self.rules_discovered,
            "self_generated_goals": self.self_generated_goals}

    @classmethod
    def restore(cls, value: dict) -> "OpenWorldMind":
        if value.get("format") != cls.format: raise ValueError("unsupported open-world memory")
        result = cls(); result.worlds = value.get("worlds", {})
        result.form_roles = defaultdict(lambda: Counter(), {
            key: Counter(rows) for key, rows in value.get("form_roles", {}).items()})
        for name in ("total_experiments", "total_predictions", "correct_predictions",
                     "rules_discovered", "self_generated_goals"):
            setattr(result, name, int(value.get(name, 0)))
        return result

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.export(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "OpenWorldMind":
        return cls.restore(json.loads(Path(path).read_text(encoding="utf-8")))


def _run_until_success(mind: OpenWorldMind, world: OpenWorldNursery, budget: int,
                       seed: int, episode_horizon: int = 45) -> dict:
    interactions = 0; episodes = 0; trace = []; success = False
    while interactions < budget and not success:
        frame = world.reset(seed + episodes); mind.begin(frame); episodes += 1
        for _ in range(min(episode_horizon, budget - interactions)):
            action, evidence = mind.act(frame); nxt, reward, done, info = world.step(action)
            learned = mind.observe(frame, action, nxt, reward, done); interactions += 1
            trace.append({"interaction": interactions, "reason": evidence["reason"],
                          "known_rules": evidence["known_rules"], "new_concept": learned["new_visual_concept"],
                          "reward": reward})
            frame = nxt
            if done: success = True; break
    return {"success": success, "interactions": interactions, "episodes": episodes,
            "rules_known": len(mind.memory["rules"]), "trace": trace}


def _random_baseline(world: OpenWorldNursery, budget: int, seed: int,
                     episode_horizon: int = 45) -> dict:
    rng = np.random.default_rng(seed); interactions = 0; success = False; episodes = 0
    while interactions < budget and not success:
        frame = world.reset(seed + episodes); episodes += 1
        for _ in range(min(episode_horizon, budget - interactions)):
            view = perceive(frame); objects = view["objects"]
            picks = rng.choice(len(objects), 2, replace=False); a, b = objects[int(picks[0])], objects[int(picks[1])]
            frame, reward, done, _ = world.step((a.x, a.y, b.x, b.y)); interactions += 1
            if done: success = True; break
    return {"success": success, "interactions": interactions, "episodes": episodes}


def _summary(rows: list[dict]) -> dict:
    return {"worlds": len(rows), "successes": sum(row["success"] for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_interactions": float(np.mean([row["interactions"] for row in rows]))}


def run_open_world_benchmark(output: Path, *, worlds: int = 24, budget: int = 420,
                             seed: int = 73_001) -> dict:
    """Run frozen discovery, ablation, persistence, and retention evaluations."""
    mind = OpenWorldMind(); full, random_rows, scratch_rows = [], [], []
    no_transfer_rows = []
    for index in range(worlds):
        world_seed = seed + index * 7919
        full.append(_run_until_success(mind, OpenWorldNursery(world_seed, synthesis=True), budget,
                                       seed + 1_000_000 + index * 1000))
        no_transfer_rows.append(_run_until_success(
            OpenWorldMind(), OpenWorldNursery(world_seed, synthesis=True), budget,
            seed + 1_000_000 + index * 1000))
        random_rows.append(_random_baseline(OpenWorldNursery(world_seed, synthesis=True), budget,
                                            seed + 2_000_000 + index * 1000))
        # Destroy cross-episode memory: each short episode gets a fresh mind.
        remaining = budget; success = False; used = 0; attempts = 0
        while remaining and not success:
            row = _run_until_success(OpenWorldMind(), OpenWorldNursery(world_seed, synthesis=True),
                                     min(45, remaining), seed + 3_000_000 + index * 1000 + attempts)
            success = row["success"]; used += row["interactions"]; remaining -= row["interactions"]; attempts += 1
        scratch_rows.append({"success": success, "interactions": used, "episodes": attempts})

    # Revisit old worlds after all intervening learning. A changed visual goal
    # tests whether remembered causal graphs can be recomposed, not replayed.
    revisits = []
    for index in range(worlds):
        world_seed = seed + index * 7919
        revisits.append(_run_until_success(mind, OpenWorldNursery(world_seed, synthesis=True, goal="branch"),
                                           40, seed + 4_000_000 + index * 1000))

    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    brain = output.with_name("OPEN_WORLD_MIND.json"); mind.save(brain)
    restored = OpenWorldMind.load(brain)
    reload_rows = []
    for index in range(min(8, worlds)):
        world_seed = seed + index * 7919
        reload_rows.append(_run_until_success(restored, OpenWorldNursery(world_seed, synthesis=True, goal="shallow"),
                                              20, seed + 5_000_000 + index * 1000))
    report = {"format": "wailah-open-world-nursery-audit-v1",
        "protocol": {"seed": seed, "worlds": worlds, "interaction_budget": budget,
            "observation": "raw RGB pixels", "actions": "two visually selected object coordinates",
            "reward": "zero until the visually requested object exists",
            "agent_not_given": ["object ids", "object classes", "recipe graph", "rule count",
                                "tool identities", "intermediate rewards", "solution sequence"],
            "mechanism": "self-generated pair interventions, visual differencing, causal graph planning"},
        "aggregate": {"open_world_mind": _summary(full), "random_interventions": _summary(random_rows),
                      "local_memory_without_cross_world_affordances": _summary(no_transfer_rows),
                      "memory_erased_each_episode": _summary(scratch_rows),
                      "later_revisit_new_goals": _summary(revisits),
                      "reloaded_memory": _summary(reload_rows)},
        "growth": mind.status(),
        "learning_to_learn": {
            "first_quartile_mean_interactions": float(np.mean(
                [row["interactions"] for row in full[:max(1, worlds // 4)]])),
            "last_quartile_mean_interactions": float(np.mean(
                [row["interactions"] for row in full[-max(1, worlds // 4):]])),
            "no_cross_world_affordance_mean_interactions": float(np.mean(
                [row["interactions"] for row in no_transfer_rows])),
            "cross_world_speedup_vs_local_only": float(
                np.mean([row["interactions"] for row in no_transfer_rows]) /
                np.mean([row["interactions"] for row in full]))},
        "per_world": full,
        "persistent_brain": str(brain),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=24); parser.add_argument("--budget", type=int, default=420)
    parser.add_argument("--seed", type=int, default=73_001); args = parser.parse_args(argv)
    report = run_open_world_benchmark(args.output, worlds=args.worlds, budget=args.budget, seed=args.seed)
    print(json.dumps(report["aggregate"] | {"growth": report["growth"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
