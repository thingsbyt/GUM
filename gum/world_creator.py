"""Procedural practice-world creator with explicit evolutionary lineage."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .protocol import PublicWorldSpec, Transition


MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class GridGenome:
    seed: int
    difficulty: int
    size: int
    walls: tuple[tuple[int, int], ...]
    hazards: tuple[tuple[int, int], ...]
    goal: tuple[int, int]
    action_map: tuple[int, ...]
    parent_world_id: str | None
    mutation: str

    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


class ProceduralGridWorld:
    """Pixels + anonymous discrete actions + scalar consequence."""
    def __init__(self, genome: GridGenome):
        self.genome = genome; self.world_id = f"grid-{genome.identity()}"
        self._rng = np.random.default_rng(genome.seed + 7001); self.steps = 0; self.position = (0, 0)
        self._free = [(r, c) for r in range(genome.size) for c in range(genome.size)
                      if (r, c) not in genome.walls and (r, c) not in genome.hazards and (r, c) != genome.goal]

    def public_spec(self):
        pixels = self.genome.size * 12
        return PublicWorldSpec(self.world_id, "procedural-navigation", "gum-grid-v1", 1, "rgb",
            (pixels, pixels, 3), "discrete-anonymous", 4, self.genome.size * self.genome.size * 2,
            (-1.0, 1.0))

    def reset(self, seed=None):
        if seed is not None: self._rng = np.random.default_rng(int(seed))
        self.position = self._free[int(self._rng.integers(len(self._free)))]; self.steps = 0
        return self._render(labels=False)

    def step(self, action):
        action = int(action); reward = -0.01; terminated = False
        if not 0 <= action < 4: raise ValueError(action)
        semantic_move = self.genome.action_map[action]; dr, dc = MOVES[semantic_move]
        nxt = self.position[0] + dr, self.position[1] + dc
        if not (0 <= nxt[0] < self.genome.size and 0 <= nxt[1] < self.genome.size) or nxt in self.genome.walls:
            reward = -0.04
        else: self.position = nxt
        self.steps += 1
        if self.position in self.genome.hazards: reward = -1.0; terminated = True
        elif self.position == self.genome.goal: reward = 1.0; terminated = True
        truncated = self.steps >= self.public_spec().horizon and not terminated
        return Transition(self._render(labels=False), reward, terminated, truncated,
                          {"success": bool(terminated and reward > 0)})

    def _render(self, labels=False):
        size, scale = self.genome.size, 12
        image = Image.new("RGB", (size * scale, size * scale), (16, 25, 36)); draw = ImageDraw.Draw(image)
        for row in range(size):
            for col in range(size):
                x0, y0 = col * scale, row * scale
                draw.rectangle((x0, y0, x0 + scale - 1, y0 + scale - 1), outline=(34, 48, 63))
        for row, col in self.genome.walls:
            x0, y0 = col * scale, row * scale; draw.rectangle((x0 + 1, y0 + 1, x0 + 10, y0 + 10), fill=(92, 101, 114))
        for row, col in self.genome.hazards:
            x0, y0 = col * scale, row * scale; draw.rectangle((x0 + 2, y0 + 2, x0 + 9, y0 + 9), fill=(230, 72, 85))
        row, col = self.genome.goal; x0, y0 = col * scale, row * scale
        draw.ellipse((x0 + 2, y0 + 2, x0 + 9, y0 + 9), fill=(75, 224, 148))
        row, col = self.position; x0, y0 = col * scale, row * scale
        draw.polygon(((x0 + 6, y0 + 1), (x0 + 10, y0 + 10), (x0 + 2, y0 + 10)), fill=(83, 194, 255))
        return np.asarray(image, dtype=np.uint8)

    def human_frame(self): return self._render(labels=True)
    def audit_state(self):
        return {"position": self.position, "goal": self.genome.goal, "hazards": self.genome.hazards,
                "walls": self.genome.walls, "action_map": self.genome.action_map, "steps": self.steps}


class WorldCreator:
    def __init__(self, root: Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _reachable(size, walls, blocked, goal):
        queue = deque([goal]); seen = {goal}
        while queue:
            row, col = queue.popleft()
            for dr, dc in MOVES:
                nxt = row + dr, col + dc
                if 0 <= nxt[0] < size and 0 <= nxt[1] < size and nxt not in walls and nxt not in blocked and nxt not in seen:
                    seen.add(nxt); queue.append(nxt)
        return seen

    def create(self, *, seed: int, difficulty: int = 0, parent_world_id: str | None = None,
               mutation: str = "origin") -> Path:
        rng = np.random.default_rng(seed); size = min(8, 5 + difficulty)
        cells = [(r, c) for r in range(size) for c in range(size)]; rng.shuffle(cells)
        goal = cells.pop(); hazards = tuple(cells.pop() for _ in range(max(1, difficulty)))
        walls = []
        for candidate in cells:
            if len(walls) >= difficulty * 2: break
            trial = set(walls) | {candidate}
            reachable = self._reachable(size, trial, set(hazards), goal)
            if len(reachable) >= size * size - len(trial) - len(hazards): walls.append(candidate)
        genome = GridGenome(seed, difficulty, size, tuple(sorted(walls)), tuple(sorted(hazards)), goal,
                            tuple(int(x) for x in rng.permutation(4)), parent_world_id, mutation)
        world_id = f"grid-{genome.identity()}"; folder = self.root / world_id; folder.mkdir(parents=True, exist_ok=True)
        world = ProceduralGridWorld(genome); spec = world.public_spec()
        public = spec.to_json() | {"created_by": "GUM World Creator v1",
                                  "parent_world_id": parent_world_id, "mutation": mutation}
        (folder / "world.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
        (folder / "genome.private.json").write_text(json.dumps(asdict(genome), indent=2), encoding="utf-8")
        return folder

    def mutate(self, parent_folder: Path, *, seed: int, harder=True) -> Path:
        parent = load_genome(Path(parent_folder)); difficulty = max(0, parent.difficulty + (1 if harder else -1))
        return self.create(seed=seed, difficulty=difficulty, parent_world_id=f"grid-{parent.identity()}",
                           mutation="increase-difficulty" if harder else "decrease-difficulty")


def load_genome(folder: Path) -> GridGenome:
    value = json.loads((Path(folder) / "genome.private.json").read_text(encoding="utf-8"))
    for key in ("walls", "hazards"): value[key] = tuple(tuple(row) for row in value[key])
    value["goal"] = tuple(value["goal"]); value["action_map"] = tuple(value["action_map"])
    return GridGenome(**value)


def load_world(folder: Path) -> ProceduralGridWorld:
    return ProceduralGridWorld(load_genome(folder))
