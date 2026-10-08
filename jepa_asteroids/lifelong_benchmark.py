"""Multi-game pixel-control benchmark for the growable WAILAH skill bank."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .lifelong import LifelongSkillBank, Skill, TaskSpec
from .runtime import Guard, atomic_json
from .state_memory import perceptual_state_key


SIZE = 32


def _block(frame: np.ndarray, x: int, y: int, value: int = 255, radius: int = 1) -> None:
    x = int(np.clip(x, radius, SIZE - radius - 1))
    y = int(np.clip(y, radius, SIZE - radius - 1))
    frame[0, y-radius:y+radius+1, x-radius:x+radius+1] = value


class CatchGame:
    """Move a paddle under falling targets. State is private; observations are pixels."""
    action_dim = 3
    def __init__(self, horizon: int = 72): self.horizon = horizon
    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed); self.step_count = 0
        self.paddle = int(self.rng.integers(4, SIZE - 4)); self.caught = self.missed = 0
        self._spawn(); return self.render()
    def _spawn(self):
        self.target = int(self.rng.integers(2, SIZE - 2)); self.height = 2
    def step(self, action: int):
        self.paddle = int(np.clip(self.paddle + (action == 2) - (action == 1), 3, SIZE - 4))
        self.height += 3; reward = -.003
        if self.height >= SIZE - 4:
            if abs(self.paddle - self.target) <= 3: reward += 1.0; self.caught += 1
            else: reward -= .35; self.missed += 1
            self._spawn()
        self.step_count += 1; done = self.step_count >= self.horizon
        return self.render(), reward, done, {'caught': self.caught, 'missed': self.missed}
    def render(self):
        frame = np.zeros((1, SIZE, SIZE), dtype=np.uint8)
        _block(frame, self.target, self.height, 255, 1)
        frame[0, SIZE-3:SIZE-1, self.paddle-3:self.paddle+4] = 180
        return frame


class AvoidGame:
    """Dodge falling hazards. It shares controls with Catch but reverses the objective."""
    action_dim = 3
    def __init__(self, horizon: int = 72): self.horizon = horizon
    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed); self.step_count = 0
        self.paddle = int(self.rng.integers(4, SIZE - 4)); self.dodged = self.collisions = 0
        self._spawn(); return self.render()
    def _spawn(self):
        self.hazard = int(self.rng.integers(2, SIZE - 2)); self.height = 2
    def step(self, action: int):
        self.paddle = int(np.clip(self.paddle + (action == 2) - (action == 1), 3, SIZE - 4))
        self.height += 3; reward = .002
        if self.height >= SIZE - 4:
            if abs(self.paddle - self.hazard) <= 3: reward -= 1.0; self.collisions += 1
            else: reward += .35; self.dodged += 1
            self._spawn()
        self.step_count += 1; done = self.step_count >= self.horizon
        return self.render(), reward, done, {'dodged': self.dodged, 'collisions': self.collisions}
    def render(self):
        frame = np.zeros((1, SIZE, SIZE), dtype=np.uint8)
        _block(frame, self.hazard, self.height, 220, 2)
        frame[0, SIZE-3:SIZE-1, self.paddle-3:self.paddle+4] = 255
        return frame


class NavigateGame:
    """Navigate to changing goals in a small visual grid using four actions."""
    action_dim = 4
    grid = 6
    def __init__(self, horizon: int = 72): self.horizon = horizon
    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed); self.step_count = self.goals = 0
        self.player = np.asarray([int(self.rng.integers(self.grid)), int(self.rng.integers(self.grid))])
        self._goal(); return self.render()
    def _goal(self):
        while True:
            goal = np.asarray([int(self.rng.integers(self.grid)), int(self.rng.integers(self.grid))])
            if np.any(goal != self.player): self.goal = goal; return
    def step(self, action: int):
        old = int(np.abs(self.player - self.goal).sum())
        delta = np.asarray(((-1,0),(1,0),(0,-1),(0,1))[action])
        self.player = np.clip(self.player + delta, 0, self.grid - 1)
        new = int(np.abs(self.player - self.goal).sum())
        reward = .04 * (old - new) - .005
        if new == 0:
            reward += 1.0; self.goals += 1; self._goal()
        self.step_count += 1; done = self.step_count >= self.horizon
        return self.render(), reward, done, {'goals': self.goals}
    def render(self):
        frame = np.zeros((1, SIZE, SIZE), dtype=np.uint8); cell = SIZE // self.grid
        gx, gy = int(self.goal[1]*cell+cell//2), int(self.goal[0]*cell+cell//2)
        px, py = int(self.player[1]*cell+cell//2), int(self.player[0]*cell+cell//2)
        _block(frame, gx, gy, 150, 2); _block(frame, px, py, 255, 1)
        return frame


class KeyDoorGame:
    """Acquire a key, then unlock a door while avoiding visible hazards."""
    action_dim = 4
    grid = 6
    def __init__(self, horizon: int = 96): self.horizon = horizon
    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed); self.step_count = 0
        self.keys = self.unlocks = self.hazard_hits = 0; self.has_key = False
        self._layout(); return self.render()
    def _cell(self):
        return np.asarray([int(self.rng.integers(self.grid)), int(self.rng.integers(self.grid))])
    def _layout(self):
        occupied = set()
        def unique():
            while True:
                cell = self._cell(); item = tuple(int(x) for x in cell)
                if item not in occupied: occupied.add(item); return cell
        self.player = unique(); self.key = unique(); self.door = unique()
        self.hazards = [unique() for _ in range(4)]
    def _distance(self, destination):
        return int(np.abs(self.player - destination).sum())
    def step(self, action: int):
        destination = self.door if self.has_key else self.key
        old = self._distance(destination)
        delta = np.asarray(((-1,0),(1,0),(0,-1),(0,1))[action])
        candidate = np.clip(self.player + delta, 0, self.grid - 1)
        hit = any(np.array_equal(candidate, hazard) for hazard in self.hazards)
        if hit:
            reward = -.22; self.hazard_hits += 1
        else:
            self.player = candidate
            reward = .045 * (old - self._distance(destination)) - .006
        if not self.has_key and np.array_equal(self.player, self.key):
            self.has_key = True; self.keys += 1; reward += .55
        if self.has_key and np.array_equal(self.player, self.door):
            self.unlocks += 1; reward += 1.35; self.has_key = False; self._layout()
        self.step_count += 1; done = self.step_count >= self.horizon
        return self.render(), float(reward), done, {'keys': self.keys,
            'unlocks': self.unlocks, 'hazard_hits': self.hazard_hits}
    def render(self):
        frame = np.zeros((1, SIZE, SIZE), dtype=np.uint8); cell = SIZE // self.grid
        def xy(position):
            return int(position[1] * cell + cell // 2), int(position[0] * cell + cell // 2)
        for hazard in self.hazards:
            hx, hy = xy(hazard); _block(frame, hx, hy, 65, 2)
        dx, dy = xy(self.door); _block(frame, dx, dy, 125, 2)
        if not self.has_key:
            kx, ky = xy(self.key); _block(frame, kx, ky, 185, 1)
            frame[0, ky, max(0,kx-2):min(SIZE,kx+3)] = 220
        px, py = xy(self.player); _block(frame, px, py, 255, 1)
        if self.has_key: frame[0, 0:2, :] = 210
        return frame


class SignalGame:
    """Discover a hidden five-symbol control language from immediate rewards."""
    action_dim = 5
    mapping = (2, 4, 1, 3, 0)
    def __init__(self, horizon: int = 64): self.horizon = horizon
    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed); self.step_count = 0
        self.correct = self.errors = 0; self._next(); return self.render()
    def _next(self):
        self.symbol = int(self.rng.integers(5))
        self.cx = int(self.rng.integers(7, SIZE - 7)); self.cy = int(self.rng.integers(7, SIZE - 7))
        self.noise_seed = int(self.rng.integers(2**31))
    def step(self, action: int):
        success = int(action) == self.mapping[self.symbol]
        reward = 1.0 if success else -.25
        if success: self.correct += 1
        else: self.errors += 1
        self.step_count += 1; done = self.step_count >= self.horizon
        self._next()
        return self.render(), reward, done, {'correct': self.correct, 'errors': self.errors}
    def render(self):
        rng = np.random.default_rng(self.noise_seed)
        frame = rng.integers(0, 24, size=(1, SIZE, SIZE), dtype=np.uint8)
        x, y = self.cx, self.cy
        if self.symbol == 0:
            frame[0, y-5:y+6, x-1:x+2] = 255
        elif self.symbol == 1:
            frame[0, y-1:y+2, x-5:x+6] = 255
        elif self.symbol == 2:
            for d in range(-5, 6): frame[0, y+d, x+d] = 255; frame[0, y+d, x-d] = 255
        elif self.symbol == 3:
            frame[0, y-5:y+6, x-5:x-3] = 255; frame[0, y-5:y+6, x+4:x+6] = 255
            frame[0, y-5:y-3, x-5:x+6] = 255; frame[0, y+4:y+6, x-5:x+6] = 255
        else:
            frame[0, y-5:y+6, x-1:x+2] = 255; frame[0, y-1:y+2, x-5:x+6] = 255
        return frame


class MazeGame:
    """Escape a procedurally generated perfect maze from a full pixel view."""
    action_dim = 4
    cells = 5
    side = cells * 2 + 1
    def __init__(self, horizon: int = 256): self.horizon = horizon
    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed); self.step_count = self.wall_hits = 0
        self.success = False; self.maze = np.ones((self.side, self.side), dtype=np.bool_)
        start_cell = (int(self.rng.integers(self.cells)), int(self.rng.integers(self.cells)))
        visited = {start_cell}; stack = [start_cell]
        while stack:
            row, column = stack[-1]
            choices = [(row+dr,column+dc) for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))
                       if 0 <= row+dr < self.cells and 0 <= column+dc < self.cells
                       and (row+dr,column+dc) not in visited]
            if not choices: stack.pop(); continue
            nxt = choices[int(self.rng.integers(len(choices)))]; visited.add(nxt)
            ar, ac = 2*row+1, 2*column+1; br, bc = 2*nxt[0]+1, 2*nxt[1]+1
            self.maze[ar,ac] = False; self.maze[br,bc] = False
            self.maze[(ar+br)//2,(ac+bc)//2] = False; stack.append(nxt)
        self.player = np.asarray((2*start_cell[0]+1, 2*start_cell[1]+1))
        distances = self._distances(tuple(self.player)); farthest = max(distances, key=distances.get)
        self.exit = np.asarray(farthest); self.initial_distance = distances[farthest]
        return self.render()
    def _distances(self, origin):
        distances = {origin: 0}; queue = [origin]
        for row, column in queue:
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nxt = (row+dr,column+dc)
                if (0 <= nxt[0] < self.side and 0 <= nxt[1] < self.side
                        and not self.maze[nxt] and nxt not in distances):
                    distances[nxt] = distances[(row,column)] + 1; queue.append(nxt)
        return distances
    def _distance_to_exit(self):
        return self._distances(tuple(self.exit))[tuple(self.player)]
    def step(self, action: int):
        old = self._distance_to_exit(); delta = np.asarray(((-1,0),(1,0),(0,-1),(0,1))[action])
        candidate = self.player + delta
        if self.maze[tuple(candidate)]:
            reward = -.055; self.wall_hits += 1
        else:
            self.player = candidate; reward = .055 * (old - self._distance_to_exit()) - .003
        if np.array_equal(self.player, self.exit):
            reward += 2.0; self.success = True
        self.step_count += 1; done = self.success or self.step_count >= self.horizon
        return self.render(), float(reward), done, {'success': self.success,
            'steps': self.step_count, 'wall_hits': self.wall_hits,
            'initial_distance': self.initial_distance}
    def render(self):
        frame = np.zeros((1, SIZE, SIZE), dtype=np.uint8); scale = 2; margin = 5
        for row in range(self.side):
            for column in range(self.side):
                if self.maze[row,column]:
                    y, x = margin + row*scale, margin + column*scale
                    frame[0,y:y+scale,x:x+scale] = 55
        ey, ex = margin + int(self.exit[0])*scale, margin + int(self.exit[1])*scale
        py, px = margin + int(self.player[0])*scale, margin + int(self.player[1])*scale
        frame[0,ey:ey+scale,ex:ex+scale] = 175
        frame[0,py:py+scale,px:px+scale] = 255
        return frame


class PixelMazeMemory:
    """Online DFS over noise-tolerant visual identities; never receives the maze graph."""
    inverse = (1, 0, 3, 2)
    def __init__(self,action_dim:int=4): self.action_dim=action_dim;self.reset()
    def reset(self):
        self.nodes = {}; self.parent = {}; self.last_key = None; self.last_action = None
    @staticmethod
    def _key(observation: np.ndarray) -> str:
        return perceptual_state_key(observation)
    def act(self, observation: np.ndarray) -> int:
        key = self._key(observation); node = self.nodes.setdefault(key, {'tried': set(), 'edges': {}})
        previous=self.parent[key][1] if key in self.parent else None
        back=self.inverse[previous] if previous is not None and previous < len(self.inverse) else None
        candidates = [action for action in range(self.action_dim) if action not in node['tried'] and action != back]
        if candidates: action = candidates[0]
        elif back is not None: action = back
        else:
            remaining = [action for action in range(self.action_dim) if action not in node['tried']]
            action = remaining[0] if remaining else 0
        node['tried'].add(action); self.last_key = key; self.last_action = action
        return action
    def observe(self, next_observation: np.ndarray) -> None:
        nxt = self._key(next_observation); is_new = nxt not in self.nodes
        self.nodes.setdefault(nxt, {'tried': set(), 'edges': {}})
        self.nodes[self.last_key]['edges'][self.last_action] = nxt
        if nxt != self.last_key and is_new:
            self.parent[nxt] = (self.last_key, self.last_action)


def evaluate_pixel_maze_memory(*, episodes: int = 256, seed_base: int = 9_800_000) -> dict:
    rows = []
    for index in range(episodes):
        env = MazeGame(); obs = env.reset(seed_base + index); solver = PixelMazeMemory()
        total = 0.0
        while True:
            action = solver.act(obs); nxt, reward, done, info = env.step(action)
            solver.observe(nxt); obs = nxt; total += reward
            if done: break
        rows.append({'return': total, 'states_discovered': len(solver.nodes), **info})
    successes = [row for row in rows if row['success']]
    return {'episodes': episodes, 'mean_return': float(np.mean([row['return'] for row in rows])),
            'maze_success_rate': len(successes) / max(1, len(rows)),
            'steps_on_success': float(np.mean([row['steps'] for row in successes])) if successes else None,
            'worst_success_steps': max((row['steps'] for row in successes), default=None),
            'mean_states_discovered': float(np.mean([row['states_discovered'] for row in rows])),
            'wall_hits_per_episode': float(np.mean([row['wall_hits'] for row in rows]))}


def run_maze_memory_benchmark(output: Path) -> dict:
    random = evaluate_random(tasks()['maze'], episodes=256, seed_base=9_800_000)
    learned = evaluate_pixel_maze_memory(episodes=256, seed_base=9_800_000)
    report = {'format': 'wailah-pixel-maze-memory-v1',
              'input': '32x32 pixels, rewards, done, four actions and reversible-control pairs',
              'not_available_to_solver': ['player coordinates','exit coordinates','maze array',
                                          'maze graph','shortest path','distance shaping value'],
              'mechanism': 'online topological memory from frame identity and experienced transitions',
              'evaluation': '256 unseen procedurally generated perfect mazes',
              'random': random, 'pixel_memory': learned,
              'promoted': learned['maze_success_rate'] >= .90}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    if report['promoted']:
        atomic_json(output.parent/'living'/'strategies.json', {
            'format': 'wailah-strategy-skills-v1',
            'strategies': {'maze-memory': {
                'status': 'promoted', 'input': report['input'],
                'mechanism': report['mechanism'],
                'held_out_success_rate': learned['maze_success_rate'],
                'evaluation_mazes': learned['episodes']}}})
    return report


def render_maze_showcase(output: Path, *, seed: int = 9_812_337) -> Path:
    """Animate one unseen maze and the topology learned online from pixels."""
    from PIL import Image, ImageDraw, ImageFont
    env = MazeGame(); obs = env.reset(seed); solver = PixelMazeMemory(); rows = []
    while True:
        action = solver.act(obs); nxt, reward, done, info = env.step(action); solver.observe(nxt)
        rows.append((obs.copy(), action, reward, len(solver.nodes), dict(info)))
        obs = nxt
        if done: break
    scale = 7; panel = SIZE*scale; side = 250; font = ImageFont.load_default(); animation = []
    for index, (raw, action, reward, discovered, info) in enumerate(rows):
        canvas = Image.new('RGB', (panel+side,panel), (7,13,25))
        gray = raw[0]; rgb = np.stack((gray,gray,gray), axis=-1)
        image = Image.fromarray(rgb, mode='RGB').resize((panel,panel),Image.Resampling.NEAREST)
        canvas.paste(image,(0,0)); draw=ImageDraw.Draw(canvas); x=panel+16
        white=(230,237,247); muted=(135,151,174); green=(96,220,170)
        draw.text((x,16),'MAZE / PIXEL MEMORY',font=font,fill=green)
        draw.text((x,39),'no grid, path, or coordinates',font=font,fill=muted)
        draw.text((x,66),f'step: {index+1} / 256',font=font,fill=white)
        draw.text((x,88),f'action: {action}  reward: {reward:+.3f}',font=font,fill=white)
        draw.text((x,114),'ONLINE MEMORY',font=font,fill=muted)
        draw.text((x,132),f'visual states: {discovered}',font=font,fill=white)
        draw.text((x,151),f'wall probes: {info["wall_hits"]}',font=font,fill=white)
        draw.text((x,178),'status: ESCAPED' if info['success'] else 'status: exploring',
                  font=font,fill=green if info['success'] else white)
        draw.text((x,204),'topology learned during episode',font=font,fill=muted)
        animation.append(canvas)
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    animation[0].save(output,save_all=True,append_images=animation[1:],duration=85,
                      loop=0,optimize=False,disposal=2)
    return output


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    spec: TaskSpec
    game: type
    metric: str


def tasks() -> dict[str, BenchmarkTask]:
    common = dict(observation_shape=(1, SIZE, SIZE), replay_capacity=18_000,
                  batch_size=64, latent_dim=128, learning_rate=5e-4,
                  discount=.97, target_interval=100)
    return {
        'catch': BenchmarkTask('catch', TaskSpec('catch', action_dim=3, seed=101, **common), CatchGame, 'catch_rate'),
        'avoid': BenchmarkTask('avoid', TaskSpec('avoid', action_dim=3, seed=202, **common), AvoidGame, 'dodge_rate'),
        'navigate': BenchmarkTask('navigate', TaskSpec('navigate', action_dim=4, seed=303, **common), NavigateGame, 'goals_per_episode'),
        'keydoor': BenchmarkTask('keydoor', TaskSpec('keydoor', action_dim=4, seed=404, **common), KeyDoorGame, 'unlocks_per_episode'),
        'signals': BenchmarkTask('signals', TaskSpec('signals', action_dim=5, seed=505, **common), SignalGame, 'signal_accuracy'),
        'maze': BenchmarkTask('maze', TaskSpec('maze', action_dim=4, seed=606, **common), MazeGame, 'maze_success_rate'),
    }


def _play(skill: Skill, task: BenchmarkTask, seed: int, *, learning: bool,
          epsilon: float = 0.0) -> dict:
    env = task.game(); obs = env.reset(seed); total = 0.0
    for _ in range(env.horizon):
        action = skill.act(obs, epsilon=epsilon if learning else 0.0)
        nxt, reward, done, info = env.step(action)
        if learning: skill.observe(obs, action, reward, nxt, done)
        obs = nxt; total += reward
        if done: break
    return {'return': float(total), **info}


def train(skill: Skill, task: BenchmarkTask, guard: Guard | None, *, episodes: int,
          updates_per_episode: int, seed_base: int) -> dict:
    start = time.monotonic(); returns = []
    for episode in range(episodes):
        if guard is not None: guard.check()
        ratio = episode / max(1, episodes - 1)
        epsilon = 1.0 + ratio * (.05 - 1.0)
        row = _play(skill, task, seed_base + episode, learning=True, epsilon=epsilon)
        returns.append(row['return']); skill.learn(updates_per_episode)
    return {'episodes': episodes, 'updates': episodes * updates_per_episode,
            'mean_last_20_return': float(np.mean(returns[-20:])),
            'seconds': time.monotonic() - start}


def train_actor_critic(skill: Skill, task: BenchmarkTask, guard: Guard | None, *,
                       episodes: int, epochs_per_batch: int = 4, batch_episodes: int = 8,
                       seed_base: int) -> dict:
    """On-policy reward learning for skills with delayed, ordered objectives."""
    start = time.monotonic(); episode_returns = []; updates = 0
    all_obs, all_actions, all_returns = [], [], []
    skill.online.train()
    for episode in range(episodes):
        if guard is not None: guard.check()
        env = task.game(); obs = env.reset(seed_base + episode)
        observations, actions, rewards = [], [], []
        temperature = 1.35 + episode / max(1, episodes - 1) * (.35 - 1.35)
        for _ in range(env.horizon):
            tensor = torch.from_numpy(obs[None]).to(skill.device)
            with torch.no_grad():
                logits = skill.online(tensor)[0] / temperature
                distribution = torch.distributions.Categorical(logits=logits)
                action = int(distribution.sample())
            nxt, reward, done, _ = env.step(action)
            skill.observe(obs, action, reward, nxt, done)
            observations.append(obs); actions.append(action); rewards.append(reward); obs = nxt
            if done: break
        discounted = []; value = 0.0
        for reward in reversed(rewards):
            value = reward + skill.spec.discount * value; discounted.append(value)
        all_obs.extend(observations); all_actions.extend(actions)
        all_returns.extend(reversed(discounted)); episode_returns.append(float(sum(rewards)))
        if (episode + 1) % batch_episodes == 0 or episode + 1 == episodes:
            obs_t = torch.from_numpy(np.asarray(all_obs, dtype=np.uint8)).to(skill.device)
            action_t = torch.tensor(all_actions, dtype=torch.long, device=skill.device)
            return_t = torch.tensor(all_returns, dtype=torch.float32, device=skill.device)
            for _ in range(epochs_per_batch):
                logits = skill.online(obs_t); values = logits.mean(1)
                log_prob = logits.log_softmax(1).gather(1, action_t[:,None]).squeeze(1)
                probability = logits.softmax(1)
                entropy = -(probability * logits.log_softmax(1)).sum(1).mean()
                advantage = return_t - values
                policy_loss = -(log_prob * advantage.detach()).mean()
                value_loss = F.smooth_l1_loss(values, return_t)
                loss = policy_loss + .6 * value_loss - .012 * entropy
                skill.optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(skill.online.parameters(), 5.0)
                skill.optimizer.step(); updates += 1; skill.state['updates'] += 1
            skill.target.load_state_dict(skill.online.state_dict())
            all_obs.clear(); all_actions.clear(); all_returns.clear()
    return {'algorithm': 'pixel actor-critic with full reward returns',
            'episodes': episodes, 'updates': updates,
            'mean_last_20_return': float(np.mean(episode_returns[-20:])),
            'seconds': time.monotonic() - start}


def evaluate(skill: Skill, task: BenchmarkTask, *, episodes: int = 64,
             seed_base: int = 9_100_000) -> dict:
    rows = [_play(skill, task, seed_base + index, learning=False) for index in range(episodes)]
    result = {'episodes': episodes, 'mean_return': float(np.mean([r['return'] for r in rows]))}
    if task.name == 'catch':
        caught = sum(r['caught'] for r in rows); total = caught + sum(r['missed'] for r in rows)
        result['catch_rate'] = caught / max(1, total)
    elif task.name == 'avoid':
        dodged = sum(r['dodged'] for r in rows); total = dodged + sum(r['collisions'] for r in rows)
        result['dodge_rate'] = dodged / max(1, total)
    elif task.name == 'navigate':
        result['goals_per_episode'] = float(np.mean([r['goals'] for r in rows]))
    elif task.name == 'keydoor':
        result['unlocks_per_episode'] = float(np.mean([r['unlocks'] for r in rows]))
        result['keys_per_episode'] = float(np.mean([r['keys'] for r in rows]))
        result['hazard_hits_per_episode'] = float(np.mean([r['hazard_hits'] for r in rows]))
    elif task.name == 'signals':
        correct = sum(r['correct'] for r in rows); attempts = correct + sum(r['errors'] for r in rows)
        result['signal_accuracy'] = correct / max(1, attempts)
    else:
        successes = [r for r in rows if r['success']]
        result['maze_success_rate'] = len(successes) / max(1, len(rows))
        result['steps_on_success'] = float(np.mean([r['steps'] for r in successes])) if successes else None
        result['wall_hits_per_episode'] = float(np.mean([r['wall_hits'] for r in rows]))
    return result


def evaluate_random(task: BenchmarkTask, *, episodes: int = 64,
                    seed_base: int = 9_100_000) -> dict:
    rows = []
    for index in range(episodes):
        seed = seed_base + index; rng = np.random.default_rng(seed + 44_009)
        env = task.game(); env.reset(seed); total = 0.0
        for _ in range(env.horizon):
            _, reward, done, info = env.step(int(rng.integers(task.spec.action_dim)))
            total += reward
            if done: break
        rows.append({'return': total, **info})
    result = {'episodes': episodes, 'mean_return': float(np.mean([r['return'] for r in rows]))}
    if task.name == 'catch':
        caught = sum(r['caught'] for r in rows); total = caught + sum(r['missed'] for r in rows)
        result['catch_rate'] = caught / max(1, total)
    elif task.name == 'avoid':
        dodged = sum(r['dodged'] for r in rows); total = dodged + sum(r['collisions'] for r in rows)
        result['dodge_rate'] = dodged / max(1, total)
    elif task.name == 'navigate':
        result['goals_per_episode'] = float(np.mean([r['goals'] for r in rows]))
    elif task.name == 'keydoor':
        result['unlocks_per_episode'] = float(np.mean([r['unlocks'] for r in rows]))
        result['keys_per_episode'] = float(np.mean([r['keys'] for r in rows]))
        result['hazard_hits_per_episode'] = float(np.mean([r['hazard_hits'] for r in rows]))
    elif task.name == 'signals':
        correct = sum(r['correct'] for r in rows); attempts = correct + sum(r['errors'] for r in rows)
        result['signal_accuracy'] = correct / max(1, attempts)
    else:
        successes = [r for r in rows if r['success']]
        result['maze_success_rate'] = len(successes) / max(1, len(rows))
        result['steps_on_success'] = float(np.mean([r['steps'] for r in successes])) if successes else None
        result['wall_hits_per_episode'] = float(np.mean([r['wall_hits'] for r in rows]))
    return result


def run_keydoor_extension(workspace: Path, output: Path,
                          device: torch.device | str = 'cpu', guard: Guard | None = None,
                          *, episodes: int = 420, updates_per_episode: int = 14) -> dict:
    """Grow a fourth two-stage skill and prove that prior skills survive unchanged."""
    workspace, output = Path(workspace), Path(output)
    bank = LifelongSkillBank(workspace, device); catalogue = tasks(); task = catalogue['keydoor']
    old_ids = sorted(task_id for task_id in bank.registry['tasks'] if task_id != 'keydoor')
    if not old_ids: raise RuntimeError('Train the continual benchmark before adding KeyDoor.')
    old_hashes = {task_id: hashlib.sha256((bank.skills/task_id/'skill.pt').read_bytes()).hexdigest()
                  for task_id in old_ids}
    random = evaluate_random(task, episodes=128, seed_base=9_300_000)
    skill = bank.open(task.spec)
    before = evaluate(skill, task, episodes=64, seed_base=9_300_000)
    training = train_actor_critic(skill, task, guard, episodes=episodes,
                                  epochs_per_batch=max(2, updates_per_episode // 3),
                                  seed_base=8_800_000)
    after = evaluate(skill, task, episodes=128, seed_base=9_300_000)
    event = bank.consolidate(skill, after[task.metric])
    new_hashes = {task_id: hashlib.sha256((bank.skills/task_id/'skill.pt').read_bytes()).hexdigest()
                  for task_id in old_ids}
    retention = {}
    for task_id in old_ids:
        known = catalogue[task_id]; retained = bank.open(known.spec)
        retention[task_id] = evaluate(retained, known, episodes=64, seed_base=9_100_000)
    from .living import LivingSystem
    living = LivingSystem(workspace, device)
    organism = living.build(router_updates=800, world_updates=600)
    routed = []
    for index in range(128):
        env = KeyDoorGame(); obs = env.reset(9_400_000 + index)
        for _ in range(env.horizon):
            action, meta = living.act(obs, 4, top_k=1)
            routed.append(meta['routes'][0]['expert'])
            obs, _, done, _ = env.step(action)
            if done: break
    report = {'format': 'wailah-keydoor-extension-v1',
              'task': 'find key, then unlock door, while avoiding four hazards',
              'observation': '32x32 pixels only', 'random': random,
              'untrained_transferred_policy': before, 'training': training,
              'learned': after, 'consolidation': event,
              'relative_to_random': after[task.metric] / max(1e-9, random[task.metric]),
              'old_skill_hashes_unchanged': old_hashes == new_hashes,
              'old_skill_hashes_before': old_hashes, 'old_skill_hashes_after': new_hashes,
              'retention_after_keydoor': retention, 'organism': organism,
              'keydoor_router_frame_accuracy': routed.count('keydoor') / max(1, len(routed))}
    output.parent.mkdir(parents=True, exist_ok=True); atomic_json(output, report); return report


def run_signal_extension(workspace: Path, output: Path,
                         device: torch.device | str = 'cpu', guard: Guard | None = None,
                         *, episodes: int = 220, updates_per_episode: int = 12) -> dict:
    """Learn a noisy five-way visual action language and expand the organism."""
    workspace, output = Path(workspace), Path(output)
    bank = LifelongSkillBank(workspace, device); catalogue = tasks(); task = catalogue['signals']
    protected = [task_id for task_id in ('catch','avoid','navigate') if task_id in bank.registry['tasks']]
    hashes_before = {task_id: hashlib.sha256((bank.skills/task_id/'skill.pt').read_bytes()).hexdigest()
                     for task_id in protected}
    random = evaluate_random(task, episodes=128, seed_base=9_600_000)
    skill = bank.open(task.spec); before = evaluate(skill, task, episodes=64, seed_base=9_600_000)
    training = train(skill, task, guard, episodes=episodes,
                     updates_per_episode=updates_per_episode, seed_base=8_950_000)
    learned = evaluate(skill, task, episodes=128, seed_base=9_600_000)
    event = bank.consolidate(skill, learned[task.metric])
    hashes_after = {task_id: hashlib.sha256((bank.skills/task_id/'skill.pt').read_bytes()).hexdigest()
                    for task_id in protected}
    retention = {}
    for task_id in protected:
        known = catalogue[task_id]; retention[task_id] = evaluate(bank.open(known.spec), known)
    from .living import LivingSystem
    living = LivingSystem(workspace, device); organism = living.build(800, 700)
    route_correct = route_total = 0
    for index in range(128):
        env = SignalGame(); obs = env.reset(9_700_000 + index)
        for _ in range(env.horizon):
            action, meta = living.act(obs, 5, top_k=1)
            route_correct += int(meta['routes'][0]['expert'] == 'signals'); route_total += 1
            obs, _, done, _ = env.step(action)
            if done: break
    report = {'format': 'wailah-signal-extension-v1',
              'task': 'infer a hidden five-symbol action mapping under position and pixel noise',
              'mapping_was_not_given_to_learner': True, 'random': random, 'before': before,
              'training': training, 'learned': learned, 'consolidation': event,
              'accuracy_gain': learned['signal_accuracy'] - random['signal_accuracy'],
              'relative_to_random': learned['signal_accuracy'] / max(1e-9, random['signal_accuracy']),
              'protected_skill_hashes_unchanged': hashes_before == hashes_after,
              'protected_skill_hashes_before': hashes_before,
              'protected_skill_hashes_after': hashes_after,
              'retention_after_signals': retention, 'organism': organism,
              'signal_router_frame_accuracy': route_correct / max(1, route_total)}
    output.parent.mkdir(parents=True, exist_ok=True); atomic_json(output, report); return report


def run_maze_extension(workspace: Path, output: Path,
                       device: torch.device | str = 'cpu', guard: Guard | None = None,
                       *, episodes: int = 360, updates_per_episode: int = 14) -> dict:
    """Learn to escape unseen procedural mazes and promote only above threshold."""
    workspace, output = Path(workspace), Path(output)
    bank = LifelongSkillBank(workspace, device); catalogue = tasks(); task = catalogue['maze']
    protected = [task_id for task_id in ('catch','avoid','navigate','signals')
                 if task_id in bank.registry['tasks']]
    hashes_before = {task_id: hashlib.sha256((bank.skills/task_id/'skill.pt').read_bytes()).hexdigest()
                     for task_id in protected}
    random = evaluate_random(task, episodes=256, seed_base=9_800_000)
    skill = bank.open(task.spec); before = evaluate(skill, task, episodes=64, seed_base=9_800_000)
    training = train(skill, task, guard, episodes=episodes,
                     updates_per_episode=updates_per_episode, seed_base=8_990_000)
    learned = evaluate(skill, task, episodes=256, seed_base=9_800_000)
    threshold = max(.70, random['maze_success_rate'] + .30)
    promoted = learned['maze_success_rate'] >= threshold
    if promoted:
        event = bank.consolidate(skill, learned[task.metric])
    else:
        skill.save(); event = bank.quarantine('maze',
            f"Held-out success {learned['maze_success_rate']:.3f} below promotion threshold {threshold:.3f}")
    hashes_after = {task_id: hashlib.sha256((bank.skills/task_id/'skill.pt').read_bytes()).hexdigest()
                    for task_id in protected}
    retention = {}
    for task_id in protected:
        known = catalogue[task_id]; retention[task_id] = evaluate(bank.open(known.spec), known)
    organism = None; route_accuracy = None
    if promoted:
        from .living import LivingSystem
        living = LivingSystem(workspace, device); organism = living.build(900, 800)
        correct = total = 0
        for index in range(128):
            env = MazeGame(); obs = env.reset(9_900_000 + index)
            while True:
                action, meta = living.act(obs, 4, top_k=1)
                correct += int(meta['routes'][0]['expert'] == 'maze'); total += 1
                obs, _, done, _ = env.step(action)
                if done: break
        route_accuracy = correct / max(1, total)
    report = {'format': 'wailah-maze-extension-v1',
              'task': 'escape procedurally generated perfect mazes',
              'policy_input': '32x32 pixels only; no coordinates, maze graph, path or solver output',
              'evaluation': '256 unseen deterministic maze seeds', 'random': random,
              'before': before, 'training': training, 'learned': learned,
              'promotion_threshold': threshold, 'promoted': promoted, 'event': event,
              'success_gain': learned['maze_success_rate'] - random['maze_success_rate'],
              'protected_skill_hashes_unchanged': hashes_before == hashes_after,
              'protected_skill_hashes_before': hashes_before,
              'protected_skill_hashes_after': hashes_after,
              'retention_after_maze': retention, 'organism': organism,
              'maze_router_frame_accuracy': route_accuracy}
    output.parent.mkdir(parents=True, exist_ok=True); atomic_json(output, report); return report


def _skill_hash(skill: Skill) -> str:
    return hashlib.sha256(skill.path.read_bytes()).hexdigest()


def run_real_benchmark(workspace: Path, device: torch.device | str = 'cpu',
                       guard: Guard | None = None, *, scale: int = 1) -> dict:
    """Train Catch -> Avoid -> Navigate -> Catch and retain the full score matrix."""
    if scale < 1: raise ValueError('scale must be positive.')
    workspace = Path(workspace)
    report_path = workspace / 'lifelong_real_benchmark.json'
    if report_path.exists():
        raise FileExistsError('Use a new benchmark workspace; this run is intentionally append-free.')
    bank = LifelongSkillBank(workspace, device); catalogue = tasks()
    random_baselines = {name: evaluate_random(task) for name, task in catalogue.items()}
    schedule = [('catch', 45 * scale, 8), ('avoid', 140 * scale, 10),
                ('navigate', 180 * scale, 12), ('catch', 160 * scale, 10)]
    learned = []; phases = []; catch_hash_before_other_tasks = None
    for phase_index, (name, episodes, updates) in enumerate(schedule):
        task = catalogue[name]; skill = bank.open(task.spec)
        before = evaluate(skill, task, episodes=32, seed_base=9_000_000) if skill.path.exists() else None
        training = train(skill, task, guard, episodes=episodes, updates_per_episode=updates,
                         seed_base=8_000_000 + phase_index * 100_000)
        after = evaluate(skill, task, episodes=64, seed_base=9_100_000)
        metric_score = after[task.metric]
        event = bank.consolidate(skill, metric_score)
        if name not in learned: learned.append(name)
        matrix = {}
        for known in learned:
            known_task = catalogue[known]; known_skill = bank.open(known_task.spec)
            matrix[known] = evaluate(known_skill, known_task, episodes=64, seed_base=9_100_000)
        if phase_index == 0: catch_hash_before_other_tasks = _skill_hash(skill)
        phases.append({'phase': phase_index + 1, 'task': name, 'before': before,
                       'training': training, 'after': after, 'consolidation': event,
                       'retention_matrix': matrix})
    catch_final = bank.open(catalogue['catch'].spec)
    report = {'format': 'wailah-real-continual-benchmark-v1',
              'curriculum': [x[0] for x in schedule], 'scale': scale,
              'observation': '32x32 pixels only; private environment state never enters a policy',
              'training_signal': 'discrete actions, observed rewards and episode endings',
              'random_baselines': random_baselines,
              'phases': phases,
              'catch_checkpoint_was_isolated_until_revisit':
                  phases[1]['retention_matrix']['catch'] == phases[0]['retention_matrix']['catch'] and
                  phases[2]['retention_matrix']['catch'] == phases[0]['retention_matrix']['catch'],
              'catch_first_score': phases[0]['after']['catch_rate'],
              'catch_before_revisit': phases[2]['retention_matrix']['catch']['catch_rate'],
              'catch_after_revisit': phases[3]['after']['catch_rate'],
              'catch_revisit_gain': phases[3]['after']['catch_rate'] - phases[0]['after']['catch_rate'],
              'final_skill_count': len(bank.registry['tasks']),
              'final_catch_hash': _skill_hash(catch_final),
              'initial_catch_hash': catch_hash_before_other_tasks}
    atomic_json(report_path, report)
    return report


def render_showcase(workspace: Path, output: Path,
                    device: torch.device | str = 'cpu') -> Path:
    """Render synchronized held-out play from all three committed skills."""
    from PIL import Image, ImageDraw, ImageFont
    bank = LifelongSkillBank(workspace, device); catalogue = tasks()
    colors = {'catch': (50, 205, 220), 'avoid': (240, 95, 85), 'navigate': (95, 220, 140)}
    streams = {}
    for offset, name in enumerate(('catch', 'avoid', 'navigate')):
        task = catalogue[name]; skill = bank.open(task.spec); env = task.game()
        obs = env.reset(9_200_000 + offset); frames = []
        for _ in range(env.horizon):
            frames.append(obs.copy()); action = skill.act(obs)
            obs, _, done, _ = env.step(action)
            if done: break
        streams[name] = frames
    font = ImageFont.load_default(); panels = 3; scale = 5
    panel_w = SIZE * scale; header = 24
    animation = []
    for index in range(max(len(x) for x in streams.values())):
        canvas = Image.new('RGB', (panel_w * panels, panel_w + header), (8, 15, 28))
        draw = ImageDraw.Draw(canvas)
        for column, name in enumerate(('catch', 'avoid', 'navigate')):
            raw = streams[name][min(index, len(streams[name]) - 1)][0]
            color = np.asarray(colors[name], dtype=np.float32)
            rgb = (raw[..., None].astype(np.float32) / 255.0 * color).astype(np.uint8)
            image = Image.fromarray(rgb, mode='RGB').resize((panel_w, panel_w), Image.Resampling.NEAREST)
            canvas.paste(image, (column * panel_w, header))
            draw.text((column * panel_w + 7, 7), name.upper(), font=font, fill=colors[name])
        animation.append(canvas)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation[0].save(output, save_all=True, append_images=animation[1:], duration=85,
                      loop=0, optimize=False, disposal=2)
    return output


def render_signal_showcase(workspace: Path, output: Path,
                           device: torch.device | str = 'cpu') -> Path:
    """Animate held-out noisy symbols, learned actions and outcomes."""
    from PIL import Image, ImageDraw, ImageFont
    bank = LifelongSkillBank(workspace, device); task = tasks()['signals']
    skill = bank.open(task.spec); env = SignalGame(horizon=64); obs = env.reset(9_888_121)
    frames = []; running = 0
    for step in range(env.horizon):
        symbol = env.symbol; action = skill.act(obs); expected = env.mapping[symbol]
        nxt, reward, done, _ = env.step(action); running += int(action == expected)
        frames.append((obs.copy(), step, symbol, action, expected, reward, running))
        obs = nxt
        if done: break
    scale = 7; panel = SIZE * scale; side = 250; font = ImageFont.load_default()
    animation = []; names = ('vertical', 'horizontal', 'cross-x', 'box', 'plus')
    for raw, step, symbol, action, expected, reward, running in frames:
        canvas = Image.new('RGB', (panel + side, panel), (7, 13, 25))
        image = Image.fromarray(raw[0], mode='L').convert('RGB').resize(
            (panel, panel), Image.Resampling.NEAREST)
        canvas.paste(image, (0, 0)); draw = ImageDraw.Draw(canvas); x = panel + 16
        white = (230, 237, 247); muted = (135, 151, 174); green = (96, 220, 170)
        draw.text((x, 16), 'NEW SKILL / SIGNALS', font=font, fill=green)
        draw.text((x, 39), 'input: noisy pixels only', font=font, fill=muted)
        draw.text((x, 62), f'trial {step + 1:02d} / 64', font=font, fill=white)
        draw.text((x, 88), f'visual class: {names[symbol]}', font=font, fill=white)
        draw.text((x, 114), 'LEARNED RESPONSE', font=font, fill=muted)
        draw.text((x, 132), f'action: {action}', font=font,
                  fill=green if action == expected else (245,95,90))
        draw.text((x, 151), f'reward: {reward:+.2f}', font=font, fill=white)
        draw.text((x, 177), f'accuracy: {running}/{step + 1} ({running/(step+1):.0%})',
                  font=font, fill=white)
        draw.text((x, 203), 'mapping learned from reward', font=font, fill=muted)
        animation.append(canvas)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation[0].save(output, save_all=True, append_images=animation[1:], duration=180,
                      loop=0, optimize=False, disposal=2)
    return output
