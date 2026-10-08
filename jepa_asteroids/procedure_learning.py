"""Learn a named multi-stage rescue procedure from visual demonstrations."""
from __future__ import annotations

from collections import Counter, deque
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .runtime import atomic_json
from .strategy_selector import LearnedStrategySelector, probe_features


GRID, CELL = 11, 8
BACKGROUND = np.asarray((8, 12, 18), dtype=np.uint8)
WALL = np.asarray((72, 78, 88), dtype=np.uint8)
AGENT = np.asarray((242, 244, 248), dtype=np.uint8)
ACCESS = np.asarray((238, 204, 45), dtype=np.uint8)
GATE = np.asarray((224, 62, 68), dtype=np.uint8)
STUCK = np.asarray((48, 198, 226), dtype=np.uint8)
EXIT = np.asarray((55, 218, 105), dtype=np.uint8)
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))


class RescueWorld:
    """A sparse-reward world whose solution is an ordered reusable procedure."""
    action_dim = 5
    horizon = 128

    def reset(self, seed: int):
        rng = np.random.default_rng(seed); self.rng = rng
        self.gate = (int(rng.integers(2, 9)), 5)
        left = [(y, x) for y in range(1, 10) for x in range(1, 5)]
        right = [(y, x) for y in range(1, 10) for x in range(6, 10)]
        picks = rng.choice(len(left), 2, replace=False); self.agent, self.access = [left[int(i)] for i in picks]
        picks = rng.choice(len(right), 2, replace=False); self.stuck, self.exit = [right[int(i)] for i in picks]
        self.has_access = self.gate_open = self.rescued = self.success = False
        self.step_count = 0; return self.render()

    def _wall(self, position):
        y, x = position
        return y in (0, GRID - 1) or x in (0, GRID - 1) or (x == 5 and position != self.gate)

    @staticmethod
    def _adjacent(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def step(self, action: int):
        action = int(action); self.step_count += 1
        if action < 4:
            dy, dx = MOVES[action]; nxt = (self.agent[0] + dy, self.agent[1] + dx)
            blocked = self._wall(nxt) or (not self.gate_open and nxt == self.gate) or (
                not self.rescued and nxt == self.stuck)
            if not blocked:
                self.agent = nxt
                if not self.has_access and self.agent == self.access: self.has_access = True
                if self.rescued and self.agent == self.exit: self.success = True
        else:
            if self.has_access and not self.gate_open and self._adjacent(self.agent, self.gate):
                self.gate_open = True
            elif self.gate_open and not self.rescued and self._adjacent(self.agent, self.stuck):
                self.rescued = True
        done = self.success or self.step_count >= self.horizon
        reward = 1.0 if self.success else 0.0
        return self.render(), reward, done, {'success': self.success, 'steps': self.step_count}

    def render(self):
        rng = np.random.default_rng((self.step_count + 1) * 997 + int(self.agent[0] * 31 + self.agent[1]))
        frame = rng.integers(0, 5, (3, GRID * CELL, GRID * CELL), dtype=np.uint8)
        frame += BACKGROUND[:, None, None]
        def paint(position, color):
            y, x = position; frame[:, y * CELL + 1:(y + 1) * CELL - 1,
                                   x * CELL + 1:(x + 1) * CELL - 1] = color[:, None, None]
        for y in range(GRID):
            for x in range(GRID):
                if self._wall((y, x)): paint((y, x), WALL)
        if not self.has_access: paint(self.access, ACCESS)
        if not self.gate_open: paint(self.gate, GATE)
        if not self.rescued: paint(self.stuck, STUCK)
        paint(self.exit, EXIT); paint(self.agent, AGENT)
        return frame


def _cells(frame):
    rgb = np.asarray(frame, dtype=np.float32)
    return np.asarray([[rgb[:, y * CELL + 1:(y + 1) * CELL - 1,
                                x * CELL + 1:(x + 1) * CELL - 1].mean((1, 2))
                        for x in range(GRID)] for y in range(GRID)])


def _bfs(start, goals, blocked):
    queue = deque([start]); previous = {start: None}; move_to = {}
    while queue:
        point = queue.popleft()
        if point in goals:
            actions = []
            while previous[point] is not None:
                actions.append(move_to[point]); point = previous[point]
            return list(reversed(actions))
        for action, (dy, dx) in enumerate(MOVES):
            nxt = (point[0] + dy, point[1] + dx)
            if 0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID and nxt not in blocked and nxt not in previous:
                previous[nxt] = point; move_to[nxt] = action; queue.append(nxt)
    raise RuntimeError('no visual route to the next procedure event')


def _oracle_actions(env: RescueWorld):
    actions = []
    def route(goal, adjacent=False):
        blocked = {(y, x) for y in range(GRID) for x in range(GRID) if env._wall((y, x))}
        if not env.gate_open: blocked.add(env.gate)
        if not env.rescued: blocked.add(env.stuck)
        goals = ({(goal[0] + dy, goal[1] + dx) for dy, dx in MOVES} - blocked) if adjacent else {goal}
        return _bfs(env.agent, goals, blocked)
    for target, mode in ((env.access, 'enter'), (env.gate, 'use'), (env.stuck, 'use'), (env.exit, 'enter')):
        plan = route(target, adjacent=mode == 'use')
        for action in plan: env.step(action); actions.append(action)
        if mode == 'use': env.step(4); actions.append(4)
    return actions


def demonstration(seed):
    env = RescueWorld(); first = env.reset(seed)
    # Re-run the oracle while recording the visual stream seen by the learner.
    oracle = RescueWorld(); oracle.reset(seed); actions = _oracle_actions(oracle)
    env = RescueWorld(); frames = [env.reset(seed)]; rewards = []
    for action in actions:
        frame, reward, done, _ = env.step(action); frames.append(frame); rewards.append(reward)
    return {'frames': frames, 'actions': actions, 'rewards': rewards}


class ProcedureMemory:
    """Stores an ordered sequence of visually inferred event targets and interactions."""
    def __init__(self, name=None, events=None, agent_color=None, wall_color=None, background_color=None):
        self.name = name; self.events = events or []
        self.agent_color = None if agent_color is None else np.asarray(agent_color, dtype=np.float32)
        self.wall_color = None if wall_color is None else np.asarray(wall_color, dtype=np.float32)
        self.background_color = None if background_color is None else np.asarray(background_color, dtype=np.float32)

    @staticmethod
    def _nearest(grid, color):
        distance = ((grid - color[None, None]) ** 2).mean(2)
        index = int(np.argmin(distance)); return divmod(index, GRID), float(distance.flat[index])

    def learn(self, name, demos):
        sequences = []
        for demo in demos:
            grids = [_cells(frame) for frame in demo['frames']]
            sums = grids[0].sum(2); agent = divmod(int(np.argmax(sums)), GRID)
            agent_color = grids[0][agent]
            flattened = np.round(grids[0].reshape(-1, 3), -1)
            counts = Counter(map(tuple, flattened.astype(int)))
            common = [np.asarray(value, dtype=np.float32) for value, _ in counts.most_common(2)]
            common.sort(key=lambda value: float(value.sum()))
            background, wall = common[0], common[-1]
            events = []
            for index, action in enumerate(demo['actions']):
                before, after = grids[index], grids[index + 1]
                before_agent, _ = self._nearest(before, agent_color); after_agent, _ = self._nearest(after, agent_color)
                candidate = None; mode = None
                if action < 4 and after_agent != before_agent:
                    color = before[after_agent]
                    persistent_change = bool(demo['rewards'][index] > 0)
                    if not persistent_change:
                        # A destination can be temporarily hidden under the
                        # moving agent. Only call it an event if it stays gone
                        # after the agent leaves that cell.
                        for later in range(index + 2, len(grids)):
                            later_agent, _ = self._nearest(grids[later], agent_color)
                            if later_agent != after_agent:
                                persistent_change = np.mean((grids[later][after_agent] - background) ** 2) < 400
                                break
                    if (persistent_change and np.mean((color - background) ** 2) > 400 and
                            np.mean((color - wall) ** 2) > 400): candidate, mode = color, 'enter'
                elif action == 4:
                    for dy, dx in MOVES:
                        point = (before_agent[0] + dy, before_agent[1] + dx)
                        if 0 <= point[0] < GRID and 0 <= point[1] < GRID:
                            if np.mean((before[point] - after[point]) ** 2) > 1000:
                                candidate, mode = before[point], 'use'; break
                if candidate is not None and all(np.mean((candidate - row['prototype']) ** 2) > 400 for row in events):
                    events.append({'prototype': candidate, 'mode': mode})
            if len(events) != 4: raise RuntimeError(f'demonstration exposed {len(events)} events, expected four')
            sequences.append((events, agent_color, wall, background))
        self.name = str(name)
        self.events = [{'prototype': np.mean([seq[0][i]['prototype'] for seq in sequences], axis=0),
                        'mode': sequences[0][0][i]['mode']} for i in range(4)]
        self.agent_color = np.mean([seq[1] for seq in sequences], axis=0)
        self.wall_color = np.mean([seq[2] for seq in sequences], axis=0)
        self.background_color = np.mean([seq[3] for seq in sequences], axis=0)
        return self.explain()

    def explain(self):
        return {'procedure': self.name, 'learned_steps': [
            {'order': index + 1, 'interaction': event['mode'],
             'visual_signature_rgb': [int(round(x)) for x in event['prototype']]}
            for index, event in enumerate(self.events)],
            'source': 'ordered visual changes in successful demonstrations'}

    def save(self, path):
        atomic_json(path, {'format': 'wailah-visual-procedure-v1', 'name': self.name,
            'events': [{'prototype': row['prototype'].tolist(), 'mode': row['mode']} for row in self.events],
            'agent_color': self.agent_color.tolist(), 'wall_color': self.wall_color.tolist(),
            'background_color': self.background_color.tolist(), 'explanation': self.explain()})

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        events = [{'prototype': np.asarray(row['prototype']), 'mode': row['mode']} for row in value['events']]
        return cls(value['name'], events, value['agent_color'], value['wall_color'], value['background_color'])


class ProcedureStrategy:
    def __init__(self, memory): self.memory = memory; self.index = 0
    @classmethod
    def load(cls, workspace): return cls(ProcedureMemory.load(Path(workspace) / 'rescue.json'))
    def reset(self, observation): self.index = 0

    def act(self, observation):
        grid = _cells(observation)
        agent, _ = self.memory._nearest(grid, self.memory.agent_color)
        while self.index < len(self.memory.events):
            event = self.memory.events[self.index]; target, error = self.memory._nearest(grid, event['prototype'])
            if error < 500: break
            # The agent may already be standing on the final destination when
            # the stranded object becomes attached. Step away once so the
            # destination becomes visible, then deliberately re-enter it.
            if self.index == len(self.memory.events) - 1 and event['mode'] == 'enter':
                wall_error = ((grid - self.memory.wall_color[None, None]) ** 2).mean(2)
                for action, (dy, dx) in enumerate(MOVES):
                    point = (agent[0] + dy, agent[1] + dx)
                    if (0 <= point[0] < GRID and 0 <= point[1] < GRID and
                            wall_error[point] >= 500): return action
            self.index += 1
        if self.index >= len(self.memory.events): return 0
        event = self.memory.events[self.index]; target, _ = self.memory._nearest(grid, event['prototype'])
        wall_error = ((grid - self.memory.wall_color[None, None]) ** 2).mean(2)
        blocked = {tuple(x) for x in np.argwhere(wall_error < 500)}
        # Any not-yet-completed use target is an obstacle until its learned interaction occurs.
        for later in self.memory.events[self.index:]:
            if later['mode'] == 'use':
                point, error = self.memory._nearest(grid, later['prototype'])
                if error < 500: blocked.add(point)
        if event['mode'] == 'use':
            if abs(agent[0] - target[0]) + abs(agent[1] - target[1]) == 1: return 4
            goals = {(target[0] + dy, target[1] + dx) for dy, dx in MOVES}
            goals = {point for point in goals if point not in blocked}
        else: goals = {target}; blocked.discard(target)
        plan = _bfs(agent, goals, blocked); return plan[0] if plan else (4 if event['mode'] == 'use' else 0)


def _evaluate(strategy_factory, episodes, seed_base):
    success = 0; steps = []
    for episode in range(episodes):
        env = RescueWorld(); obs = env.reset(seed_base + episode); strategy = strategy_factory(); strategy.reset(obs)
        while True:
            obs, reward, done, info = env.step(strategy.act(obs))
            if done: break
        success += int(reward > 0); steps.append(info['steps'])
    return {'episodes': episodes, 'success_rate': success / episodes,
            'mean_steps': float(np.mean(steps)), 'max_steps': int(max(steps))}


def _random(episodes, seed_base):
    success = 0
    for episode in range(episodes):
        env = RescueWorld(); obs = env.reset(seed_base + episode); rng = np.random.default_rng(seed_base + episode)
        while True:
            obs, reward, done, _ = env.step(int(rng.integers(env.action_dim)))
            if done: break
        success += int(reward > 0)
    return {'episodes': episodes, 'success_rate': success / episodes}


def run_procedure_benchmark(workspace: Path, output: Path):
    workspace, output = Path(workspace), Path(output)
    brain = workspace / 'living' / 'procedures'; brain.mkdir(parents=True, exist_ok=True)
    old_hashes = {str(path.relative_to(workspace)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (workspace / 'living').rglob('*')
                  if path.is_file() and path.name != 'strategy_selector.json' and
                  'procedures' not in path.relative_to(workspace).parts}
    demos = [demonstration(150_000_000 + index) for index in range(3)]
    memory = ProcedureMemory(); explanation = memory.learn('rescue', demos)
    memory.save(brain / 'rescue.json')
    evaluation = _evaluate(lambda: ProcedureStrategy(memory), 256, 150_100_000)
    random = _random(256, 150_200_000)
    loaded = ProcedureMemory.load(brain / 'rescue.json')
    revisit = _evaluate(lambda: ProcedureStrategy(loaded), 128, 150_300_000)
    selector_path = workspace / 'living' / 'strategy_selector.json'; selector = LearnedStrategySelector.load(selector_path)
    factory = lambda: RescueWorld()
    features = np.stack([probe_features(factory, 150_400_000 + i) for i in range(64)])
    selector.centroids['procedure-rescue-memory'] = features.mean(0)
    selector.scale = np.maximum(selector.scale, features.std(0)); selector.save(selector_path)
    from .autonomy import AutonomousCompetenceLoop
    loop = AutonomousCompetenceLoop(workspace, 'cpu', procedure_workspace=brain)
    decision = loop.select(factory, probe_seed=150_500_000)
    competence = loop.evaluate_selected(factory, decision['method'], seed_base=150_600_000, episodes=128)
    new_hashes = {key: hashlib.sha256((workspace / key).read_bytes()).hexdigest() for key in old_hashes}
    report = {'format': 'wailah-named-procedure-v6', 'name': 'rescue',
        'learner_inputs': 'pixel streams, demonstrated motor actions, terminal reward and done',
        'demonstrations': {'count': len(demos), 'successful': sum(row['rewards'][-1] > 0 for row in demos),
                           'mean_actions': float(np.mean([len(row['actions']) for row in demos]))},
        'learned_explanation': explanation, 'unseen_layouts': evaluation, 'random_baseline': random,
        'persistence': {'reloaded_unseen_layouts': revisit, 'older_memories_unchanged': old_hashes == new_hashes},
        'autonomous_integration': {'decision': decision, 'competence': competence},
        'passed': bool(evaluation['success_rate'] == revisit['success_rate'] == 1.0 and
                       decision['method'] == 'procedure-rescue-memory' and
                       competence.get('success_rate', 0) == 1.0 and old_hashes == new_hashes)}
    atomic_json(output, report); return report


def render_procedure_summary(report_path: Path, output: Path):
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    try:
        title = ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf', 27)
        body = ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf', 15)
        small = ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf', 12)
    except OSError: title = body = small = ImageFont.load_default()
    canvas = Image.new('RGB', (1120, 630), (7, 13, 25)); draw = ImageDraw.Draw(canvas)
    green, white, muted, panel = (75, 225, 164), (232, 239, 248), (139, 156, 180), (14, 25, 43)
    draw.text((34, 24), 'WAILAH  /  LEARNED PROCEDURE: RESCUE', font=title, fill=white)
    draw.text((35, 64), 'watch success → infer event order → plan through new layouts → remember', font=body, fill=muted)
    cards = [('DEMONSTRATIONS', report['demonstrations']['count']),
             ('UNSEEN RESCUES', report['unseen_layouts']['success_rate']),
             ('RELOADED MEMORY', report['persistence']['reloaded_unseen_layouts']['success_rate']),
             ('AUTONOMOUS GATE', report['autonomous_integration']['competence']['success_rate'])]
    for index, (label, value) in enumerate(cards):
        x = 34 + index * 270; draw.rounded_rectangle((x, 105, x + 246, 197), radius=12, fill=panel)
        draw.text((x + 16, 121), label, font=small, fill=muted)
        display = str(value) if index == 0 else f'{value:.1%}'
        draw.text((x + 16, 151), display, font=title, fill=green)
    draw.text((35, 229), 'LEARNED INTERNAL EXPLANATION', font=small, fill=muted)
    labels = ('acquire access', 'open barrier', 'connect stranded object', 'reach safe destination')
    for index, (event, label) in enumerate(zip(report['learned_explanation']['learned_steps'], labels)):
        y = 265 + index * 62; draw.rounded_rectangle((28, y - 7, 1092, y + 40), radius=8, fill=panel)
        draw.text((45, y), f"{index + 1}", font=title, fill=green)
        draw.text((100, y), label, font=body, fill=white)
        draw.text((560, y), f"inferred interaction: {event['interaction']}  /  visual signature {event['visual_signature_rgb']}", font=body, fill=muted)
    draw.text((34, 534), f"Random rescue: {report['random_baseline']['success_rate']:.1%}   |   Old memories unchanged: {report['persistence']['older_memories_unchanged']}", font=body, fill=green)
    draw.text((34, 578), 'The human labels summarize learned visual event signatures; they were not supplied to the procedure learner.', font=small, fill=muted)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); canvas.save(output); return output
