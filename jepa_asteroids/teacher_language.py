"""A small visual teacher bridge: typed words grounded by demonstrations."""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .literacy import COLORS
from .grammar_discovery import _kmeans
from .ontology_growth import RelationalConcepts
from .runtime import atomic_json
from .strategy_selector import LearnedStrategySelector, probe_features


WIDTH, HEIGHT = 360, 150
ROWS = (3, 22)
CELL_W, CELL_H = 58, 16
SLOTS = ((40, 94), (180, 94), (320, 94), (40, 130), (180, 130), (320, 130))
COLORS_WORDS = ('red', 'green', 'blue')
SHAPE_WORDS = ('circle', 'square', 'triangle')
VERB_WORDS = ('touch', 'mark')
RELATION_WORDS = ('left', 'right', 'above', 'below')
BASE_WORDS = COLORS_WORDS + SHAPE_WORDS + VERB_WORDS + RELATION_WORDS


def _font():
    try: return ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf', 12)
    except OSError: return ImageFont.load_default()


def _spatial(operator: int, target: int, reference: int) -> bool:
    tx, ty = SLOTS[target]; rx, ry = SLOTS[reference]
    if operator == 0: return tx < rx
    if operator == 1: return tx > rx
    if operator == 2: return ty < ry
    if operator == 3: return ty > ry
    distance = abs(tx - rx) + abs(ty - ry)
    if operator == 4: return distance <= 140
    if operator == 5: return distance > 140
    raise ValueError(operator)


class TaughtInstructionWorld:
    """English-looking commands whose meanings are exposed only by demonstrations."""
    action_dim = 12
    horizon = 2

    def __init__(self, *, held_out=False, relation: int | None=None,
                 color_alias: str | None=None):
        self.held_out = held_out; self.forced_relation = relation
        self.color_alias = color_alias

    @staticmethod
    def _held(command):
        tc, ts, verb, relation, rc, rs = command
        return (tc * 13 + ts * 7 + verb * 5 + relation * 3 + rc * 2 + rs) % 7 == 0

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)
        for _ in range(1000):
            descriptors = [(c, s) for c in range(3) for s in range(3)]
            self.rng.shuffle(descriptors)
            target_a, target_b, reference_a, reference_b = descriptors[:4]
            objects = [target_a, target_a, target_b, target_b, reference_a, reference_b]
            self.rng.shuffle(objects); valid = []
            relations = (self.forced_relation,) if self.forced_relation is not None else range(4)
            for target in (target_a, target_b):
                candidates = [i for i, value in enumerate(objects) if value == target]
                for reference in (reference_a, reference_b):
                    ref = objects.index(reference)
                    for relation in relations:
                        matches = [i for i in candidates if _spatial(relation, i, ref)]
                        if len(matches) != 1: continue
                        for verb in range(2):
                            command = (target[0], target[1], verb, relation,
                                       reference[0], reference[1])
                            if self.forced_relation is None and self._held(command) != self.held_out:
                                continue
                            if self.color_alias is not None and target[0] != 0 and reference[0] != 0:
                                continue
                            valid.append((command, matches[0]))
            if len(valid) >= 2:
                self.objects = objects; break
        else: raise RuntimeError('could not construct taught instruction scene')
        picks = self.rng.choice(len(valid), 2, replace=False)
        chosen = [valid[int(i)] for i in picks]
        self.commands = [row[0] for row in chosen]
        self.target_actions = [command[2] * 6 + slot for command, slot in chosen]
        self.clauses = [self._words(command) for command in self.commands]
        self.actions = []; self.step_count = 0
        self.noise_seed = int(self.rng.integers(2**31)); return self.render()

    def _words(self, command):
        tc, ts, verb, relation, rc, rs = command
        color = list(COLORS_WORDS)
        if self.color_alias is not None: color[0] = self.color_alias
        return [VERB_WORDS[verb], color[tc], SHAPE_WORDS[ts],
                ('near' if relation == 4 else ('far' if relation == 5 else RELATION_WORDS[relation])),
                color[rc], SHAPE_WORDS[rs]]

    def step(self, action: int):
        self.actions.append(int(action)); self.step_count += 1; done = self.step_count == 2
        reward = 0.0 if not done else (1.0 if self.actions == self.target_actions else -1.0)
        return self.render(), reward, done, {'success': bool(done and reward > 0), 'chain_length': 2}

    def render(self):
        rng = np.random.default_rng(self.noise_seed + self.step_count)
        frame = rng.integers(0, 13, (3, HEIGHT, WIDTH), dtype=np.uint8)
        image = Image.fromarray(np.moveaxis(frame, 0, 2), mode='RGB'); draw = ImageDraw.Draw(image)
        font = _font()
        for row, clause in zip(ROWS, self.clauses):
            for index, word in enumerate(clause):
                dx, dy = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
                draw.text((4 + index * CELL_W + dx, row + dy), word, font=font, fill=(240, 240, 240))
        frame = np.moveaxis(np.asarray(image), 2, 0).copy(); frame[:, 43:44, 5:355] = 55
        yy, xx = np.ogrid[:HEIGHT, :WIDTH]
        for (cx, cy), (color, shape) in zip(SLOTS, self.objects):
            if shape == 0: mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 81
            elif shape == 1: mask = (abs(xx - cx) <= 9) & (abs(yy - cy) <= 9)
            else: mask = (yy >= cy - 10) & (yy <= cy + 10) & (abs(xx - cx) <= ((yy - (cy - 10)) // 2))
            for channel in range(3): frame[channel, mask] = COLORS[color, channel]
        return frame


def _word_patch(frame: np.ndarray, row: int, position: int) -> np.ndarray:
    gray = np.mean(frame, axis=0)
    x = 2 + position * CELL_W
    return (gray[max(0, row - 2):row - 2 + CELL_H, x:x + CELL_W - 2] >= 128).astype(np.float32)


class VisualWordMemory:
    """An expandable visual lexicon learned from teacher-presented spellings."""
    def __init__(self, prototypes=None):
        self.prototypes = prototypes or {}; self._rebuild()

    def _rebuild(self):
        self.templates = {}
        for word, values in self.prototypes.items():
            self.templates[word] = np.asarray([
                np.roll(value, (dy, dx), (0, 1)) for value in values
                for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)], dtype=np.float32)

    def observe(self, frame, clauses):
        for row, clause in zip(ROWS, clauses):
            for position, word in enumerate(clause):
                values = self.prototypes.setdefault(word, [])
                if len(values) < 4: values.append(_word_patch(frame, row, position))
        self._rebuild()

    def decode(self, frame):
        output = []
        for row in ROWS:
            clause = []
            for position in range(6):
                patch = _word_patch(frame, row, position)
                word = min(self.templates, key=lambda key: float(np.mean(
                    np.abs(self.templates[key] - patch[None]), axis=(1, 2)).min()))
                clause.append(word)
            output.append(clause)
        return output

    def to_json(self):
        return {word: [value.astype(int).tolist() for value in values]
                for word, values in self.prototypes.items()}

    @classmethod
    def from_json(cls, value):
        return cls({word: [np.asarray(x, dtype=np.float32) for x in values]
                    for word, values in value.items()})


def _features(frame):
    rgb = np.asarray(frame, dtype=np.float32); colors = []; shapes = []
    for cx, cy in SLOTS:
        patch = rgb[:, cy - 12:cy + 13, cx - 11:cx + 12]; mask = patch.max(0) > 35
        colors.append(patch[:, mask].mean(1) / 255.0); shapes.append(mask.astype(np.float32).reshape(-1))
    return np.asarray(colors), np.asarray(shapes)


class TaughtConcepts(RelationalConcepts):
    @classmethod
    def discover(cls, factory, observations=64, seed_base=129_000_000):
        colors = []; shapes = []
        for index in range(observations):
            c, s = _features(factory().reset(seed_base + index)); colors.extend(c); shapes.extend(s)
        return cls(_kmeans(np.asarray(colors)), _kmeans(np.asarray(shapes)))

    def classify(self, frame):
        colors, shapes = _features(frame)
        return (((colors[:, None] - self.colors[None]) ** 2).mean(2).argmin(1),
                ((shapes[:, None] - self.shapes[None]) ** 2).mean(2).argmin(1))


def _base_maps(operators=(0, 1, 2, 3)):
    maps = []
    for colors in itertools.permutations(range(3)):
      for shapes in itertools.permutations(range(3, 6)):
       for verbs in itertools.permutations(range(6, 8)):
        for relations in itertools.permutations(operators, 4):
            maps.append(np.asarray(colors + shapes + verbs + tuple(8 + x for x in relations), dtype=np.int8))
    return np.asarray(maps)


class TaughtLanguageAgent:
    """Grounds visual words from demonstrations and keeps an open alias vocabulary."""
    def __init__(self, concepts, reader=None, maps=None, aliases=None):
        self.concepts = concepts; self.reader = reader or VisualWordMemory()
        self.maps = _base_maps() if maps is None else np.asarray(maps, dtype=np.int8)
        self.active = np.ones(len(self.maps), dtype=bool)
        self.aliases = dict(aliases or {})

    def _meaning(self, word, maps):
        if word in self.aliases: return np.full(len(maps), self.aliases[word], dtype=np.int8)
        return maps[:, BASE_WORDS.index(word)]

    def _codes(self, frame, clauses, indices, temporary=None):
        maps = self.maps[indices]; colors, shapes = self.concepts.classify(frame)
        first = np.full((3, 3), -1, dtype=np.int8); second = np.full((3, 3), -1, dtype=np.int8)
        for slot, (color, shape) in enumerate(zip(colors, shapes)):
            color, shape = int(color), int(shape)
            if first[color, shape] < 0: first[color, shape] = slot
            else: second[color, shape] = slot
        actions = []; valid_all = np.ones(len(maps), dtype=bool)
        for clause in clauses:
            semantic = []
            for word in clause:
                if temporary and word in temporary: semantic.append(np.full(len(maps), temporary[word], dtype=np.int8))
                else: semantic.append(self._meaning(word, maps))
            verb, tc, ts, operator, rc, rs = semantic
            verb = verb - 6; ts = ts - 3; operator = operator - 8; rs = rs - 3
            t0 = first[tc.clip(0, 2), ts.clip(0, 2)]; t1 = second[tc.clip(0, 2), ts.clip(0, 2)]
            ref = first[rc.clip(0, 2), rs.clip(0, 2)]
            holds0 = np.asarray([_spatial(int(op), int(t), int(r))
                                 for op, t, r in zip(operator.clip(0, 5), t0.clip(0, 5), ref.clip(0, 5))])
            holds1 = np.asarray([_spatial(int(op), int(t), int(r))
                                 for op, t, r in zip(operator.clip(0, 5), t1.clip(0, 5), ref.clip(0, 5))])
            valid = ((verb >= 0) & (verb < 2) & (tc >= 0) & (tc < 3) &
                     (ts >= 0) & (ts < 3) & (operator >= 0) & (operator < 6) &
                     (rc >= 0) & (rc < 3) & (rs >= 0) & (rs < 3) &
                     (t0 >= 0) & (t1 >= 0) & (ref >= 0) & np.logical_xor(holds0, holds1))
            actions.append((verb * 6 + np.where(holds0, t0, t1)).astype(np.int64)); valid_all &= valid
        codes = (actions[0] + 1) + 13 * (actions[1] + 1); codes[~valid_all] = -1
        return codes

    @staticmethod
    def _code(actions): return (int(actions[0]) + 1) + 13 * (int(actions[1]) + 1)
    @staticmethod
    def _actions(code): return [int(code % 13) - 1, int(code // 13) - 1]

    def teach(self, frame, written_clauses, demonstrated_actions):
        """Learn from what the teacher wrote, showed, and successfully did."""
        self.reader.observe(frame, written_clauses)
        indices = np.flatnonzero(self.active)
        codes = self._codes(frame, written_clauses, indices)
        self.active[indices] &= codes == self._code(demonstrated_actions)
        if not self.active.any(): raise RuntimeError('teacher demonstration contradicted every language hypothesis')

    def teach_new_word(self, frame, written_clauses, demonstrated_actions, word, role):
        """Add a word after the base language has resolved, without reopening old meanings."""
        if self.active.sum() != 1: raise RuntimeError('resolve the base teaching language first')
        self.reader.observe(frame, written_clauses)
        ranges = {'color': range(3), 'shape': range(3, 6), 'verb': range(6, 8), 'relation': range(8, 14)}
        expected = self._code(demonstrated_actions); candidates = []
        for meaning in ranges[role]:
            code = self._codes(frame, written_clauses, np.flatnonzero(self.active), {word: meaning})[0]
            if code == expected: candidates.append(int(meaning))
        previous = set(self.aliases[word]) if isinstance(self.aliases.get(word), list) else None
        candidates = sorted(set(candidates) if previous is None else set(candidates) & previous)
        self.aliases[word] = candidates
        if len(candidates) == 1: self.aliases[word] = candidates[0]
        if not candidates: raise RuntimeError(f'teaching eliminated every meaning for {word}')

    def plan(self, frame):
        clauses = self.reader.decode(frame); indices = np.flatnonzero(self.active)
        if any(isinstance(value, list) for value in self.aliases.values()):
            raise RuntimeError('a newly taught word is still ambiguous')
        codes = self._codes(frame, clauses, indices); eligible = codes[codes >= 0]
        if not len(eligible): raise RuntimeError('no taught interpretation fits this scene')
        return self._actions(int(Counter(eligible.tolist()).most_common(1)[0][0]))

    def save(self, path):
        atomic_json(path, {'format': 'wailah-taught-language-v1', 'base_words': list(BASE_WORDS),
            'active_maps': self.maps[self.active].astype(int).tolist(), 'aliases': self.aliases,
            'reader': self.reader.to_json(), 'concepts': self.concepts.to_json()})

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(TaughtConcepts.from_json(value['concepts']), VisualWordMemory.from_json(value['reader']),
                   np.asarray(value['active_maps'], dtype=np.int8), value['aliases'])


class TeacherStrategy:
    def __init__(self, agent): self.agent = agent; self.queue = []
    def reset(self, observation): self.queue = self.agent.plan(observation)
    def act(self, observation): return self.queue.pop(0) if self.queue else 0
    @classmethod
    def load(cls, workspace):
        return cls(TaughtLanguageAgent.load(Path(workspace) / 'english_teacher_memory.json'))


def _demonstrate(agent, factory, episodes, seed_base):
    start = int(agent.active.sum()); resolved = None; trace = []
    for episode in range(episodes):
        env = factory(); frame = env.reset(seed_base + episode)
        agent.teach(frame, env.clauses, env.target_actions); trace.append(int(agent.active.sum()))
        if agent.active.sum() == 1 and resolved is None: resolved = episode + 1
    return {'demonstrations': episodes, 'initial_hypotheses': start,
            'remaining_hypotheses': int(agent.active.sum()), 'resolved_after': resolved, 'trace': trace}


def _teach_extension(agent, factory, word, role, episodes, seed_base):
    trace = []
    for episode in range(episodes):
        env = factory(); frame = env.reset(seed_base + episode)
        agent.teach_new_word(frame, env.clauses, env.target_actions, word, role)
        value = agent.aliases[word]; trace.append(len(value) if isinstance(value, list) else 1)
        if not isinstance(value, list): break
    return {'word': word, 'role': role, 'demonstrations': len(trace),
            'remaining_meanings': trace[-1], 'meaning_trace': trace}


def _evaluate(agent, factory, episodes, seed_base):
    success = 0
    for episode in range(episodes):
        env = factory(); frame = env.reset(seed_base + episode); actions = agent.plan(frame)
        reward = 0.0
        for action in actions: _, reward, _, _ = env.step(action)
        success += int(reward > 0)
    return {'episodes': episodes, 'success_rate': success / episodes, 'random_chain_success': 1 / 144}


def run_teacher_language_benchmark(workspace: Path, output: Path) -> dict:
    workspace, output = Path(workspace), Path(output)
    brain = workspace / 'living' / 'teacher_language'; brain.mkdir(parents=True, exist_ok=True)
    source = json.loads((workspace / 'living' / 'ontology_growth' / 'relation_language_00.json').read_text())
    concepts = TaughtConcepts.from_json(source['concepts'])
    agent = TaughtLanguageAgent(concepts)
    base = _demonstrate(agent, lambda: TaughtInstructionWorld(), 24, 130_000_000)
    base_eval = _evaluate(agent, lambda: TaughtInstructionWorld(held_out=True), 256, 130_100_000)
    base_hash_map = agent.maps[np.flatnonzero(agent.active)[0]].copy()

    near = _teach_extension(agent, lambda: TaughtInstructionWorld(relation=4),
                            'near', 'relation', 12, 130_200_000)
    near_eval = _evaluate(agent, lambda: TaughtInstructionWorld(relation=4), 256, 130_300_000)
    crimson = _teach_extension(agent, lambda: TaughtInstructionWorld(color_alias='crimson'),
                               'crimson', 'color', 12, 130_400_000)
    crimson_eval = _evaluate(agent, lambda: TaughtInstructionWorld(held_out=True, color_alias='crimson'),
                             256, 130_500_000)
    combined = _evaluate(agent, lambda: TaughtInstructionWorld(relation=4, color_alias='crimson'),
                         256, 130_600_000)
    retention = _evaluate(agent, lambda: TaughtInstructionWorld(held_out=True), 256, 130_700_000)
    path = brain / 'english_teacher_memory.json'; agent.save(path)
    loaded = TaughtLanguageAgent.load(path)
    reload_eval = _evaluate(loaded, lambda: TaughtInstructionWorld(relation=4, color_alias='crimson'),
                            128, 130_800_000)
    registry = {'format': 'wailah-open-teacher-vocabulary-v1',
        'known_surface_words': sorted(agent.reader.prototypes),
        'base_words': list(BASE_WORDS), 'later_words': ['near', 'crimson'],
        'new_concepts': [{'word': 'near', 'kind': 'binary-spatial-predicate',
                          'internal_id': int(agent.aliases['near'] - 8)}],
        'aliases': {'crimson': 'same learned visual color concept as red'},
        'memory_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    atomic_json(brain / 'teacher_registry.json', registry)
    selector_path = workspace / 'living' / 'strategy_selector.json'
    selector = LearnedStrategySelector.load(selector_path)
    held_factory = lambda: TaughtInstructionWorld(held_out=True)
    features = np.stack([probe_features(held_factory, 131_000_000 + index) for index in range(64)])
    selector.centroids['teacher-language-memory'] = features.mean(0)
    selector.scale = np.maximum(selector.scale, features.std(0)); selector.save(selector_path)
    selector_accuracy = float(np.mean([selector.rank(held_factory, 131_100_000 + index,
        list(selector.centroids))[0][0] == 'teacher-language-memory' for index in range(32)]))
    from .autonomy import AutonomousCompetenceLoop
    loop = AutonomousCompetenceLoop(workspace, 'cpu', teacher_workspace=brain)
    decision = loop.select(held_factory, probe_seed=131_200_000)
    competence = loop.evaluate_selected(held_factory, decision['method'], seed_base=131_300_000, episodes=128)
    report = {'format': 'wailah-teacher-language-v5',
        'interaction': 'teacher-written visual words plus successful action demonstrations; evaluation receives pixels only',
        'base_teaching': base, 'base_vocabulary': list(BASE_WORDS), 'base_withheld': base_eval,
        'later_teaching': {'near': near, 'crimson': crimson},
        'new_word_evaluations': {'near': near_eval, 'crimson': crimson_eval,
                                 'near_plus_crimson': combined},
        'retention': {'base_after_new_words': retention,
                      'base_dictionary_unchanged': bool(np.array_equal(base_hash_map, agent.maps[np.flatnonzero(agent.active)[0]]))},
        'persistence': {'reload_combined': reload_eval, 'registry': registry},
        'autonomous_integration': {'selector_accuracy': selector_accuracy,
                                   'decision': decision, 'competence_evaluation': competence},
        'passed': bool(base_eval['success_rate'] == near_eval['success_rate'] ==
                       crimson_eval['success_rate'] == combined['success_rate'] ==
                       retention['success_rate'] == reload_eval['success_rate'] == 1.0 and
                       decision['method'] == 'teacher-language-memory' and
                       competence.get('success_rate', 0) == 1.0)}
    atomic_json(output, report); return report


def render_teacher_summary(report_path: Path, output: Path):
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    try:
        title = ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf', 27)
        body = ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf', 15)
        small = ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf', 12)
    except OSError: title = body = small = ImageFont.load_default()
    canvas = Image.new('RGB', (1120, 650), (7, 13, 25)); draw = ImageDraw.Draw(canvas)
    green, white, muted, panel = (75, 225, 164), (232, 239, 248), (139, 156, 180), (14, 25, 43)
    draw.text((34, 24), 'WAILAH  /  LEARNING FROM A TEACHER', font=title, fill=white)
    draw.text((35, 64), 'typed words → visual memory → demonstrated meaning → new combinations', font=body, fill=muted)
    evaluations = [('BASE ENGLISH', report['base_withheld']['success_rate']),
                   ('NEW: NEAR', report['new_word_evaluations']['near']['success_rate']),
                   ('NEW: CRIMSON', report['new_word_evaluations']['crimson']['success_rate']),
                   ('COMBINED', report['new_word_evaluations']['near_plus_crimson']['success_rate'])]
    for index, (label, value) in enumerate(evaluations):
        x = 34 + index * 270; draw.rounded_rectangle((x, 105, x + 246, 197), radius=12, fill=panel)
        draw.text((x + 16, 121), label, font=small, fill=muted); draw.text((x + 16, 151), f'{value:.1%}', font=title, fill=green)
    rows = [('Initial taught vocabulary', f"{len(report['base_vocabulary'])} ordinary typed words"),
            ('Meaning hypotheses', f"{report['base_teaching']['initial_hypotheses']:,} → 1"),
            ('Base demonstrations', str(report['base_teaching']['resolved_after'])),
            ('Later lessons', 'near = new relation  /  crimson = color alias'),
            ('Old-language retention', f"{report['retention']['base_after_new_words']['success_rate']:.1%}"),
            ('Saved-memory reload', f"{report['persistence']['reload_combined']['success_rate']:.1%}"),
            ('Agent input at evaluation', 'pixels only')]
    draw.text((35, 228), 'CAPABILITY', font=small, fill=muted); draw.text((560, 228), 'MEASURED RESULT', font=small, fill=muted)
    for index, (name, value) in enumerate(rows):
        y = 260 + index * 49; draw.rounded_rectangle((28, y - 7, 1092, y + 31), radius=8, fill=panel)
        draw.text((43, y), name, font=body, fill=white); draw.text((560, y), value, font=body, fill=green)
    draw.text((34, 618), 'The teacher supplies spellings and demonstrations—not hidden dictionary entries or test answers.', font=small, fill=muted)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); canvas.save(output); return output
