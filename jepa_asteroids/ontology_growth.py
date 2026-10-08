"""Autonomous ontology expansion from unexplained relational task failures."""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .grammar_discovery import SymbolAlphabet, SymbolEchoTerminal, glyph_bank, _kmeans
from .literacy import COLORS
from .runtime import atomic_json
from .strategy_selector import LearnedStrategySelector, probe_features


VOCAB = 12
CLAUSE_ROWS = (4, 20)
SLOTS = ((30, 79), (90, 79), (150, 79), (30, 113), (90, 113), (150, 113))


def _paint(frame: np.ndarray, tokens: list[int], y: int, bank: np.ndarray, rng) -> None:
    for index, token in enumerate(tokens):
        x = 8 + index * 15 + int(rng.integers(-1, 2)); dy = int(rng.integers(-1, 2))
        frame[:, y + dy:y + dy + 7, x:x + 5] = np.maximum(
            frame[:, y + dy:y + dy + 7, x:x + 5], bank[token][None] * 235)


def _relation(operator: int, target: int, reference: int) -> bool:
    tx, ty = SLOTS[target]; rx, ry = SLOTS[reference]
    if operator == 0: return tx < rx
    if operator == 1: return tx > rx
    if operator == 2: return ty < ry
    if operator == 3: return ty > ry
    distance = abs(tx - rx) + abs(ty - ry)
    if operator == 4: return distance <= 60
    if operator == 5: return distance > 60
    raise ValueError(operator)


class RelationalLanguageWorld:
    """Duplicate objects force acquisition of a previously absent spatial ontology."""
    action_dim = 12
    horizon = 2
    def __init__(self, language_seed: int = 10101, *, bank: np.ndarray | None = None,
                 held_out: bool = False):
        self.bank = glyph_bank(language_seed + 31) if bank is None else np.asarray(bank)
        self.held_out = held_out
        rng = np.random.default_rng(language_seed)
        self.semantic_to_glyph = rng.permutation(VOCAB).tolist()
        roles = list(range(6))  # target color/shape, verb, relation, reference color/shape
        legal = [order for order in itertools.permutations(roles)
                 if order.index(0) < order.index(4) and order.index(1) < order.index(5)]
        self.role_order = list(legal[int(rng.integers(len(legal)))])

    @staticmethod
    def _held(command: tuple[int, int, int, int, int, int]) -> bool:
        tc, ts, verb, relation, rc, rs = command
        return (tc * 13 + ts * 7 + verb * 5 + relation * 3 + rc * 2 + rs) % 7 == 0

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)
        for _ in range(500):
            descriptors = [(c, s) for c in range(3) for s in range(3)]
            self.rng.shuffle(descriptors)
            target_a, target_b, reference_a, reference_b = descriptors[:4]
            objects = [target_a, target_a, target_b, target_b, reference_a, reference_b]
            self.rng.shuffle(objects)
            valid = []
            for target in (target_a, target_b):
                candidates = [index for index, value in enumerate(objects) if value == target]
                for reference in (reference_a, reference_b):
                    reference_index = objects.index(reference)
                    for relation in range(4):
                        matches = [index for index in candidates if _relation(relation, index, reference_index)]
                        if len(matches) == 1:
                            for verb in range(2):
                                command = (target[0], target[1], verb, relation, reference[0], reference[1])
                                if self._held(command) == self.held_out:
                                    valid.append((command, matches[0]))
            if len(valid) >= 2:
                self.objects = objects; break
        else: raise RuntimeError('could not generate a relational scene')
        picks = self.rng.choice(len(valid), 2, replace=False)
        self.commands = [valid[int(index)][0] for index in picks]
        self.target_actions = [command[2] * 6 + valid[int(index)][1]
                               for command, index in zip(self.commands, picks)]
        self.clause_glyphs = []
        for command in self.commands:
            tc, ts, verb, relation, rc, rs = command
            semantic = (tc, 3 + ts, 6 + verb, 8 + relation, rc, 3 + rs)
            self.clause_glyphs.append([self.semantic_to_glyph[semantic[role]] for role in self.role_order])
        self.actions = []; self.step_count = 0; self.noise_seed = int(self.rng.integers(2**31))
        return self.render()

    def step(self, action: int):
        self.actions.append(int(action)); self.step_count += 1; done = self.step_count == 2
        reward = 0.0 if not done else (1.0 if self.actions == self.target_actions else -1.0)
        return self.render(), reward, done, {'success': bool(done and reward > 0), 'chain_length': 2}

    def render(self):
        rng = np.random.default_rng(self.noise_seed + self.step_count)
        frame = rng.integers(0, 13, (3, 132, 180), dtype=np.uint8)
        for row, tokens in zip(CLAUSE_ROWS, self.clause_glyphs): _paint(frame, tokens, row, self.bank, rng)
        frame[:, 42:43, 5:175] = 55
        yy, xx = np.ogrid[:132, :180]
        for index, ((cx, cy), (color, shape)) in enumerate(zip(SLOTS, self.objects)):
            if shape == 0: mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 81
            elif shape == 1: mask = (abs(xx - cx) <= 9) & (abs(yy - cy) <= 9)
            else: mask = (yy >= cy - 10) & (yy <= cy + 10) & (abs(xx - cx) <= ((yy - (cy - 10)) // 2))
            for channel in range(3): frame[channel, mask] = COLORS[color, channel]
        return frame


def _decode(alphabet: SymbolAlphabet, frame: np.ndarray) -> list[list[int]]:
    gray = np.mean(frame, axis=0); clauses = []
    for y in CLAUSE_ROWS:
        tokens = []
        for index in range(6):
            x = 8 + index * 15
            patch = (gray[y - 2:y + 10, x - 3:x + 9] >= 128).astype(np.float32)
            if patch.sum() < 4: raise RuntimeError('relational sentence decoding failed')
            tokens.append(alphabet.decode_patch(patch))
        clauses.append(tokens)
    return clauses


def _features(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(frame, dtype=np.float32); colors = []; shapes = []
    for cx, cy in SLOTS:
        patch = rgb[:, cy - 12:cy + 13, cx - 11:cx + 12]; mask = patch.max(0) > 35
        colors.append(patch[:, mask].mean(1) / 255.0); shapes.append(mask.astype(np.float32).reshape(-1))
    return np.asarray(colors), np.asarray(shapes)


class RelationalConcepts:
    def __init__(self, colors: np.ndarray, shapes: np.ndarray):
        self.colors = np.asarray(colors, dtype=np.float32); self.shapes = np.asarray(shapes, dtype=np.float32)
    @classmethod
    def discover(cls, factory, observations=96, seed_base=102_000_000):
        colors = []; shapes = []
        for index in range(observations):
            c, s = _features(factory().reset(seed_base + index)); colors.extend(c); shapes.extend(s)
        return cls(_kmeans(np.asarray(colors)), _kmeans(np.asarray(shapes)))
    def classify(self, frame):
        colors, shapes = _features(frame)
        return (((colors[:, None] - self.colors[None]) ** 2).mean(2).argmin(1),
                ((shapes[:, None] - self.shapes[None]) ** 2).mean(2).argmin(1))
    def to_json(self): return {'colors': self.colors.tolist(), 'shapes': self.shapes.tolist()}
    @classmethod
    def from_json(cls, value): return cls(np.asarray(value['colors']), np.asarray(value['shapes']))


class RelationalStructure:
    def __init__(self, position_groups: list[int], groups: list[set[int]]):
        self.position_groups = position_groups; self.groups = groups
    def to_json(self): return {'position_groups': self.position_groups, 'groups': [sorted(x) for x in self.groups]}
    @classmethod
    def from_json(cls, value): return cls(value['position_groups'], [set(x) for x in value['groups']])


def discover_structure(factory, alphabet: SymbolAlphabet, observations=160,
                       seed_base=103_000_000) -> tuple[RelationalStructure, dict]:
    clauses = []
    for index in range(observations): clauses.extend(_decode(alphabet, factory().reset(seed_base + index)))
    position_sets = [set(clause[position] for clause in clauses) for position in range(6)]
    groups = [] ; position_groups = []
    for token_set in position_sets:
        try: group = next(index for index, existing in enumerate(groups) if existing == token_set)
        except StopIteration: groups.append(token_set); group = len(groups) - 1
        position_groups.append(group)
    if any(not group for group in groups): raise RuntimeError('empty relational word family')
    evidence = {'observations': observations, 'clauses': len(clauses), 'human_labels': 0,
                'families_discovered': len(groups), 'family_sizes': sorted(len(group) for group in groups),
                'position_family_pattern': position_groups}
    return RelationalStructure(position_groups, groups), evidence


def _maps(structure: RelationalStructure, operators: tuple[int, ...]) -> np.ndarray:
    size_three = [index for index, group in enumerate(structure.groups) if len(group) == 3]
    size_two = [index for index, group in enumerate(structure.groups) if len(group) == 2]
    size_four = [index for index, group in enumerate(structure.groups) if len(group) == 4]
    if len(size_three) != 2 or len(size_two) != 1 or len(size_four) != 1:
        raise RuntimeError('discovered structure is incompatible with current grounding ontology')
    results = []
    for color_group, shape_group in (size_three, size_three[::-1]):
      for colors in itertools.permutations(range(3)):
       for shapes in itertools.permutations(range(3, 6)):
        for verbs in itertools.permutations(range(6, 8)):
         for relations in itertools.permutations(operators, 4):
            mapping = np.full(VOCAB, -1, dtype=np.int8)
            for token, value in zip(sorted(structure.groups[color_group]), colors): mapping[token] = value
            for token, value in zip(sorted(structure.groups[shape_group]), shapes): mapping[token] = value
            for token, value in zip(sorted(structure.groups[size_two[0]]), verbs): mapping[token] = value
            for token, value in zip(sorted(structure.groups[size_four[0]]), relations): mapping[token] = 8 + value
            results.append(mapping)
    return np.asarray(results, dtype=np.int8)


class OntologyExpander:
    """Selects new relational primitives from generic pairwise visual features."""
    def __init__(self, alphabet: SymbolAlphabet, concepts: RelationalConcepts,
                 structure: RelationalStructure, operators: tuple[int, ...] = tuple(range(6)),
                 maps: np.ndarray | None = None):
        self.alphabet = alphabet; self.concepts = concepts; self.structure = structure
        self.operator_basis = tuple(operators)
        self.maps = _maps(structure, self.operator_basis) if maps is None else np.asarray(maps, dtype=np.int8)
        self.active = np.ones(len(self.maps), dtype=bool)
    @property
    def active_count(self): return int(self.active.sum())

    def _context(self, frame):
        colors, shapes = self.concepts.classify(frame)
        first = np.full((3, 3), -1, dtype=np.int8); second = np.full((3, 3), -1, dtype=np.int8)
        for index, (color, shape) in enumerate(zip(colors, shapes)):
            color, shape = int(color), int(shape)
            if first[color, shape] < 0: first[color, shape] = index
            else: second[color, shape] = index
        predicates = np.zeros((6, 6, 6), dtype=bool)
        for operator in range(6):
            for target in range(6):
                for reference in range(6): predicates[operator, target, reference] = _relation(operator, target, reference)
        return first, second, predicates

    def _codes(self, frame, clauses, indices: np.ndarray) -> np.ndarray:
        maps = self.maps[indices]; first, second, predicates = self._context(frame)
        actions = [] ; valid_all = np.ones(len(maps), dtype=bool)
        positions = np.arange(6)[None]
        for clause in clauses:
            semantic = maps[:, np.asarray(clause)]
            cmask = semantic < 3; smask = (semantic >= 3) & (semantic < 6)
            vmask = (semantic >= 6) & (semantic < 8); rmask = semantic >= 8
            valid = (cmask.sum(1) == 2) & (smask.sum(1) == 2) & (vmask.sum(1) == 1) & (rmask.sum(1) == 1)
            cfirst = np.where(cmask, positions, 99).min(1); csecond = np.where(cmask & (positions > cfirst[:, None]), positions, 99).min(1)
            sfirst = np.where(smask, positions, 99).min(1); ssecond = np.where(smask & (positions > sfirst[:, None]), positions, 99).min(1)
            vpos = np.where(vmask, positions, 99).min(1); rpos = np.where(rmask, positions, 99).min(1)
            safe = lambda values: np.take_along_axis(semantic, np.minimum(values, 5)[:, None], axis=1)[:, 0]
            tc, rc = safe(cfirst), safe(csecond); ts, rs = safe(sfirst) - 3, safe(ssecond) - 3
            verb, operator = safe(vpos) - 6, safe(rpos) - 8
            t0 = first[tc.clip(0, 2), ts.clip(0, 2)]; t1 = second[tc.clip(0, 2), ts.clip(0, 2)]
            ref = first[rc.clip(0, 2), rs.clip(0, 2)]
            safe_t0, safe_t1, safe_ref = t0.clip(0, 5), t1.clip(0, 5), ref.clip(0, 5)
            holds0 = predicates[operator.clip(0, 5), safe_t0, safe_ref]
            holds1 = predicates[operator.clip(0, 5), safe_t1, safe_ref]
            valid &= (t0 >= 0) & (t1 >= 0) & (ref >= 0) & np.logical_xor(holds0, holds1)
            target = np.where(holds0, t0, t1)
            actions.append((verb * 6 + target).astype(np.int64)); valid_all &= valid
        codes = (actions[0] + 1) + 13 * (actions[1] + 1); codes[~valid_all] = -1
        return codes.astype(np.int64)

    @staticmethod
    def _actions(code: int): return [int(code % 13) - 1, int(code // 13) - 1]

    def plan(self, frame):
        clauses = _decode(self.alphabet, frame); indices = np.flatnonzero(self.active)
        codes = self._codes(frame, clauses, indices); eligible = codes[codes >= 0]
        if not len(eligible): raise RuntimeError('no ontology program can interpret the relational scene')
        code = int(Counter(eligible.tolist()).most_common(1)[0][0])
        return self._actions(code), code, clauses

    def observe(self, frame, clauses, code, reward):
        indices = np.flatnonzero(self.active); codes = self._codes(frame, clauses, indices); matches = codes == code
        # A proposal that cannot parse the current scene is contradicted by the
        # observation itself; a failed action only rules out proposals that
        # predicted that exact chain.
        keep = matches if reward > 0 else ((codes >= 0) & ~matches); self.active[indices] &= keep
        if not self.active.any(): raise RuntimeError('evidence eliminated every ontology proposal')

    def best_map(self): return self.maps[int(np.flatnonzero(self.active)[0])]
    def promoted_operators(self):
        relation_tokens = next(group for group in self.structure.groups if len(group) == 4)
        return sorted({int(self.best_map()[token] - 8) for token in relation_tokens})
    def save(self, path: Path):
        atomic_json(path, {'format': 'wailah-grown-ontology-memory-v1',
            'active_maps': self.maps[self.active].astype(int).tolist(), 'operator_basis': list(self.operator_basis),
            'promoted_operators': self.promoted_operators(), 'alphabet': self.alphabet.to_json(),
            'concepts': self.concepts.to_json(), 'structure': self.structure.to_json()})
    @classmethod
    def load(cls, path: Path):
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(SymbolAlphabet.from_json(value['alphabet']), RelationalConcepts.from_json(value['concepts']),
                   RelationalStructure.from_json(value['structure']), tuple(value['operator_basis']),
                   np.asarray(value['active_maps'], dtype=np.int8))


class OntologyStrategy:
    def __init__(self, learner): self.learner = learner; self.queue = []
    def reset(self, observation): self.queue = self.learner.plan(observation)[0]
    def act(self, observation): return self.queue.pop(0) if self.queue else 0
    @classmethod
    def load(cls, workspace): return cls(OntologyExpander.load(Path(workspace) / 'relation_language_00.json'))


def _train(learner, factory, episodes, seed_base):
    rewards = []; resolved = None; trace = []; start = learner.active_count
    for episode in range(episodes):
        env = factory(); frame = env.reset(seed_base + episode); actions, code, clauses = learner.plan(frame)
        reward = 0.0
        for action in actions: _, reward, done, _ = env.step(action)
        learner.observe(frame, clauses, code, reward); rewards.append(reward); trace.append(learner.active_count)
        if learner.active_count == 1 and resolved is None: resolved = episode + 1
    return {'episodes': episodes, 'initial_programs': start, 'remaining_programs': learner.active_count,
            'episodes_to_single_program': resolved,
            'failures_before_resolution': int(np.sum(np.asarray(rewards[:resolved]) <= 0)) if resolved else None,
            'success_first_32': float(np.mean(np.asarray(rewards[:32]) > 0)),
            'success_last_64': float(np.mean(np.asarray(rewards[-64:]) > 0)),
            'trace': [trace[i] for i in range(0, len(trace), max(1, episodes // 16))] + [trace[-1]]}


def _evaluate(learner, factory, episodes, seed_base):
    success = 0
    for index in range(episodes):
        env = factory(); frame = env.reset(seed_base + index); actions, _, _ = learner.plan(frame); reward = 0.0
        for action in actions: _, reward, done, _ = env.step(action)
        success += int(reward > 0)
    return {'episodes': episodes, 'success_rate': success / episodes, 'random_chain_success': 1 / 144}


def _oracle_without_relation(factory, episodes, seed_base):
    """Upper bound for the old ontology: all old meanings known, relation unavailable."""
    success = 0
    for index in range(episodes):
        env = factory(); env.reset(seed_base + index); actions = []
        for command in env.commands:
            tc, ts, verb, _, _, _ = command
            candidates = [slot for slot, value in enumerate(env.objects) if value == (tc, ts)]
            actions.append(verb * 6 + candidates[0])
        reward = 0.0
        for action in actions: _, reward, done, _ = env.step(action)
        success += int(reward > 0)
    return {'episodes': episodes, 'success_rate': success / episodes,
            'privilege': 'all old color/shape/verb meanings supplied; new relation unavailable'}


def run_ontology_growth_benchmark(workspace: Path, output: Path, device='cpu') -> dict:
    workspace, output = Path(workspace), Path(output); brain = workspace / 'living' / 'ontology_growth'; brain.mkdir(parents=True, exist_ok=True)
    seed = 10101; bank = glyph_bank(seed + 31); key_seed = 10102
    train_factory = lambda: RelationalLanguageWorld(seed, bank=bank, held_out=False)
    test_factory = lambda: RelationalLanguageWorld(seed, bank=bank, held_out=True)
    alphabet = SymbolAlphabet.discover(lambda: SymbolEchoTerminal(bank, key_seed), samples=5, seed_base=104_000_000)
    structure, evidence = discover_structure(train_factory, alphabet)
    concepts = RelationalConcepts.discover(train_factory)
    old_baseline = _oracle_without_relation(test_factory, 512, 105_000_000)
    learner = OntologyExpander(alphabet, concepts, structure)
    training = _train(learner, train_factory, 220, 106_000_000)
    evaluation = _evaluate(learner, test_factory, 512, 107_000_000)
    learner.save(brain / 'relation_language_00.json'); original_hash = hashlib.sha256((brain / 'relation_language_00.json').read_bytes()).hexdigest()
    promoted_ops = tuple(learner.promoted_operators())
    registry = {'format': 'wailah-expandable-ontology-v1', 'source': 'generic pairwise pixel geometry',
        'candidate_primitives': 6, 'promoted_primitives': [{'internal_id': int(value), 'kind': 'binary-spatial-predicate'} for value in promoted_ops]}
    atomic_json(brain / 'ontology_registry.json', registry)

    audits = []
    for audit_index in range(6):
        language_seed = 11101 + audit_index * 101; audit_bank = glyph_bank(language_seed + 31)
        factory = lambda ls=language_seed, b=audit_bank: RelationalLanguageWorld(ls, bank=b, held_out=False)
        held = lambda ls=language_seed, b=audit_bank: RelationalLanguageWorld(ls, bank=b, held_out=True)
        alpha = SymbolAlphabet.discover(lambda b=audit_bank, k=11200 + audit_index: SymbolEchoTerminal(b, k),
            samples=3, seed_base=108_000_000 + audit_index * 10_000)
        found, found_evidence = discover_structure(factory, alpha, observations=160,
            seed_base=108_100_000 + audit_index * 10_000)
        transferred = OntologyExpander(alpha, concepts, found, promoted_ops)
        learned = _train(transferred, factory, 140, 108_200_000 + audit_index * 10_000)
        tested = _evaluate(transferred, held, 128, 108_300_000 + audit_index * 10_000)
        path = brain / f'relation_language_{audit_index + 1:02d}.json'; transferred.save(path)
        audits.append({'language_index': audit_index + 1, 'structure': found_evidence,
            'training': learned, 'evaluation': tested, 'memory_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

    compare_seed = 12121; compare_bank = glyph_bank(compare_seed + 31)
    compare_factory = lambda: RelationalLanguageWorld(compare_seed, bank=compare_bank, held_out=False)
    compare_alpha = SymbolAlphabet.discover(lambda: SymbolEchoTerminal(compare_bank, 12122), samples=3, seed_base=109_000_000)
    compare_structure, _ = discover_structure(compare_factory, compare_alpha, observations=160, seed_base=109_100_000)
    scratch = OntologyExpander(compare_alpha, concepts, compare_structure)
    transfer = OntologyExpander(compare_alpha, concepts, compare_structure, promoted_ops)
    scratch_result = _train(scratch, compare_factory, 180, 109_200_000)
    transfer_result = _train(transfer, compare_factory, 180, 109_200_000)

    revisit = _evaluate(OntologyExpander.load(brain / 'relation_language_00.json'), test_factory, 512, 110_000_000)
    retained_hash = hashlib.sha256((brain / 'relation_language_00.json').read_bytes()).hexdigest()

    selector_path = workspace / 'living' / 'strategy_selector.json'; selector = LearnedStrategySelector.load(selector_path)
    features = np.stack([probe_features(test_factory, 111_000_000 + index) for index in range(64)])
    selector.centroids['ontology-growth-memory'] = features.mean(0)
    selector.scale = np.maximum(selector.scale, features.std(0)); selector.save(selector_path)
    selector_accuracy = float(np.mean([selector.rank(test_factory, 111_100_000 + index,
        list(selector.centroids))[0][0] == 'ontology-growth-memory' for index in range(32)]))
    from .autonomy import AutonomousCompetenceLoop
    loop = AutonomousCompetenceLoop(workspace, device, ontology_workspace=brain)
    decision = loop.select(test_factory, probe_seed=111_300_000)
    competence = loop.evaluate_selected(test_factory, decision['method'], seed_base=111_400_000, episodes=128)

    report = {'format': 'wailah-ontology-growth-v4',
        'learner_inputs': 'pixels, self-chosen motor actions, terminal reward, done',
        'failure_trigger': {'known_descriptor_aliases_per_target': 2, 'old_ontology': old_baseline,
            'interpretation': 'color and shape cannot distinguish the rewarded object'},
        'structure_discovery': evidence,
        'ontology_expansion': {'candidate_pairwise_primitives': 6, 'promoted_primitives': list(promoted_ops),
            'new_primitive_count': len(promoted_ops), 'registry': registry, 'training': training,
            'withheld_composition': evaluation},
        'cross_language_transfer': {'languages': len(audits), 'runs': audits,
            'all_resolved': all(row['training']['remaining_programs'] == 1 for row in audits),
            'aggregate_accuracy': float(np.mean([row['evaluation']['success_rate'] for row in audits])),
            'mean_resolution_episodes': float(np.mean([row['training']['episodes_to_single_program'] for row in audits])),
            'controlled_comparison': {'scratch': scratch_result, 'transferred_ontology': transfer_result,
                'initial_search_reduction': scratch_result['initial_programs'] / transfer_result['initial_programs']}},
        'retention': {'before': evaluation, 'after_six_languages': revisit,
            'memory_hash_unchanged': original_hash == retained_hash},
        'autonomous_integration': {'selector_accuracy': selector_accuracy, 'decision': decision,
            'competence_evaluation': competence},
        'promoted': decision['method'] == 'ontology-growth-memory' and competence.get('success_rate', 0) >= .95}
    atomic_json(output, report); return report


def render_ontology_growth_summary(report_path: Path, output: Path) -> Path:
    from PIL import Image, ImageDraw, ImageFont
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    try:
        title = ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf', 27); body = ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf', 15); small = ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf', 12)
    except OSError: title = body = small = ImageFont.load_default()
    canvas = Image.new('RGB', (1120, 690), (7, 13, 25)); draw = ImageDraw.Draw(canvas)
    green, white, muted, panel = (75, 225, 164), (232, 239, 248), (139, 156, 180), (14, 25, 43)
    draw.text((34, 24), 'WAILAH  /  ONTOLOGY GROWTH v4', font=title, fill=white)
    draw.text((35, 64), 'detect missing concepts  •  propose predicates  •  test through action  •  promote  •  transfer', font=body, fill=muted)
    growth = report['ontology_expansion']; transfer = report['cross_language_transfer']; retention = report['retention']
    cards = [('OLD ONTOLOGY', report['failure_trigger']['old_ontology']['success_rate']), ('GROWN ONTOLOGY', growth['withheld_composition']['success_rate']),
             ('6-LANGUAGE TRANSFER', transfer['aggregate_accuracy']), ('REVISIT', retention['after_six_languages']['success_rate'])]
    for index, (label, value) in enumerate(cards):
        x = 34 + index * 270; draw.rounded_rectangle((x, 105, x + 246, 197), radius=12, fill=panel)
        draw.text((x + 16, 121), label, font=small, fill=muted); draw.text((x + 16, 151), f'{value:.1%}', font=title, fill=green)
    comparison = transfer['controlled_comparison']; rows = [
        ('Novelty trigger', 'two visually identical candidates / old concepts insufficient'),
        ('Candidate ontology', f"{growth['candidate_pairwise_primitives']} generic geometric predicates"),
        ('Promoted ontology', f"{growth['new_primitive_count']} persistent spatial primitives"),
        ('Initial grounding', f"{growth['training']['initial_programs']:,} → 1 programs"),
        ('Cross-language reuse', f"{transfer['languages']}/6 languages resolved / {transfer['aggregate_accuracy']:.1%} held-out"),
        ('Search reduction', f"{comparison['initial_search_reduction']:.1f}× fewer programs with learned ontology"),
        ('Autonomous competence gate', report['autonomous_integration']['decision']['method'])]
    draw.text((35, 228), 'CAPABILITY', font=small, fill=muted); draw.text((560, 228), 'MEASURED RESULT', font=small, fill=muted)
    for index, (name, value) in enumerate(rows):
        y = 260 + index * 52; draw.rounded_rectangle((28, y - 7, 1092, y + 34), radius=8, fill=panel)
        draw.text((43, y), name, font=body, fill=white); draw.text((560, y), value, font=body, fill=green)
    draw.text((34, 642), 'The relation labels are never supplied. Candidate predicates are generated from generic pairwise pixel geometry.', font=small, fill=muted)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); canvas.save(output); return output
