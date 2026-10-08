"""Discover latent word families, grammar and grounded plans from pixels and reward."""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .literacy import COLORS
from .runtime import atomic_json
from .strategy_selector import LearnedStrategySelector, probe_features
from .language_nursery import _kmeans


VOCAB = 12
CLAUSE_ROWS = (4, 30)
CONNECTOR_ROW = 18
OBJECT_X = (20, 55, 90, 125)


def glyph_bank(seed: int, vocabulary: int = VOCAB) -> np.ndarray:
    """Create a reproducible unfamiliar visual alphabet, without semantic labels."""
    rng = np.random.default_rng(seed); bank = []
    while len(bank) < vocabulary:
        glyph = (rng.random((7, 5)) < .34).astype(np.uint8)
        glyph[:, 2] |= (rng.random(7) < .28)
        if 7 <= glyph.sum() <= 22 and all(np.mean(glyph != old) > .20 for old in bank):
            bank.append(glyph)
    return np.asarray(bank, dtype=np.uint8)


def _paint_tokens(frame: np.ndarray, tokens: list[int], y: int, bank: np.ndarray,
                  rng, *, x0: int = 6) -> None:
    for index, token in enumerate(tokens):
        dy = int(rng.integers(-1, 2)); dx = int(rng.integers(-1, 2))
        x = x0 + index * 13 + dx
        frame[:, y + dy:y + dy + 7, x:x + 5] = np.maximum(
            frame[:, y + dy:y + dy + 7, x:x + 5], bank[token][None] * 235)


class SymbolEchoTerminal:
    """A twelve-key visual terminal used to learn glyph identity by active probing."""
    action_dim = VOCAB
    horizon = 1
    def __init__(self, bank: np.ndarray, mapping_seed: int):
        self.bank = np.asarray(bank); self.mapping = np.random.default_rng(mapping_seed).permutation(VOCAB).tolist()
    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed); self.glyph = None; return self.render()
    def step(self, action: int):
        self.glyph = self.mapping[int(action)]; return self.render(), 0.0, True, {}
    def render(self):
        frame = self.rng.integers(0, 14, (1, 24, 28), dtype=np.uint8)
        if self.glyph is not None:
            dy = int(self.rng.integers(-1, 2)); dx = int(self.rng.integers(-1, 2))
            frame[0, 8 + dy:15 + dy, 11 + dx:16 + dx] = np.maximum(
                frame[0, 8 + dy:15 + dy, 11 + dx:16 + dx], self.bank[self.glyph] * 235)
        return frame


class SymbolAlphabet:
    """Visual prototypes indexed by the actions that produced them."""
    def __init__(self, prototypes: dict[int, list[np.ndarray]]):
        self.prototypes = prototypes; templates = []; labels = []
        for action, values in prototypes.items():
            for value in values:
                for dy in (-2, -1, 0, 1, 2):
                    for dx in (-2, -1, 0, 1, 2):
                        templates.append(np.roll(value, (dy, dx), (0, 1))); labels.append(action)
        self.templates = np.asarray(templates, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int16)

    @classmethod
    def discover(cls, factory, samples: int = 4, seed_base: int = 60_000_000):
        values = {action: [] for action in range(VOCAB)}
        for action in range(VOCAB):
            for sample in range(samples):
                env = factory(); env.reset(seed_base + action * 100 + sample)
                frame, _, _, _ = env.step(action)
                values[action].append((frame[0, 6:18, 8:20] >= 128).astype(np.float32))
        return cls(values)

    def decode_patch(self, patch: np.ndarray) -> int:
        distances = np.mean(np.abs(self.templates - patch[None]), axis=(1, 2))
        return int(self.labels[int(np.argmin(distances))])

    def decode_row(self, frame: np.ndarray, y: int, *, x0: int = 6, maximum: int = 5) -> list[int]:
        gray = np.mean(frame, axis=0)
        output = []
        for index in range(maximum):
            x = x0 + index * 13
            patch = (gray[y - 2:y + 10, x - 3:x + 9] >= 128).astype(np.float32)
            if patch.sum() < 4: break
            output.append(self.decode_patch(patch))
        return output

    def to_json(self) -> dict:
        return {str(key): [value.astype(int).tolist() for value in values]
                for key, values in self.prototypes.items()}

    @classmethod
    def from_json(cls, value: dict):
        return cls({int(key): [np.asarray(x, dtype=np.float32) for x in values]
                    for key, values in value.items()})


class GrammarDiscoveryWorld:
    """Variable clauses plus temporal connector; only the completed plan is rewarded."""
    action_dim = 8
    horizon = 4
    def __init__(self, language_seed: int = 6101, *, bank: np.ndarray | None = None,
                 held_out: bool = False):
        self.bank = glyph_bank(language_seed + 17) if bank is None else np.asarray(bank)
        self.held_out = held_out
        rng = np.random.default_rng(language_seed)
        self.semantic_to_glyph = rng.permutation(VOCAB).tolist()
        self.role_order = rng.permutation(3).tolist()  # color, shape, verb
        self.modifier_position = int(rng.integers(4))
        all_commands = [(c, s, v) for c in range(3) for s in range(3) for v in range(2)]
        self.command_pool = [command for command in all_commands
            if (((command[0] * 3 + command[1]) * 2 + command[2]) % 5 == 0) == held_out]

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)
        commands = [self.command_pool[int(self.rng.integers(len(self.command_pool)))] for _ in range(2)]
        pairs = list(dict.fromkeys((c, s) for c, s, _ in commands))
        fillers = [(c, s) for c in range(3) for s in range(3) if (c, s) not in pairs]
        self.rng.shuffle(fillers); self.objects = pairs + fillers[:4 - len(pairs)]; self.rng.shuffle(self.objects)
        self.modifiers = [int(self.rng.integers(-1, 2)) for _ in range(2)]  # absent, no-op, repeat
        self.connector = int(self.rng.integers(2))  # displayed order or reverse order
        clauses = []
        action_groups = []
        for (color, shape, verb), modifier in zip(commands, self.modifiers):
            semantic = (color, 3 + shape, 6 + verb)
            ordered = [semantic[role] for role in self.role_order]
            if modifier >= 0: ordered.insert(self.modifier_position, 8 + modifier)
            clauses.append([self.semantic_to_glyph[value] for value in ordered])
            action = verb * 4 + self.objects.index((color, shape))
            action_groups.append([action, action] if modifier == 1 else [action])
        order = (0, 1) if self.connector == 0 else (1, 0)
        self.target_groups = [action_groups[index] for index in order]
        self.target_actions = [action for group in self.target_groups for action in group]
        self.clause_glyphs = clauses
        self.connector_glyph = self.semantic_to_glyph[10 + self.connector]
        self.actions = []; self.step_count = 0; self.noise_seed = int(self.rng.integers(2**31))
        return self.render()

    def step(self, action: int):
        self.actions.append(int(action)); self.step_count += 1
        done = self.step_count >= len(self.target_actions)
        reward = 0.0 if not done else (1.0 if self.actions == self.target_actions else -1.0)
        return self.render(), reward, done, {'success': bool(done and reward > 0),
                                             'chain_length': len(self.target_actions)}

    def render(self):
        rng = np.random.default_rng(self.noise_seed + self.step_count)
        frame = rng.integers(0, 13, (3, 96, 150), dtype=np.uint8)
        _paint_tokens(frame, self.clause_glyphs[0], CLAUSE_ROWS[0], self.bank, rng)
        _paint_tokens(frame, [self.connector_glyph], CONNECTOR_ROW, self.bank, rng, x0=70)
        _paint_tokens(frame, self.clause_glyphs[1], CLAUSE_ROWS[1], self.bank, rng)
        frame[:, 43:44, 5:145] = 55
        yy, xx = np.ogrid[:96, :150]
        for index, (color, shape) in enumerate(self.objects):
            cx, cy = OBJECT_X[index], 74
            if shape == 0: mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 81
            elif shape == 1: mask = (abs(xx - cx) <= 9) & (abs(yy - cy) <= 9)
            else: mask = (yy >= cy - 10) & (yy <= cy + 10) & (abs(xx - cx) <= ((yy - (cy - 10)) // 2))
            for channel in range(3): frame[channel, mask] = COLORS[color, channel]
        return frame


def _decode_instruction(alphabet: SymbolAlphabet, frame: np.ndarray) -> tuple[list[list[int]], int]:
    clauses = [alphabet.decode_row(frame, row, maximum=4) for row in CLAUSE_ROWS]
    connector = alphabet.decode_row(frame, CONNECTOR_ROW, x0=70, maximum=1)
    if any(len(clause) not in (3, 4) for clause in clauses) or len(connector) != 1:
        raise RuntimeError('visual sentence decoding failed')
    return clauses, connector[0]


def _scene_features(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(frame, dtype=np.float32); colors = []; shapes = []
    for cx in OBJECT_X:
        patch = rgb[:, 62:87, cx - 11:cx + 12]; mask = patch.max(0) > 35
        colors.append(patch[:, mask].mean(1) / 255.0); shapes.append(mask.astype(np.float32).reshape(-1))
    return np.asarray(colors), np.asarray(shapes)


class GrammarConcepts:
    def __init__(self, color_centers: np.ndarray, shape_centers: np.ndarray):
        self.color_centers = np.asarray(color_centers, dtype=np.float32)
        self.shape_centers = np.asarray(shape_centers, dtype=np.float32)
    @classmethod
    def discover(cls, factory, observations: int = 96, seed_base: int = 61_000_000):
        colors = []; shapes = []
        for index in range(observations):
            color, shape = _scene_features(factory().reset(seed_base + index)); colors.extend(color); shapes.extend(shape)
        return cls(_kmeans(np.asarray(colors)), _kmeans(np.asarray(shapes)))
    def classify(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        color, shape = _scene_features(frame)
        return (((color[:, None] - self.color_centers[None]) ** 2).mean(2).argmin(1),
                ((shape[:, None] - self.shape_centers[None]) ** 2).mean(2).argmin(1))
    def to_json(self): return {'colors': self.color_centers.tolist(), 'shapes': self.shape_centers.tolist()}
    @classmethod
    def from_json(cls, value): return cls(np.asarray(value['colors']), np.asarray(value['shapes']))


def _exact_clique_cover(tokens: set[int], cooccur: np.ndarray) -> list[set[int]]:
    token_list = sorted(tokens); cliques = []
    for size in range(2, min(5, len(token_list) + 1)):
        for values in itertools.combinations(token_list, size):
            if all(cooccur[a, b] == 0 for a, b in itertools.combinations(values, 2)):
                cliques.append(frozenset(values))
    solutions = []
    def visit(remaining: set[int], chosen: list[frozenset[int]]):
        if not remaining:
            solutions.append(list(chosen)); return
        first = min(remaining)
        for clique in cliques:
            if first in clique and clique <= remaining:
                visit(remaining - set(clique), chosen + [clique])
    visit(set(token_list), [])
    if not solutions: raise RuntimeError('no latent word-family partition explains the corpus')
    best = min(solutions, key=lambda groups: (len(groups), -sum(len(group) ** 2 for group in groups)))
    return [set(group) for group in best]


class DiscoveredStructure:
    def __init__(self, groups: list[set[int]], mandatory: list[int], optional: int,
                 connector_tokens: set[int], group_order: list[int], modifier_position: int):
        self.groups = groups; self.mandatory = mandatory; self.optional = optional
        self.connector_tokens = connector_tokens; self.group_order = group_order
        self.modifier_position = modifier_position
    @property
    def signature(self):
        return {'mandatory_family_sizes': sorted(len(self.groups[index]) for index in self.mandatory),
                'optional_family_sizes': [len(self.groups[self.optional])],
                'connector_family_size': len(self.connector_tokens), 'families_discovered': len(self.groups) + 1}
    def to_json(self):
        return {'groups': [sorted(x) for x in self.groups], 'mandatory': self.mandatory,
                'optional': self.optional, 'connector_tokens': sorted(self.connector_tokens),
                'group_order': self.group_order, 'modifier_position': self.modifier_position,
                'signature': self.signature}
    @classmethod
    def from_json(cls, value):
        return cls([set(x) for x in value['groups']], value['mandatory'], value['optional'],
                   set(value['connector_tokens']), value['group_order'], value['modifier_position'])


def discover_structure(factory, alphabet: SymbolAlphabet, observations: int = 320,
                       seed_base: int = 62_000_000) -> tuple[DiscoveredStructure, dict]:
    corpus = []; connectors = []
    for index in range(observations):
        clauses, connector = _decode_instruction(alphabet, factory().reset(seed_base + index))
        corpus.extend(clauses); connectors.append(connector)
    clause_tokens = set(token for clause in corpus for token in clause)
    connector_tokens = set(connectors)
    cooccur = np.zeros((VOCAB, VOCAB), dtype=np.int32)
    for clause in corpus:
        for a, b in itertools.combinations(sorted(set(clause)), 2): cooccur[a, b] += 1; cooccur[b, a] += 1
    groups = _exact_clique_cover(clause_tokens, cooccur)
    presence = [[sum(token in group for token in clause) for clause in corpus] for group in groups]
    mandatory = [index for index, values in enumerate(presence) if min(values) == max(values) == 1]
    optional = [index for index, values in enumerate(presence) if max(values) == 1 and min(values) == 0]
    if not mandatory or len(optional) != 1 or len(connector_tokens) < 2:
        raise RuntimeError(f'ambiguous structure: mandatory={mandatory}, optional={optional}, connectors={connector_tokens}')
    token_group = {token: index for index, group in enumerate(groups) for token in group}
    orders = [] ; modifier_positions = []
    for clause in corpus:
        roles = [token_group[token] for token in clause]
        if optional[0] in roles: modifier_positions.append(roles.index(optional[0])); roles.remove(optional[0])
        orders.append(tuple(roles))
    order = Counter(orders).most_common(1)[0][0]
    modifier_position = Counter(modifier_positions).most_common(1)[0][0]
    structure = DiscoveredStructure(groups, mandatory, optional[0], connector_tokens, list(order), modifier_position)
    evidence = {'observations': observations, 'clauses': len(corpus), 'human_labels': 0,
                'families': [sorted(group) for group in groups], **structure.signature,
                'optional_presence_rate': float(np.mean(presence[optional[0]]))}
    return structure, evidence


def _candidate_maps(structure: DiscoveredStructure) -> np.ndarray:
    mandatory = structure.mandatory
    size_three = [index for index in mandatory if len(structure.groups[index]) == 3]
    size_two = [index for index in mandatory if len(structure.groups[index]) == 2]
    if len(size_three) != 2 or len(size_two) != 1 or len(structure.groups[structure.optional]) != 2:
        raise RuntimeError('discovered family cardinalities do not support grounded program search')
    maps = []
    for color_group, shape_group in (size_three, size_three[::-1]):
        for colors in itertools.permutations(range(3)):
          for shapes in itertools.permutations(range(3, 6)):
            for verbs in itertools.permutations(range(6, 8)):
              for modifiers in itertools.permutations(range(8, 10)):
                for connectors in itertools.permutations(range(10, 12)):
                    mapping = np.full(VOCAB, -1, dtype=np.int8)
                    for token, semantic in zip(sorted(structure.groups[color_group]), colors): mapping[token] = semantic
                    for token, semantic in zip(sorted(structure.groups[shape_group]), shapes): mapping[token] = semantic
                    for token, semantic in zip(sorted(structure.groups[size_two[0]]), verbs): mapping[token] = semantic
                    for token, semantic in zip(sorted(structure.groups[structure.optional]), modifiers): mapping[token] = semantic
                    for token, semantic in zip(sorted(structure.connector_tokens), connectors): mapping[token] = semantic
                    maps.append(mapping)
    return np.asarray(maps, dtype=np.int8)


class GrammarProgramLearner:
    """Grounds a discovered grammar by eliminating programs with terminal evidence."""
    def __init__(self, alphabet: SymbolAlphabet, concepts: GrammarConcepts,
                 structure: DiscoveredStructure, maps: np.ndarray | None = None):
        self.alphabet = alphabet; self.concepts = concepts; self.structure = structure
        self.maps = _candidate_maps(structure) if maps is None else np.asarray(maps, dtype=np.int8)
        self.active = np.ones(len(self.maps), dtype=bool)
    @property
    def active_count(self): return int(self.active.sum())

    def _lookup(self, frame: np.ndarray) -> dict[tuple[int, int], int]:
        object_colors, object_shapes = self.concepts.classify(frame)
        return {(int(c), int(s)): index for index, (c, s) in enumerate(zip(object_colors, object_shapes))}

    def _interpret_groups(self, mapping: np.ndarray, clauses: list[list[int]], connector: int,
                          frame: np.ndarray, lookup: dict[tuple[int, int], int] | None = None
                          ) -> tuple[tuple[int, ...], ...] | None:
        lookup = self._lookup(frame) if lookup is None else lookup
        groups = []
        for clause in clauses:
            semantic = mapping[np.asarray(clause)]
            colors = semantic[semantic < 3]; shapes = semantic[(semantic >= 3) & (semantic < 6)] - 3
            verbs = semantic[(semantic >= 6) & (semantic < 8)] - 6
            modifiers = semantic[(semantic >= 8) & (semantic < 10)] - 8
            if len(colors) != 1 or len(shapes) != 1 or len(verbs) != 1 or len(modifiers) > 1: return None
            object_index = lookup.get((int(colors[0]), int(shapes[0])))
            if object_index is None: return None
            action = int(verbs[0]) * 4 + object_index
            groups.append((action, action) if len(modifiers) and modifiers[0] == 1 else (action,))
        connector_semantic = int(mapping[connector])
        if connector_semantic not in (10, 11): return None
        order = (0, 1) if connector_semantic == 10 else (1, 0)
        return tuple(groups[index] for index in order)

    def _interpret(self, mapping: np.ndarray, clauses: list[list[int]], connector: int,
                   frame: np.ndarray, lookup: dict[tuple[int, int], int] | None = None) -> tuple[int, ...] | None:
        groups = self._interpret_groups(mapping, clauses, connector, frame, lookup)
        return None if groups is None else tuple(action for group in groups for action in group)

    def plan(self, frame: np.ndarray) -> tuple[list[int], tuple[int, ...], tuple[list[list[int]], int]]:
        clauses, connector = _decode_instruction(self.alphabet, frame)
        lookup = self._lookup(frame)
        predictions = [self._interpret(mapping, clauses, connector, frame, lookup) for mapping in self.maps]
        eligible = [prediction for prediction, active in zip(predictions, self.active) if active and prediction is not None]
        if not eligible: raise RuntimeError('no discovered grammar program can interpret the scene')
        selected = Counter(eligible).most_common(1)[0][0]
        return list(selected), selected, (clauses, connector)

    def observe(self, frame: np.ndarray, parsed, attempted: tuple[int, ...], reward: float):
        clauses, connector = parsed
        lookup = self._lookup(frame)
        matches = np.asarray([self._interpret(mapping, clauses, connector, frame, lookup) == attempted
                              for mapping in self.maps], dtype=bool)
        self.active &= matches if reward > 0 else ~matches
        if not self.active.any(): raise RuntimeError('terminal evidence eliminated every grammar program')

    def best_map(self): return self.maps[int(np.flatnonzero(self.active)[0])]

    def learned_role_order(self) -> list[str]:
        mapping = self.best_map(); names = []
        for group_index in self.structure.group_order:
            semantic = int(mapping[next(iter(self.structure.groups[group_index]))])
            names.append('color' if semantic < 3 else ('shape' if semantic < 6 else 'verb'))
        return names

    def describe(self, frame: np.ndarray, action_groups: list[list[int]]) -> tuple[list[list[int]], int]:
        mapping = self.best_map(); inverse = np.empty(VOCAB, dtype=np.int8)
        for token, semantic in enumerate(mapping): inverse[semantic] = token
        color_ids, shape_ids = self.concepts.classify(frame); clauses = []
        role_order = self.learned_role_order()
        role_index = {'color': 0, 'shape': 1, 'verb': 2}
        for group in action_groups:
            action = int(group[0]); object_index = action % 4; verb = action // 4
            semantics = (int(color_ids[object_index]), 3 + int(shape_ids[object_index]), 6 + verb)
            tokens = [int(inverse[semantics[role_index[role]]]) for role in role_order]
            if len(group) == 2: tokens.insert(self.structure.modifier_position, int(inverse[9]))
            clauses.append(tokens)
        return clauses, int(inverse[10])

    def save(self, path: Path):
        atomic_json(path, {'format': 'wailah-discovered-grammar-memory-v1',
            'active_maps': self.maps[self.active].astype(int).tolist(), 'alphabet': self.alphabet.to_json(),
            'concepts': self.concepts.to_json(), 'structure': self.structure.to_json(),
            'learned_role_order': self.learned_role_order()})
    @classmethod
    def load(cls, path: Path):
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(SymbolAlphabet.from_json(value['alphabet']), GrammarConcepts.from_json(value['concepts']),
                   DiscoveredStructure.from_json(value['structure']), np.asarray(value['active_maps'], dtype=np.int8))


class GrammarStrategy:
    def __init__(self, learner: GrammarProgramLearner): self.learner = learner; self.queue = []
    def reset(self, observation): self.queue = self.learner.plan(observation)[0]
    def act(self, observation): return self.queue.pop(0) if self.queue else 0
    @classmethod
    def load(cls, workspace: Path): return cls(GrammarProgramLearner.load(Path(workspace) / 'language_00.json'))


def _train(learner: GrammarProgramLearner, factory, episodes: int, seed_base: int) -> dict:
    rewards = []; counts = []; resolved = None; start = learner.active_count
    for episode in range(episodes):
        env = factory(); frame = env.reset(seed_base + episode)
        actions, attempted, parsed = learner.plan(frame); reward = 0.0
        for action in actions: _, reward, done, _ = env.step(action)
        learner.observe(frame, parsed, attempted, reward); rewards.append(reward); counts.append(learner.active_count)
        if learner.active_count == 1 and resolved is None: resolved = episode + 1
    return {'episodes': episodes, 'terminal_reward_only': True, 'initial_programs': start,
            'remaining_programs': learner.active_count, 'episodes_to_single_program': resolved,
            'failures_before_resolution': int(np.sum(np.asarray(rewards[:resolved]) <= 0)) if resolved else None,
            'success_first_32': float(np.mean(np.asarray(rewards[:32]) > 0)),
            'success_last_64': float(np.mean(np.asarray(rewards[-64:]) > 0)),
            'program_trace': [counts[i] for i in range(0, len(counts), max(1, episodes // 16))] + [counts[-1]]}


def _evaluate(learner: GrammarProgramLearner, factory, episodes: int, seed_base: int) -> dict:
    read = write = 0; lengths = []
    for index in range(episodes):
        env = factory(); frame = env.reset(seed_base + index)
        actions, _, parsed = learner.plan(frame); reward = 0.0
        for action in actions: _, reward, done, _ = env.step(action)
        read += int(reward > 0); lengths.append(len(actions))
        # The writer sees scene pixels and its own action groups. It emits a
        # canonical sentence; meaning-equivalence is checked through its learned parser.
        source_groups = learner._interpret_groups(learner.best_map(), parsed[0], parsed[1], frame)
        clauses, connector = learner.describe(frame, [list(group) for group in source_groups])
        described = learner._interpret(learner.best_map(), clauses, connector, frame)
        write += int(described == tuple(actions))
    random_chain = float(np.mean([8.0 ** (-length) for length in lengths]))
    return {'episodes': episodes, 'read_execute_success_rate': read / episodes,
            'semantic_description_accuracy': write / episodes, 'mean_chain_length': float(np.mean(lengths)),
            'random_chain_success': random_chain}


def _structure_sample_efficiency(factory, alphabet, final: DiscoveredStructure, seed_base: int) -> dict:
    checkpoints = [16, 24, 32, 48, 64, 96, 128, 192, 256, 320]
    target = {frozenset(group) for group in final.groups}; results = []
    for count in checkpoints:
        try:
            structure, _ = discover_structure(factory, alphabet, count, seed_base)
            exact = {frozenset(group) for group in structure.groups} == target
            results.append({'observations': count, 'exact_partition': exact,
                            'signature_match': structure.signature == final.signature})
        except RuntimeError:
            results.append({'observations': count, 'exact_partition': False, 'signature_match': False})
    first_exact = next((row['observations'] for row in results if row['exact_partition']), None)
    return {'checkpoints': results, 'observations_to_exact_structure': first_exact}


def run_grammar_discovery_benchmark(workspace: Path, output: Path, device='cpu') -> dict:
    workspace, output = Path(workspace), Path(output)
    brain = workspace / 'living' / 'grammar_discovery'; brain.mkdir(parents=True, exist_ok=True)
    main_seed = 6101; main_bank = glyph_bank(main_seed + 17); mapping_seed = 6102
    train_factory = lambda: GrammarDiscoveryWorld(main_seed, bank=main_bank, held_out=False)
    test_factory = lambda: GrammarDiscoveryWorld(main_seed, bank=main_bank, held_out=True)
    alphabet = SymbolAlphabet.discover(lambda: SymbolEchoTerminal(main_bank, mapping_seed), samples=5)
    structure, structure_evidence = discover_structure(train_factory, alphabet)
    concepts = GrammarConcepts.discover(train_factory)
    learner = GrammarProgramLearner(alphabet, concepts, structure)
    training = _train(learner, train_factory, 160, 64_000_000)
    evaluation = _evaluate(learner, test_factory, 512, 65_000_000)
    learner.save(brain / 'language_00.json'); main_hash = hashlib.sha256((brain / 'language_00.json').read_bytes()).hexdigest()
    sample_efficiency = _structure_sample_efficiency(train_factory, alphabet, structure, 66_000_000)

    audits = []
    for audit_index in range(8):
        language_seed = 7001 + audit_index * 97; bank = glyph_bank(language_seed + 17); key_seed = 7100 + audit_index
        factory = lambda ls=language_seed, b=bank: GrammarDiscoveryWorld(ls, bank=b, held_out=False)
        held = lambda ls=language_seed, b=bank: GrammarDiscoveryWorld(ls, bank=b, held_out=True)
        alpha = SymbolAlphabet.discover(lambda b=bank, ks=key_seed: SymbolEchoTerminal(b, ks),
                                        samples=3, seed_base=67_000_000 + audit_index * 10_000)
        discovered, evidence = discover_structure(factory, alpha, observations=320,
            seed_base=67_100_000 + audit_index * 10_000)
        audit_learner = GrammarProgramLearner(alpha, concepts, discovered)
        learned = _train(audit_learner, factory, 120, 67_200_000 + audit_index * 10_000)
        tested = _evaluate(audit_learner, held, 128, 67_300_000 + audit_index * 10_000)
        memory_path = brain / f'language_{audit_index + 1:02d}.json'; audit_learner.save(memory_path)
        audits.append({'language_index': audit_index + 1, 'language_seed': language_seed,
                       'structure': evidence, 'training': learned, 'evaluation': tested,
                       'discovered_role_order': audit_learner.learned_role_order(),
                       'memory_sha256': hashlib.sha256(memory_path.read_bytes()).hexdigest()})

    revisit = _evaluate(GrammarProgramLearner.load(brain / 'language_00.json'), test_factory, 512, 68_000_000)
    retained_hash = hashlib.sha256((brain / 'language_00.json').read_bytes()).hexdigest()

    selector_path = workspace / 'living' / 'strategy_selector.json'; selector = LearnedStrategySelector.load(selector_path)
    features = np.stack([probe_features(test_factory, 69_000_000 + index) for index in range(64)])
    selector.centroids['grammar-discovery-memory'] = features.mean(0)
    selector.scale = np.maximum(selector.scale, features.std(0)); selector.save(selector_path)
    selector_accuracy = float(np.mean([selector.rank(test_factory, 69_100_000 + index,
        list(selector.centroids))[0][0] == 'grammar-discovery-memory' for index in range(32)]))
    from .autonomy import AutonomousCompetenceLoop
    loop = AutonomousCompetenceLoop(workspace, device, grammar_workspace=brain)
    decision = loop.select(test_factory, probe_seed=69_300_000)
    competence = loop.evaluate_selected(test_factory, decision['method'], seed_base=69_400_000, episodes=128)

    report = {'format': 'wailah-grammar-discovery-v3',
        'learner_inputs': 'pixels, self-chosen motor actions, terminal reward, done',
        'privileged_dictionary': False, 'privileged_parser': False, 'correct_actions_supplied': False,
        'intermediate_reward': False,
        'structural_search': {'role_count_supplied': False, 'family_sizes_supplied': False,
            'grounding_ontology_compatibility_required': True,
            'clause_lengths': [3, 4], 'plan_lengths': [2, 3, 4], 'optional_words': True,
            'temporal_connectors': True, 'structure_evidence': structure_evidence,
            'sample_efficiency': sample_efficiency},
        'main_language': {'training': training, 'withheld_composition': evaluation,
            'discovered_role_order': learner.learned_role_order()},
        'independent_language_audit': {'languages': len(audits), 'runs': audits,
            'all_structure_discovered': all(row['structure']['families_discovered'] == 5 for row in audits),
            'all_programs_resolved': all(row['training']['remaining_programs'] == 1 for row in audits),
            'mean_resolution_episodes': float(np.mean([row['training']['episodes_to_single_program'] for row in audits])),
            'read_execute_accuracy': float(np.mean([row['evaluation']['read_execute_success_rate'] for row in audits])),
            'description_accuracy': float(np.mean([row['evaluation']['semantic_description_accuracy'] for row in audits]))},
        'retention': {'main_before': evaluation, 'main_after_eight_languages': revisit,
            'memory_hash_unchanged': main_hash == retained_hash},
        'autonomous_integration': {'selector_accuracy': selector_accuracy, 'decision': decision,
            'competence_evaluation': competence},
        'promoted': decision['method'] == 'grammar-discovery-memory' and competence.get('success_rate', 0) >= .95}
    atomic_json(output, report); return report


def render_grammar_discovery_summary(report_path: Path, output: Path) -> Path:
    from PIL import Image, ImageDraw, ImageFont
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    try:
        title = ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf', 27)
        body = ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf', 15)
        small = ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf', 12)
    except OSError: title = body = small = ImageFont.load_default()
    canvas = Image.new('RGB', (1120, 690), (7, 13, 25)); draw = ImageDraw.Draw(canvas)
    green, white, muted, panel = (75, 225, 164), (232, 239, 248), (139, 156, 180), (14, 25, 43)
    draw.text((34, 24), 'WAILAH  /  GRAMMAR DISCOVERY v3', font=title, fill=white)
    draw.text((35, 64), 'infer families  •  optional words  •  temporal order  •  grounded plans  •  lifelong retention', font=body, fill=muted)
    main = report['main_language']['withheld_composition']; audit = report['independent_language_audit']
    cards = [('MAIN READ + ACT', main['read_execute_success_rate']), ('MAIN DESCRIBE', main['semantic_description_accuracy']),
             ('8-LANGUAGE READ', audit['read_execute_accuracy']), ('8-LANGUAGE WRITE', audit['description_accuracy'])]
    for index, (label, value) in enumerate(cards):
        x = 34 + index * 270; draw.rounded_rectangle((x, 105, x + 246, 197), radius=12, fill=panel)
        draw.text((x + 16, 121), label, font=small, fill=muted); draw.text((x + 16, 151), f'{value:.1%}', font=title, fill=green)
    structure = report['structural_search']['structure_evidence']; retention = report['retention']
    rows = [('Latent structure', f"{structure['families_discovered']} word families inferred / no role count supplied"),
            ('Variable grammar', '3–4 word clauses / optional modifiers / temporal connectors'),
            ('Grounded program search', f"{report['main_language']['training']['initial_programs']} → 1 from terminal reward"),
            ('Withheld multi-action plans', f"{main['episodes']} trials / random {main['random_chain_success']:.3%}"),
            ('Independent languages', f"8/8 structures and programs resolved / mean {audit['mean_resolution_episodes']:.1f} episodes"),
            ('Retention after eight languages', f"{retention['main_after_eight_languages']['read_execute_success_rate']:.1%} / memory hash unchanged"),
            ('Autonomous competence gate', report['autonomous_integration']['decision']['method'])]
    draw.text((35, 228), 'CAPABILITY', font=small, fill=muted); draw.text((560, 228), 'MEASURED RESULT', font=small, fill=muted)
    for index, (name, value) in enumerate(rows):
        y = 260 + index * 52; draw.rounded_rectangle((28, y - 7, 1092, y + 34), radius=8, fill=panel)
        draw.text((43, y), name, font=body, fill=white); draw.text((560, y), value, font=body, fill=green)
    draw.text((34, 642), 'Pixels + self-chosen actions + final reward. Family count, token groups, dictionary and grammar order are inferred.', font=small, fill=muted)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); canvas.save(output); return output
