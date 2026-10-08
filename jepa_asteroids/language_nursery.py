"""Reward-grounded language induction, multi-step execution and production."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .literacy import ARTIFICIAL, COLORS, GLYPHS, LiteracyTerminal, VisualAlphabet, _cells, _paint_sequence
from .runtime import atomic_json
from .strategy_selector import LearnedStrategySelector, probe_features


ROWS = (4, 15, 26)
OBJECT_X = (14, 41, 68, 95)


class LanguageNurseryWorld:
    """An unknown compositional language controlling delayed three-step behavior."""
    action_dim = 8
    horizon = 3

    def __init__(self, language_seed: int = 4101, *, bank=ARTIFICIAL,
                 held_out: bool = False, steps: int | None = None):
        self.bank = np.asarray(bank)
        self.held_out = held_out
        self.fixed_steps = steps
        rng = np.random.default_rng(language_seed)
        self.semantic_to_glyph = rng.permutation(8).tolist()
        self.grammar = rng.permutation(3).tolist()  # semantic roles: color, shape, verb
        all_commands = [(c, s, v) for c in range(3) for s in range(3) for v in range(2)]
        self.commands = [x for x in all_commands if (((x[0] * 3 + x[1]) * 2 + x[2]) % 5 == 0) == held_out]

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)
        count = self.fixed_steps or int(self.rng.integers(1, 4))
        selected: list[tuple[int, int, int]] = []
        for _ in range(200):
            candidate = self.commands[int(self.rng.integers(len(self.commands)))]
            pairs = {(c, s) for c, s, _ in selected + [candidate]}
            if len(pairs) <= 4:
                selected.append(candidate)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise RuntimeError('could not construct nursery episode')
        pairs = list(dict.fromkeys((c, s) for c, s, _ in selected))
        fillers = [(c, s) for c in range(3) for s in range(3) if (c, s) not in pairs]
        self.rng.shuffle(fillers)
        self.objects = pairs + fillers[:4 - len(pairs)]
        self.rng.shuffle(self.objects)
        self.command_semantics = selected
        self.target_actions = [v * 4 + self.objects.index((c, s)) for c, s, v in selected]
        self.instruction_glyphs = []
        for command in selected:
            semantic = (command[0], 3 + command[1], 6 + command[2])
            self.instruction_glyphs.append([self.semantic_to_glyph[semantic[role]] for role in self.grammar])
        self.actions = []
        self.step_count = 0
        self.noise_seed = int(self.rng.integers(2**31))
        return self.render()

    def step(self, action: int):
        self.actions.append(int(action)); self.step_count += 1
        done = self.step_count >= len(self.target_actions)
        reward = 0.0 if not done else (1.0 if self.actions == self.target_actions else -1.0)
        return self.render(), reward, done, {'success': bool(done and reward > 0),
                                             'chain_length': len(self.target_actions)}

    def render(self):
        rng = np.random.default_rng(self.noise_seed + self.step_count)
        frame = rng.integers(0, 13, (3, 78, 112), dtype=np.uint8)
        for row, glyphs in zip(ROWS, self.instruction_glyphs):
            _paint_sequence(frame, glyphs, row, self.bank, rng, channel=slice(None), jitter=True)
        frame[:, 38:39, 4:108] = 55
        yy, xx = np.ogrid[:78, :112]
        for index, (color, shape) in enumerate(self.objects):
            cx, cy = OBJECT_X[index], 59
            if shape == 0:
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 64
            elif shape == 1:
                mask = (abs(xx - cx) <= 8) & (abs(yy - cy) <= 8)
            else:
                mask = (yy >= cy - 9) & (yy <= cy + 9) & (abs(xx - cx) <= ((yy - (cy - 9)) // 2))
            for channel in range(3):
                frame[channel, mask] = COLORS[color, channel]
        return frame


def _instruction_tokens(alphabet: VisualAlphabet, frame: np.ndarray) -> list[list[int]]:
    gray = np.mean(frame, axis=0, keepdims=True).astype(np.uint8)
    commands = []
    for row in ROWS:
        tokens = alphabet.decode(gray, row)
        if len(tokens) >= 3:
            commands.append(tokens[:3])
    return commands


def _scene_features(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(frame, dtype=np.float32)
    colors, shapes = [], []
    for cx in OBJECT_X:
        patch = rgb[:, 48:70, cx - 10:cx + 11]
        mask = patch.max(0) > 35
        colors.append(patch[:, mask].mean(1) / 255.0)
        shapes.append(mask.astype(np.float32).reshape(-1))
    return np.asarray(colors), np.asarray(shapes)


def _kmeans(values: np.ndarray, clusters: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centers = [values[0]]
    while len(centers) < clusters:
        distance = np.min(((values[:, None] - np.stack(centers)[None]) ** 2).mean(2), axis=1)
        centers.append(values[int(np.argmax(distance))])
    centers = np.stack(centers)
    for _ in range(30):
        labels = ((values[:, None] - centers[None]) ** 2).mean(2).argmin(1)
        updated = np.stack([values[labels == index].mean(0) if np.any(labels == index) else centers[index]
                            for index in range(clusters)])
        if np.allclose(updated, centers):
            break
        centers = updated
    return centers


class VisualConcepts:
    """Unlabeled color/shape clusters discovered from object pixels."""
    def __init__(self, color_centers: np.ndarray, shape_centers: np.ndarray):
        self.color_centers = np.asarray(color_centers, dtype=np.float32)
        self.shape_centers = np.asarray(shape_centers, dtype=np.float32)

    @classmethod
    def discover(cls, factory, observations: int = 96, seed_base: int = 41_000_000):
        colors, shapes = [], []
        for index in range(observations):
            frame = factory().reset(seed_base + index)
            color, shape = _scene_features(frame); colors.extend(color); shapes.extend(shape)
        return cls(_kmeans(np.asarray(colors)), _kmeans(np.asarray(shapes)))

    def classify(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        color, shape = _scene_features(frame)
        color_ids = ((color[:, None] - self.color_centers[None]) ** 2).mean(2).argmin(1)
        shape_ids = ((shape[:, None] - self.shape_centers[None]) ** 2).mean(2).argmin(1)
        return color_ids, shape_ids

    def to_json(self) -> dict:
        return {'color_centers': self.color_centers.tolist(), 'shape_centers': self.shape_centers.tolist()}

    @classmethod
    def from_json(cls, value: dict):
        return cls(np.asarray(value['color_centers']), np.asarray(value['shape_centers']))


class VersionSpaceLanguage:
    """Infers token meanings from terminal reward over complete action chains."""
    def __init__(self, alphabet: VisualAlphabet, concepts: VisualConcepts,
                 maps: np.ndarray | None = None):
        self.alphabet = alphabet; self.concepts = concepts
        self.maps = (np.asarray(maps, dtype=np.int8) if maps is not None else
                     np.asarray(list(itertools.permutations(range(8))), dtype=np.int8))
        self.active = np.ones(len(self.maps), dtype=bool)
        self.reward_episodes = 0; self.successes = 0; self.last_grammar = None

    @property
    def active_count(self) -> int:
        return int(self.active.sum())

    def restrict_schema(self, position_sets: list[set[int]]) -> int:
        """Transfer the learned one-role-per-slot schema, without word meanings."""
        role = np.asarray((0, 0, 0, 1, 1, 1, 2, 2), dtype=np.int8)
        keep = np.ones(len(self.maps), dtype=bool)
        for mapping_index, mapping in enumerate(self.maps):
            assigned = []
            for tokens in position_sets:
                roles = {int(role[mapping[token]]) for token in tokens}
                if len(roles) != 1:
                    keep[mapping_index] = False; break
                assigned.append(next(iter(roles)))
            if keep[mapping_index] and len(set(assigned)) != 3:
                keep[mapping_index] = False
        self.active &= keep
        return self.active_count

    def decode(self, frame: np.ndarray) -> list[list[int]]:
        return _instruction_tokens(self.alphabet, frame)

    def _codes(self, tokens: list[list[int]], frame: np.ndarray) -> np.ndarray:
        token_array = np.asarray(tokens, dtype=np.int64)
        meanings = self.maps[:, token_array]
        color_mask = meanings < 3
        shape_mask = (meanings >= 3) & (meanings < 6)
        verb_mask = meanings >= 6
        valid = (color_mask.sum(2) == 1) & (shape_mask.sum(2) == 1) & (verb_mask.sum(2) == 1)
        colors = (meanings * color_mask).sum(2)
        shapes = ((meanings - 3) * shape_mask).sum(2)
        verbs = ((meanings - 6) * verb_mask).sum(2)
        object_colors, object_shapes = self.concepts.classify(frame)
        lookup = np.full((3, 3), -1, dtype=np.int8)
        for index, (color, shape) in enumerate(zip(object_colors, object_shapes)):
            lookup[int(color), int(shape)] = index
        objects = lookup[colors.clip(0, 2), shapes.clip(0, 2)]
        valid &= objects >= 0
        actions = verbs * 4 + objects
        codes = np.zeros(len(self.maps), dtype=np.int64)
        for step in range(len(tokens)):
            codes += (actions[:, step] + 1) * (9 ** step)
        codes[~valid.all(1)] = -1
        return codes

    @staticmethod
    def _actions(code: int, count: int) -> list[int]:
        actions = []
        for _ in range(count):
            actions.append(int(code % 9) - 1); code //= 9
        return actions

    def plan(self, frame: np.ndarray) -> tuple[list[int], int, list[list[int]]]:
        tokens = self.decode(frame); codes = self._codes(tokens, frame)
        eligible = codes[self.active & (codes >= 0)]
        if not len(eligible):
            raise RuntimeError('no grounded language hypothesis can act on this scene')
        values, counts = np.unique(eligible, return_counts=True)
        code = int(values[int(np.argmax(counts))])
        return self._actions(code, len(tokens)), code, tokens

    def observe(self, frame: np.ndarray, tokens: list[list[int]], code: int, reward: float) -> None:
        codes = self._codes(tokens, frame); predicted = codes == code
        self.active &= predicted if reward > 0 else ~predicted
        if not self.active.any():
            raise RuntimeError('terminal evidence eliminated every language hypothesis')
        self.reward_episodes += 1; self.successes += int(reward > 0)

    def best_map(self) -> np.ndarray:
        return self.maps[int(np.flatnonzero(self.active)[0])]

    def grammar(self, tokens: list[list[int]]) -> list[int]:
        mapping = self.best_map(); semantic = mapping[np.asarray(tokens[0])]
        roles = [0 if value < 3 else (1 if value < 6 else 2) for value in semantic]
        if sorted(roles) != [0, 1, 2]:
            raise RuntimeError('language grammar is not resolved')
        self.last_grammar = roles; return roles

    def describe(self, scene_frame: np.ndarray, actions: list[int], grammar: list[int]) -> list[list[int]]:
        color_ids, shape_ids = self.concepts.classify(scene_frame)
        mapping = self.best_map(); inverse = np.empty(8, dtype=np.int8)
        for token, semantic in enumerate(mapping): inverse[semantic] = token
        output = []
        for action in actions:
            object_index = int(action) % 4; verb = int(action) // 4
            semantic = (int(color_ids[object_index]), 3 + int(shape_ids[object_index]), 6 + verb)
            output.append([int(inverse[semantic[role]]) for role in grammar])
        return output

    def save(self, path: Path, grammar: list[int]) -> None:
        active_maps = self.maps[self.active]
        atomic_json(path, {'format': 'wailah-language-nursery-memory-v1',
                           'active_maps': active_maps.astype(int).tolist(),
                           'grammar': list(map(int, grammar)), 'concepts': self.concepts.to_json(),
                           'alphabet': {str(k): [x.astype(int).tolist() for x in v]
                                        for k, v in self.alphabet.prototypes.items()}})

    @classmethod
    def load(cls, path: Path):
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        alphabet = VisualAlphabet({int(k): [np.asarray(x, dtype=np.float32) for x in v]
                                   for k, v in value['alphabet'].items()})
        learner = cls(alphabet, VisualConcepts.from_json(value['concepts']),
                      np.asarray(value['active_maps'], dtype=np.int8))
        learner.last_grammar = value['grammar']; return learner


class NurseryStrategy:
    def __init__(self, learner: VersionSpaceLanguage): self.learner = learner; self.queue = []
    def reset(self, observation: np.ndarray): self.queue = self.learner.plan(observation)[0]
    def act(self, observation: np.ndarray) -> int: return self.queue.pop(0) if self.queue else 0
    @classmethod
    def load(cls, workspace: Path):
        return cls(VersionSpaceLanguage.load(Path(workspace) / 'language_one.json'))


def _position_sets(factory, alphabet: VisualAlphabet, samples: int = 64,
                   seed_base: int = 42_000_000) -> list[set[int]]:
    positions = [set(), set(), set()]
    for index in range(samples):
        frame = factory().reset(seed_base + index)
        for command in _instruction_tokens(alphabet, frame):
            for position, token in enumerate(command): positions[position].add(token)
    return positions


def _train_language(learner: VersionSpaceLanguage, factory, episodes: int, seed_base: int) -> dict:
    rewards, active_counts = [], []
    mastery_episode = resolved_episode = None
    starting_hypotheses = learner.active_count
    for episode in range(episodes):
        env = factory(); frame = env.reset(seed_base + episode)
        actions, code, tokens = learner.plan(frame)
        total = 0.0
        for action in actions:
            _, reward, done, _ = env.step(action); total += reward
        learner.observe(frame, tokens, code, total)
        rewards.append(total); active_counts.append(learner.active_count)
        if learner.active_count == 1 and resolved_episode is None:
            resolved_episode = episode + 1
        if episode >= 31 and mastery_episode is None and np.mean(np.asarray(rewards[-32:]) > 0) >= .90:
            mastery_episode = episode + 1
    return {'episodes': episodes, 'reward_only': True,
            'success_first_32': float(np.mean(np.asarray(rewards[:32]) > 0)),
            'success_last_64': float(np.mean(np.asarray(rewards[-64:]) > 0)),
            'episodes_to_90pct_rolling_32': mastery_episode,
            'episodes_to_single_hypothesis': resolved_episode,
            'failures_before_resolution': int(np.sum(np.asarray(rewards[:resolved_episode]) <= 0)) if resolved_episode else None,
            'initial_hypotheses': starting_hypotheses,
            'remaining_hypotheses': learner.active_count,
            'hypothesis_trace': [active_counts[i] for i in range(0, len(active_counts), max(1, episodes // 20))] + [active_counts[-1]]}


def _evaluate(learner: VersionSpaceLanguage, factory, episodes: int, seed_base: int) -> dict:
    read = write = 0; lengths = []
    for index in range(episodes):
        env = factory(); frame = env.reset(seed_base + index)
        actions, _, tokens = learner.plan(frame)
        for action in actions:
            _, reward, done, info = env.step(action)
        read += int(reward > 0); lengths.append(len(actions))
        grammar = learner.grammar(tokens)
        event_frame = frame.copy(); event_frame[:, :39] = 0
        # Production is conditioned on the agent's own executed motor trace, not
        # the environment's hidden target sequence.
        write += int(learner.describe(event_frame, actions, grammar) == tokens)
    mean_length = float(np.mean(lengths))
    return {'episodes': episodes, 'read_execute_success_rate': read / episodes,
            'event_description_write_accuracy': write / episodes,
            'mean_chain_length': mean_length,
            'random_chain_success': float(np.mean([8.0 ** (-length) for length in lengths])),
            'random_valid_description': float(np.mean([18.0 ** (-length) for length in lengths]))}


def run_language_nursery_benchmark(workspace: Path, output: Path, device='cpu') -> dict:
    workspace, output = Path(workspace), Path(output)
    brain = workspace / 'living' / 'language_nursery'; brain.mkdir(parents=True, exist_ok=True)

    language_one = lambda: LanguageNurseryWorld(4101, bank=ARTIFICIAL, held_out=False)
    language_one_test = lambda: LanguageNurseryWorld(4101, bank=ARTIFICIAL, held_out=True, steps=3)
    alphabet_one = VisualAlphabet.discover(lambda: LiteracyTerminal(811, bank=ARTIFICIAL),
                                            samples=6, seed_base=43_000_000)
    concepts = VisualConcepts.discover(language_one)
    learner_one = VersionSpaceLanguage(alphabet_one, concepts)
    train_one = _train_language(learner_one, language_one, 480, 44_000_000)
    before = _evaluate(learner_one, language_one_test, 512, 45_000_000)
    grammar_one = learner_one.grammar(_instruction_tokens(alphabet_one, language_one().reset(45_900_000)))
    learner_one.save(brain / 'language_one.json', grammar_one)
    hash_before = hashlib.sha256((brain / 'language_one.json').read_bytes()).hexdigest()

    language_two = lambda: LanguageNurseryWorld(9207, bank=GLYPHS, held_out=False)
    language_two_test = lambda: LanguageNurseryWorld(9207, bank=GLYPHS, held_out=True, steps=3)
    alphabet_two = VisualAlphabet.discover(lambda: LiteracyTerminal(1777, bank=GLYPHS),
                                            samples=6, seed_base=46_000_000)
    scratch = VersionSpaceLanguage(alphabet_two, concepts)
    scratch_result = _train_language(scratch, language_two, 360, 47_000_000)
    transfer = VersionSpaceLanguage(alphabet_two, concepts)
    candidates_before_schema = transfer.active_count
    positions = _position_sets(language_two, alphabet_two)
    candidates_after_schema = transfer.restrict_schema(positions)
    transfer_result = _train_language(transfer, language_two, 360, 47_000_000)
    language_two_eval = _evaluate(transfer, language_two_test, 512, 48_000_000)
    grammar_two = transfer.grammar(_instruction_tokens(alphabet_two, language_two().reset(48_900_000)))
    transfer.save(brain / 'language_two.json', grammar_two)

    after = _evaluate(learner_one, language_one_test, 512, 49_000_000)
    hash_after = hashlib.sha256((brain / 'language_one.json').read_bytes()).hexdigest()

    selector_path = workspace / 'living' / 'strategy_selector.json'
    selector = LearnedStrategySelector.load(selector_path)
    samples = np.stack([probe_features(language_one_test, 50_000_000 + index) for index in range(64)])
    selector.centroids['language-nursery-memory'] = samples.mean(0)
    selector.scale = np.maximum(selector.scale, samples.std(0)); selector.save(selector_path)
    selector_accuracy = float(np.mean([selector.rank(language_one_test, 50_100_000 + index,
        list(selector.centroids))[0][0] == 'language-nursery-memory' for index in range(32)]))

    from .autonomy import AutonomousCompetenceLoop
    loop = AutonomousCompetenceLoop(workspace, device, nursery_workspace=brain)
    decision = loop.select(language_one_test, probe_seed=50_300_000)
    competence = loop.evaluate_selected(language_one_test, decision['method'],
                                        seed_base=50_400_000, episodes=128)

    report = {'format': 'wailah-language-nursery-v2',
        'learner_inputs': 'pixels, self-chosen motor actions, terminal reward, done',
        'privileged_dictionary': False, 'correct_actions_supplied': False,
        'intermediate_reward': False, 'engineered_inductive_biases': [
            'eight-word vocabulary', 'three latent role families', 'one word per role per clause',
            'persistent modular memory'],
        'visual_concept_discovery': {'observations': 96, 'human_labels': 0,
            'color_clusters': 3, 'shape_clusters': 3},
        'language_one': {'training': train_one, 'withheld_composition': before,
            'discovered_grammar_role_order': grammar_one, 'initial_dictionary_size': 40320},
        'language_two': {'new_glyph_style': True, 'new_keyboard': True, 'new_dictionary': True,
            'new_word_order': True, 'schema_candidates_before': candidates_before_schema,
            'schema_candidates_after': candidates_after_schema,
            'scratch_training': scratch_result, 'transferred_schema_training': transfer_result,
            'withheld_composition': language_two_eval, 'discovered_grammar_role_order': grammar_two},
        'retention': {'language_one_before_language_two': before,
            'language_one_after_language_two': after, 'memory_hash_unchanged': hash_before == hash_after},
        'autonomous_integration': {'selector_accuracy': selector_accuracy,
            'decision': decision, 'competence_evaluation': competence},
        'promoted': decision['method'] == 'language-nursery-memory' and
                    competence.get('success_rate', 0) >= .95}
    atomic_json(output, report); return report


def render_language_nursery_summary(report_path: Path, output: Path) -> Path:
    from PIL import Image, ImageDraw, ImageFont
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    try:
        title = ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf', 27)
        body = ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf', 15)
        bold = ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf', 16)
        small = ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf', 12)
    except OSError: title = body = bold = small = ImageFont.load_default()
    image = Image.new('RGB', (1120, 690), (7, 13, 25)); draw = ImageDraw.Draw(image)
    green, blue, white, muted, panel = (75, 225, 164), (72, 174, 245), (232, 239, 248), (139, 156, 180), (14, 25, 43)
    draw.text((34, 24), 'WAILAH  /  LANGUAGE NURSERY v2', font=title, fill=white)
    draw.text((35, 64), 'discover concepts  •  infer words  •  execute chains  •  describe events  •  retain languages', font=body, fill=muted)
    one = report['language_one']['withheld_composition']; two = report['language_two']['withheld_composition']
    cards = [('L1 READ + EXECUTE', one['read_execute_success_rate']), ('L1 WRITE', one['event_description_write_accuracy']),
             ('L2 READ + EXECUTE', two['read_execute_success_rate']), ('L2 WRITE', two['event_description_write_accuracy'])]
    for index, (label, value) in enumerate(cards):
        x = 34 + index * 270; draw.rounded_rectangle((x, 105, x + 246, 197), radius=12, fill=panel)
        draw.text((x + 16, 121), label, font=small, fill=muted); draw.text((x + 16, 151), f'{value:.1%}', font=title, fill=green)
    rows = [
        ('Reward signal', 'terminal only — no intermediate correctness signal'),
        ('Language 1 lexicon', f"40,320 initial hypotheses → {report['language_one']['training']['remaining_hypotheses']}"),
        ('Withheld three-command chains', f"{one['episodes']} trials  /  random {one['random_chain_success']:.2%}"),
        ('Language 2 structural transfer', f"{report['language_two']['schema_candidates_before']:,} → {report['language_two']['schema_candidates_after']:,} candidates before reward"),
        ('Language 1 after Language 2', f"{report['retention']['language_one_after_language_two']['read_execute_success_rate']:.1%}  /  memory hash unchanged"),
        ('Learned strategy selection', f"{report['autonomous_integration']['selector_accuracy']:.1%} held-out"),
        ('Autonomous competence gate', report['autonomous_integration']['decision']['method']),
    ]
    draw.text((35, 228), 'CAPABILITY', font=small, fill=muted); draw.text((560, 228), 'MEASURED RESULT', font=small, fill=muted)
    for index, (name, value) in enumerate(rows):
        y = 260 + index * 52; draw.rounded_rectangle((28, y - 7, 1092, y + 34), radius=8, fill=panel)
        draw.text((43, y), name, font=body, fill=white); draw.text((560, y), value, font=body, fill=green if index else blue)
    draw.text((34, 642), 'Pixels + self-chosen actions + final reward. No dictionary, parser, object labels, or correct action sequences supplied.', font=small, fill=muted)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); image.save(output); return output
