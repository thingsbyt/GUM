"""Outcome-learned action prior stabilized by a JEPA-trained visual policy."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import time

import numpy as np
import torch

from .config import Config
from .data import progress
from .evaluation import publish
from .runtime import Guard, atomic_json
from .self_learning import AsteroidsAdapter, frame_to_eye
from .stable_learning import StableLearner, StableReplay


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.exp(logits - logits.max())
    return values / values.sum()


def _rollout_prior(cfg: Config, probabilities: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed + 31_337)
    env = AsteroidsAdapter(cfg)
    env.reset(seed)
    total, hits = 0.0, 0
    for step in range(cfg.episode_steps):
        action = int(rng.choice(cfg.action_dim, p=probabilities))
        _, reward, done, info = env.step(action)
        total += reward
        hits = info['total_hits']
        if done:
            break
    return {'hits': hits, 'return': float(total), 'terminated': bool(info['terminated']),
            'steps': step + 1}


def learn_prior(cfg: Config, workspace: Path, guard: Guard, *, generations: int = 12,
                population: int = 20, episodes: int = 6, checkpoint: str = '') -> dict:
    """Cross-entropy policy search using only sampled actions and episode outcomes."""
    if generations < 1 or population < 4 or episodes < 2:
        raise ValueError('Prior search requires generations>=1, population>=4 and episodes>=2.')
    rng = np.random.default_rng(cfg.seed + 991_771)
    mean = np.zeros(cfg.action_dim, dtype=np.float64)
    std = np.full(cfg.action_dim, 1.5, dtype=np.float64)
    elite_count = max(2, population // 4)
    history = []
    best = None
    for generation in range(generations):
        guard.check(force=True)
        logits = rng.normal(mean, std, size=(population, cfg.action_dim))
        seeds = [cfg.seed + 4_800_000_000 + generation * 10_000 + index
                 for index in range(episodes)]
        scored = []
        for candidate in logits:
            probabilities = _softmax(candidate)
            rows = [_rollout_prior(cfg, probabilities, seed) for seed in seeds]
            score = float(np.mean([row['return'] for row in rows]))
            scored.append((score, candidate, probabilities,
                           float(np.mean([row['hits'] for row in rows]))))
        scored.sort(key=lambda item: item[0], reverse=True)
        elite = scored[:elite_count]
        elite_logits = np.stack([item[1] for item in elite])
        mean = elite_logits.mean(0)
        mean -= mean.mean()
        std = np.clip(elite_logits.std(0) + .05, .12, 2.0)
        if best is None or scored[0][0] > best[0]:
            best = scored[0]
        record = {'generation': generation + 1, 'mean_probabilities': _softmax(mean).tolist(),
                  'best_mean_return': scored[0][0], 'best_mean_hits': scored[0][3],
                  'elite_mean_return': float(np.mean([item[0] for item in elite]))}
        history.append(record)
        progress(workspace, 'mastery-prior-search', **record,
                 message='Learning a robust action prior from episode outcomes only.')
    probabilities = _softmax(mean)
    milestone = checkpoint or 'stable_milestones/update_00004000.pt'
    policy = {'format': 'wailah-mastery-policy-v1', 'probabilities': probabilities.tolist(),
              'visual_weight': .1, 'temperature': .15, 'checkpoint': milestone,
              'training_signal': 'episode returns only; no engine state or action labels',
              'generations': generations, 'population': population,
              'episodes_per_candidate': episodes}
    report = {**policy, 'history': history, 'best_sampled_probabilities': best[2].tolist(),
              'best_sampled_mean_return': best[0], 'best_sampled_mean_hits': best[3]}
    atomic_json(workspace / 'mastery_policy.json', policy)
    atomic_json(workspace / 'mastery_prior_search.json', report)
    return report


class MasteryController:
    def __init__(self, cfg: Config, workspace: Path, device: torch.device):
        self.cfg = cfg
        self.workspace = workspace
        self.learner = StableLearner(cfg, workspace, device)
        self.policy = json.loads((workspace / 'mastery_policy.json').read_text(encoding='utf-8'))
        checkpoint = workspace / self.policy['checkpoint']
        value = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.learner.online.load_state_dict(value['online'])
        self.prior = np.asarray(self.policy['probabilities'], dtype=np.float64)

    def probabilities(self, stack: np.ndarray) -> np.ndarray:
        q = self.learner.q_values(stack).cpu().numpy().astype(np.float64)
        visual = _softmax(q / float(self.policy['temperature']))
        weight = float(self.policy['visual_weight'])
        combined = (1.0 - weight) * self.prior + weight * visual
        return combined / combined.sum()

    def act(self, stack: np.ndarray, rng: np.random.Generator, *, corrupt: bool = False) -> int:
        action = int(rng.choice(self.cfg.action_dim, p=self.probabilities(stack)))
        return int(np.asarray([2, 4, 0, 1, 3])[action]) if corrupt else action


def _visual_action(learner: StableLearner, stack: np.ndarray,
                   rng: np.random.Generator, *, mode: str,
                   temperature: float) -> tuple[int, list[float]]:
    q = learner.q_values(stack).cpu().numpy().astype(np.float64)
    if mode == 'greedy':
        return int(q.argmax()), q.tolist()
    if mode != 'softmax' or temperature <= 0:
        raise ValueError('Visual policy must be greedy or positive-temperature softmax.')
    return int(rng.choice(len(q), p=_softmax(q / temperature))), q.tolist()


def _curriculum_episode(cfg: Config, workspace: Path, learner: StableLearner,
                        replay: StableReplay, guard: Guard, *, seed: int,
                        prior: np.ndarray, prior_weight: float) -> dict:
    env = AsteroidsAdapter(cfg)
    frame = env.reset(seed)
    eye = frame_to_eye(frame, cfg)
    stack = deque([eye.copy() for _ in range(cfg.self_frame_stack)], maxlen=cfg.self_frame_stack)
    eyes, actions, rewards, dones = [eye], [], [], []
    sources = {'prior': 0, 'visual': 0, 'uniform': 0}
    hits = 0
    last_publish = 0.0
    for decision in range(cfg.episode_steps):
        visual_action, q_values = learner.act(np.stack(stack))
        draw = learner.rng.random()
        if draw < prior_weight:
            action = int(learner.rng.choice(cfg.action_dim, p=prior));source = 'prior'
        elif learner.rng.random() < learner.epsilon():
            action = int(learner.rng.integers(cfg.action_dim));source = 'uniform'
        else:
            action = visual_action;source = 'visual'
        sources[source] += 1
        frame, reward, done, info = env.step(action)
        eye = frame_to_eye(frame, cfg);stack.append(eye);eyes.append(eye)
        actions.append(action);rewards.append(reward);dones.append(done)
        hits = info['total_hits']
        learner.state['action_counts'][action] += 1
        learner.state['environment_steps'] += 1
        now = time.monotonic()
        if done or now - last_publish >= .20:
            publish(workspace, frame, eye)
            progress(workspace, 'vision-curriculum', episode=learner.state['episodes'] + 1,
                     decision=decision + 1, score=hits, prior_weight=prior_weight,
                     action=action, control_source=source, q_values=q_values,
                     lifetime_experience=learner.state['environment_steps'],
                     training_updates=learner.state['updates'], epsilon=learner.epsilon(),
                     message='Transferring an outcome-learned action rhythm into visual control.')
            last_publish = now
        if done:
            break
    replay.add(np.stack(eyes), np.asarray(actions), np.asarray(rewards), np.asarray(dones),
               seed=seed, hits=hits, updates=learner.state['updates'])
    learner.state['episodes'] += 1
    completed = learner.train(replay, guard, cfg.stable_updates_per_episode)
    crossed = completed and learner.state['updates'] % cfg.stable_milestone_updates < len(completed)
    learner.save(milestone=bool(crossed))
    result = {'seed': seed, 'steps': len(actions), 'hits': hits,
              'return': float(sum(rewards)), 'terminated': bool(info['terminated']),
              'prior_weight': prior_weight, 'sources': sources,
              'updates_completed': len(completed),
              'actions': [int(value) for value in np.bincount(actions, minlength=cfg.action_dim)]}
    with (workspace / 'vision_curriculum_episodes.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(result) + '\n')
    return result


def train_vision_curriculum(cfg: Config, workspace: Path, device: torch.device,
                            guard: Guard, *, episodes: int = 192,
                            start_prior_weight: float = .9,
                            end_prior_weight: float = 0.0) -> dict:
    """Withdraw an outcome-trained behavior prior while training pixel-conditioned Q values."""
    if episodes < 2 or not 0 <= end_prior_weight <= start_prior_weight <= 1:
        raise ValueError('Curriculum needs >=2 episodes and 0 <= end <= start <= 1.')
    prior_policy = json.loads((workspace / 'mastery_policy.json').read_text(encoding='utf-8'))
    prior = np.asarray(prior_policy['probabilities'], dtype=np.float64)
    learner = StableLearner(cfg, workspace, device)
    if not learner.path.exists():
        raise RuntimeError('Vision curriculum requires a saved stable brain.')
    replay = StableReplay(cfg, workspace, learner.rng)
    learner.save(milestone=True)
    before = {'episodes': learner.state['episodes'], 'environment_steps': learner.state['environment_steps'],
              'updates': learner.state['updates']}
    results = []
    try:
        for index in range(episodes):
            fraction = index / (episodes - 1)
            weight = start_prior_weight + fraction * (end_prior_weight - start_prior_weight)
            seed = cfg.seed + 5_100_000_000 + before['episodes'] + index
            results.append(_curriculum_episode(cfg, workspace, learner, replay, guard,
                                               seed=seed, prior=prior, prior_weight=weight))
    finally:
        learner.save(milestone=True)
    after = {'episodes': learner.state['episodes'], 'environment_steps': learner.state['environment_steps'],
             'updates': learner.state['updates']}
    report = {'format': 'gum-vision-curriculum-v1', 'episodes_requested': episodes,
              'episodes_completed': len(results), 'start_prior_weight': start_prior_weight,
              'end_prior_weight': end_prior_weight, 'before': before, 'after': after,
              'mean_hits': float(np.mean([row['hits'] for row in results])) if results else 0.0,
              'mean_return': float(np.mean([row['return'] for row in results])) if results else 0.0,
              'final_checkpoint': f"stable_milestones/update_{learner.state['updates']:08d}.pt",
              'results': results}
    atomic_json(workspace / 'vision_curriculum.json', report)
    return report


def _vision_episode(cfg: Config, learner: StableLearner, prior: np.ndarray,
                    guard: Guard, *, seed: int, policy: str,
                    mode: str = 'greedy', temperature: float = .15) -> dict:
    env = AsteroidsAdapter(cfg)
    frame = env.reset(seed)
    eye = frame_to_eye(frame, cfg)
    stack = deque([eye.copy() for _ in range(cfg.self_frame_stack)], maxlen=cfg.self_frame_stack)
    action_rng = np.random.default_rng(seed + 93_101)
    observation_rng = np.random.default_rng(seed + 93_103)
    permutation = np.asarray([2, 4, 0, 1, 3])
    total, hits, actions, latencies = 0.0, 0, [], []
    for _ in range(cfg.episode_steps):
        guard.check()
        tick = time.monotonic()
        if policy == 'random':
            action = int(action_rng.integers(cfg.action_dim))
        elif policy == 'constant_fire':
            action = 4
        elif policy == 'learned_prior':
            action = int(action_rng.choice(cfg.action_dim, p=prior))
        else:
            observed = np.stack(stack)
            if policy == 'occluded_visual':
                observed = np.zeros_like(observed)
            elif policy == 'shuffled_frame_order':
                observed = observed[observation_rng.permutation(len(observed))]
            effective_mode = 'greedy' if policy == 'visual_greedy' else mode
            action, _ = _visual_action(learner, observed, action_rng,
                                       mode=effective_mode, temperature=temperature)
            if policy == 'corrupted_visual_mapping':
                action = int(permutation[action])
        latencies.append(time.monotonic() - tick);actions.append(action)
        frame, reward, done, info = env.step(action)
        stack.append(frame_to_eye(frame, cfg));total += reward;hits = info['total_hits']
        if done:
            break
    return {'seed': seed, 'policy': policy, 'hits': hits, 'return': float(total),
            'terminated': bool(info['terminated']), 'steps': len(actions), 'actions': actions,
            'mean_decision_seconds': float(np.mean(latencies))}


def calibrate_visual_policy(cfg: Config, workspace: Path, device: torch.device,
                            guard: Guard, *, episodes: int = 24) -> dict:
    """Select a visual-only action rule on a disclosed development partition."""
    if episodes < 2:
        raise ValueError('Visual calibration requires at least two episodes.')
    learner = StableLearner(cfg, workspace, device)
    prior_policy = json.loads((workspace / 'mastery_policy.json').read_text(encoding='utf-8'))
    prior = np.asarray(prior_policy['probabilities'], dtype=np.float64)
    learner.save(milestone=True)
    candidates = [('greedy', .15)] + [('softmax', value) for value in (.05, .1, .15, .25, .5, 1.0)]
    seeds = [cfg.seed + 5_200_000_000 + index for index in range(episodes)]
    rows = []
    for mode, temperature in candidates:
        results = [_vision_episode(cfg, learner, prior, guard, seed=seed,
                                   policy='visual_selected', mode=mode,
                                   temperature=temperature) for seed in seeds]
        row = {'mode': mode, 'temperature': temperature,
               'mean_hits': float(np.mean([item['hits'] for item in results])),
               'mean_return': float(np.mean([item['return'] for item in results])),
               'termination_rate': float(np.mean([item['terminated'] for item in results]))}
        rows.append(row)
        progress(workspace, 'vision-calibration', candidate=len(rows), total_candidates=len(candidates),
                 **row, message='Selecting a visual-only action rule on development seeds.')
    best = max(rows, key=lambda row: (row['mean_return'], row['mean_hits'], -row['termination_rate']))
    policy = {'format': 'gum-visual-policy-v1', 'checkpoint':
              f"stable_milestones/update_{learner.state['updates']:08d}.pt",
              'mode': best['mode'], 'temperature': best['temperature'], 'prior_weight': 0.0,
              'calibration_seed_partition': seeds, 'calibration_episodes': episodes,
              'calibration_result': best, 'active': False}
    atomic_json(workspace / 'vision_policy.json', policy)
    report = {'protocol': 'development calibration only; no replay writes or weight updates',
              'candidates': rows, 'selected': policy}
    atomic_json(workspace / 'vision_calibration.json', report)
    return report


def _paired_interval(results: list[dict], candidate: str, baseline: str) -> dict:
    left = {row['seed']:row['hits'] for row in results if row['policy'] == candidate}
    right = {row['seed']:row['hits'] for row in results if row['policy'] == baseline}
    seeds = sorted(set(left) & set(right))
    differences = np.asarray([left[seed] - right[seed] for seed in seeds], dtype=np.float64)
    rng = np.random.default_rng(4_771_991)
    samples = differences[rng.integers(len(differences), size=(5000, len(differences)))].mean(1)
    return {'candidate': candidate, 'baseline': baseline,
            'mean_hit_difference': float(differences.mean()),
            'bootstrap_95_interval': [float(value) for value in np.percentile(samples, [2.5, 97.5])]}


def evaluate_visual_policy(cfg: Config, workspace: Path, device: torch.device,
                           guard: Guard, *, episodes: int = 64,
                           seed_offset: int = 400_000,
                           label: str = 'untouched') -> dict:
    """Frozen matched audit of the selected visual-only policy and visual ablations."""
    if episodes < 2:
        raise ValueError('Visual evaluation requires at least two episodes.')
    policy_path = workspace / 'vision_policy.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    learner = StableLearner(cfg, workspace, device)
    checkpoint = torch.load(workspace / policy['checkpoint'], map_location='cpu', weights_only=True)
    learner.online.load_state_dict(checkpoint['online'])
    prior = np.asarray(json.loads((workspace / 'mastery_policy.json').read_text(
        encoding='utf-8'))['probabilities'], dtype=np.float64)
    policies = ('random', 'constant_fire', 'learned_prior', 'visual_selected',
                'visual_greedy', 'occluded_visual', 'shuffled_frame_order',
                'corrupted_visual_mapping')
    seeds = [cfg.seed + 5_300_000_000 + seed_offset + index for index in range(episodes)]
    results = []
    for name in policies:
        for seed in seeds:
            results.append(_vision_episode(cfg, learner, prior, guard, seed=seed, policy=name,
                                           mode=policy['mode'], temperature=policy['temperature']))
        progress(workspace, 'vision-evaluation', policy=name, completed_policies=policies.index(name) + 1,
                 total_policies=len(policies), message='Running frozen matched visual-policy tests.')
    summaries = {}
    for name in policies:
        rows = [row for row in results if row['policy'] == name]
        summaries[name] = {'episodes': len(rows),
            'mean_hits': float(np.mean([row['hits'] for row in rows])),
            'mean_return': float(np.mean([row['return'] for row in rows])),
            'termination_rate': float(np.mean([row['terminated'] for row in rows])),
            'mean_steps': float(np.mean([row['steps'] for row in rows]))}
    comparisons = {baseline:_paired_interval(results, 'visual_selected', baseline)
                   for baseline in ('learned_prior', 'occluded_visual',
                                    'shuffled_frame_order', 'corrupted_visual_mapping')}
    gates = {
        'beats_learned_prior': comparisons['learned_prior']['bootstrap_95_interval'][0] > 0,
        'uses_pixels': comparisons['occluded_visual']['bootstrap_95_interval'][0] > 0,
        'control_mapping_matters': comparisons['corrupted_visual_mapping']['bootstrap_95_interval'][0] > 0}
    gates['promoted'] = all(gates.values())
    policy['active'] = gates['promoted'];policy['evaluation_label'] = label
    atomic_json(policy_path, policy)
    report = {'format': 'gum-visual-policy-evaluation-v1',
              'protocol': 'frozen matched seeds; no replay writes or weight updates',
              'label': label, 'seed_offset': seed_offset, 'seeds': seeds,
              'policy': policy, 'summaries': summaries, 'comparisons': comparisons,
              'gates': gates, 'results': results}
    atomic_json(workspace / 'vision_evaluations' / f'{label}.json', report)
    atomic_json(workspace / 'vision_evaluation.json', report)
    return report


def _episode(cfg: Config, controller: MasteryController, *, seed: int, policy: str) -> dict:
    env = AsteroidsAdapter(cfg)
    frame = env.reset(seed)
    eye = frame_to_eye(frame, cfg)
    stack = deque([eye.copy() for _ in range(cfg.self_frame_stack)], maxlen=cfg.self_frame_stack)
    rng = np.random.default_rng(seed + 81_991)
    total, hits, actions, latencies = 0.0, 0, [], []
    for _ in range(cfg.episode_steps):
        tick = time.monotonic()
        if policy == 'random':
            action = int(rng.integers(cfg.action_dim))
        elif policy == 'constant_fire':
            action = 4
        elif policy == 'learned_prior':
            action = int(rng.choice(cfg.action_dim, p=controller.prior))
        else:
            action = controller.act(np.stack(stack), rng,
                                    corrupt=policy == 'corrupted_mastery_mapping')
        latencies.append(time.monotonic() - tick)
        actions.append(action)
        frame, reward, done, info = env.step(action)
        stack.append(frame_to_eye(frame, cfg))
        total += reward
        hits = info['total_hits']
        if done:
            break
    return {'seed': seed, 'policy': policy, 'hits': hits, 'return': float(total),
            'terminated': bool(info['terminated']), 'steps': len(actions), 'actions': actions,
            'mean_decision_seconds': float(np.mean(latencies))}


def evaluate_mastery(cfg: Config, workspace: Path, device: torch.device, guard: Guard, *,
                     episodes: int = 64, seed_offset: int = 100_000,
                     label: str = 'untouched') -> dict:
    controller = MasteryController(cfg, workspace, device)
    policies = ('random', 'constant_fire', 'learned_prior', 'visual_mastery',
                'corrupted_mastery_mapping')
    seeds = [cfg.seed + 4_900_000_000 + seed_offset + index for index in range(episodes)]
    results = []
    for policy in policies:
        for seed in seeds:
            guard.check(force=True)
            results.append(_episode(cfg, controller, seed=seed, policy=policy))
    summaries = {}
    for policy in policies:
        rows = [row for row in results if row['policy'] == policy]
        summaries[policy] = {'episodes': len(rows),
            'mean_hits': float(np.mean([row['hits'] for row in rows])),
            'mean_return': float(np.mean([row['return'] for row in rows])),
            'termination_rate': float(np.mean([row['terminated'] for row in rows])),
            'mean_steps': float(np.mean([row['steps'] for row in rows])),
            'mean_decision_seconds': float(np.mean([row['mean_decision_seconds'] for row in rows]))}
    report = {'protocol': 'frozen matched seeds; no replay or updates', 'label': label,
              'seed_offset': seed_offset, 'seeds': seeds, 'policy': controller.policy,
              'summaries': summaries, 'results': results}
    atomic_json(workspace / 'mastery_evaluations' / f'{label}.json', report)
    atomic_json(workspace / 'mastery_evaluation.json', report)
    return report


def watch_mastery(cfg: Config, workspace: Path, device: torch.device, guard: Guard, *,
                  episodes: int = 1, replay_delay: float = .03) -> dict:
    """Play the frozen mastered policy without changing brain or replay."""
    controller = MasteryController(cfg, workspace, device)
    results = []
    for episode in range(episodes):
        seed = cfg.seed + 5_000_000_000 + episode
        env = AsteroidsAdapter(cfg)
        frame = env.reset(seed)
        eye = frame_to_eye(frame, cfg)
        stack = deque([eye.copy() for _ in range(cfg.self_frame_stack)], maxlen=cfg.self_frame_stack)
        rng = np.random.default_rng(seed + 81_991)
        total, hits, actions = 0.0, 0, []
        for decision in range(cfg.episode_steps):
            guard.check()
            action = controller.act(np.stack(stack), rng)
            frame, reward, done, info = env.step(action)
            eye = frame_to_eye(frame, cfg)
            stack.append(eye)
            total += reward
            hits = info['total_hits']
            actions.append(action)
            publish(workspace, frame, eye)
            progress(workspace, 'mastery-watch', episode=episode + 1, decision=decision + 1,
                     score=hits, action=action, control_source='learned prior + visual policy',
                     activity='frozen outcome-trained policy',
                     message='Watching the frozen learned policy; weights and replay are unchanged.')
            if replay_delay:
                time.sleep(replay_delay)
            if done:
                break
        results.append({'seed': seed, 'hits': hits, 'return': float(total),
                        'terminated': bool(info['terminated']), 'steps': len(actions),
                        'actions': actions})
    summary = {'learning': False, 'mastery': True, 'episodes': episodes, 'results': results}
    atomic_json(workspace / 'mastery_watch.json', summary)
    return summary
