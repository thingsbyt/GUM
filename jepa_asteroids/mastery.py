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
from .stable_learning import StableLearner


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
                     score=hits, activity='frozen mastered visual policy',
                     message='Watching without replay writes or learning.')
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
