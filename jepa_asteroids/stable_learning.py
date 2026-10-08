"""Stable, fully local pixel-to-action reinforcement learner.

The controller starts from random weights. Its convolutional eyes and dueling
action-value heads learn jointly from RGB game frames converted to small frame
stacks, actions, observed rewards and episode endings. The policy never sees
engine coordinates or other privileged state.

The design intentionally does not plan through the learned world model. The
earlier planner could exploit small prediction errors even while its supervised
loss improved. Here, every action value is grounded in real replay returns using
Double DQN, a lagged target network, multi-step targets and image augmentation.
"""
from __future__ import annotations

from collections import deque
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import Config
from .data import domain, progress
from .evaluation import publish
from .runtime import Guard, Stopped, atomic_json
from .self_learning import AsteroidsAdapter, _atomic_npz, frame_to_eye
from .training import atomic_torch


FORMAT = 'wailah-stable-pixel-control-v1'
REPLAY_FORMAT = 'wailah-stable-replay-v1'
STABLE_KEYS = (
    'self_observation_height', 'self_observation_width', 'self_frame_stack',
    'stable_latent_dim', 'stable_batch_size', 'stable_learning_rate',
    'stable_updates_per_episode', 'stable_warmup_transitions',
    'stable_replay_episodes', 'stable_recent_fraction', 'stable_discount',
    'stable_n_step', 'stable_target_interval', 'stable_epsilon_start',
    'stable_epsilon_end', 'stable_epsilon_decay_steps', 'stable_random_shift',
    'stable_grad_clip', 'stable_milestone_updates', 'stable_jepa_weight',
    'stable_jepa_tau', 'stable_outcome_fraction', 'action_dim')


def settings(cfg: Config) -> dict:
    return {key: getattr(cfg, key) for key in STABLE_KEYS}


def signature(cfg: Config) -> str:
    payload = json.dumps(settings(cfg), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


class StableReplay:
    """Bounded episode replay with recent data plus reservoir-sampled history."""
    def __init__(self, cfg: Config, workspace: Path, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.root = workspace / 'stable_replay'
        self.raw = self.root / 'episodes'
        self.path = self.root / 'manifest.json'
        expected = {'format': REPLAY_FORMAT, 'settings': settings(cfg), 'domain': domain(cfg)}
        if self.path.exists():
            self.manifest = json.loads(self.path.read_text(encoding='utf-8'))
            for key, value in expected.items():
                if self.manifest.get(key) != value:
                    raise RuntimeError(f'Stable replay {key} differs; use a new brain workspace.')
        else:
            self.manifest = {**expected, 'seen_episodes': 0, 'archive_candidates': 0,
                             'recent': [], 'archive': []}
            self._save()

    def _save(self) -> None:
        atomic_json(self.path, self.manifest)

    @property
    def entries(self) -> list[dict]:
        return self.manifest['recent'] + self.manifest['archive']

    def transition_count(self) -> int:
        return sum(entry['transitions'] for entry in self.entries)

    def _drop(self, entry: dict) -> None:
        path = self.raw / entry['file']
        if path.exists():
            path.unlink()

    def _rebalance(self) -> None:
        total = self.cfg.stable_replay_episodes
        recent_cap = max(1, min(total, int(math.ceil(total * self.cfg.stable_recent_fraction))))
        archive_cap = total - recent_cap
        while len(self.manifest['recent']) > recent_cap:
            candidate = self.manifest['recent'].pop(0)
            seen = int(self.manifest['archive_candidates']) + 1
            self.manifest['archive_candidates'] = seen
            archive = self.manifest['archive']
            if archive_cap == 0:
                self._drop(candidate)
            elif len(archive) < archive_cap:
                archive.append(candidate)
            else:
                position = int(self.rng.integers(seen))
                if position < archive_cap:
                    self._drop(archive[position])
                    archive[position] = candidate
                else:
                    self._drop(candidate)

    def add(self, frames: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            dones: np.ndarray, *, seed: int, hits: int, updates: int) -> dict:
        expected = (len(actions) + 1, self.cfg.self_observation_height,
                    self.cfg.self_observation_width)
        if frames.shape != expected or frames.dtype != np.uint8:
            raise ValueError(f'Eye frames must be uint8 {expected}.')
        if actions.shape != rewards.shape or actions.shape != dones.shape:
            raise ValueError('Transition arrays are misaligned.')
        number = int(self.manifest['seen_episodes'])
        identity = f'ep_{number:08d}'
        name = identity + '.npz'
        _atomic_npz(self.raw / name, frames=frames,
                    actions=actions.astype(np.uint8), rewards=rewards.astype(np.float32),
                    dones=dones.astype(np.bool_))
        entry = {'id': identity, 'file': name, 'transitions': len(actions),
                 'seed': int(seed), 'hits': int(hits), 'return': float(rewards.sum()),
                 'policy': 'epsilon-greedy dueling Double DQN',
                 'updates': int(updates)}
        self.manifest['seen_episodes'] = number + 1
        self.manifest['recent'].append(entry)
        self._rebalance()
        self._save()
        return entry

    def sample(self, batch: int):
        stack = self.cfg.self_frame_stack
        eligible = [entry for entry in self.entries if entry['transitions'] >= stack]
        if not eligible:
            raise RuntimeError('Replay has no episode long enough for a stacked transition.')
        recent_ids = {entry['id'] for entry in self.manifest['recent']}
        recent = [entry for entry in eligible if entry['id'] in recent_ids]
        archive = [entry for entry in eligible if entry['id'] not in recent_ids]
        obs, actions, returns, nxt, one_nxt, dones, discounts = [], [], [], [], [], [], []
        gamma = self.cfg.stable_discount
        for _ in range(batch):
            use_recent = recent and (not archive or self.rng.random() < self.cfg.stable_recent_fraction)
            pool = recent if use_recent else archive
            entry = pool[int(self.rng.integers(len(pool)))]
            with np.load(self.raw / entry['file'], allow_pickle=False) as data:
                first = stack - 1
                outcomes = np.flatnonzero(np.abs(data['rewards'][first:]) > .1) + first
                if len(outcomes) and self.rng.random() < self.cfg.stable_outcome_fraction:
                    index = int(outcomes[int(self.rng.integers(len(outcomes)))])
                else:
                    index = int(self.rng.integers(first, len(data['actions'])))
                total = 0.0
                terminal = False
                steps = 0
                for offset in range(self.cfg.stable_n_step):
                    at = index + offset
                    if at >= len(data['actions']):
                        break
                    total += (gamma ** offset) * float(data['rewards'][at]) / 5.0
                    steps += 1
                    if bool(data['dones'][at]):
                        terminal = True
                        break
                obs.append(data['frames'][index - stack + 1:index + 1].copy())
                one_nxt.append(data['frames'][index - stack + 2:index + 2].copy())
                nxt.append(data['frames'][index - stack + 1 + steps:index + 1 + steps].copy())
                actions.append(int(data['actions'][index]))
                returns.append(total)
                dones.append(terminal)
                discounts.append(gamma ** steps)
        return (torch.from_numpy(np.stack(obs)), torch.tensor(actions),
                torch.tensor(returns, dtype=torch.float32), torch.from_numpy(np.stack(nxt)),
                torch.from_numpy(np.stack(one_nxt)),
                torch.tensor(dones, dtype=torch.float32),
                torch.tensor(discounts, dtype=torch.float32))


class DuelingQNetwork(nn.Module):
    """Compact trainable visual encoder with separate value and advantage heads."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.eyes = nn.Sequential(
            nn.Conv2d(cfg.self_frame_stack, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(64, 96, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(96, 128, 3, 2, 1), nn.ReLU())
        height = cfg.self_observation_height // 16
        width = cfg.self_observation_width // 16
        self.trunk = nn.Sequential(nn.Flatten(),
            nn.Linear(128 * height * width, cfg.stable_latent_dim), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(cfg.stable_latent_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(nn.Linear(cfg.stable_latent_dim, 128), nn.ReLU(),
                                       nn.Linear(128, cfg.action_dim))
        self.apply(self._init)
        nn.init.orthogonal_(self.value[-1].weight, gain=.01)
        nn.init.orthogonal_(self.advantage[-1].weight, gain=.01)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.trunk(self.eyes(frames.float() / 255.0))

    def q_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        value = self.value(latent)
        advantage = self.advantage(latent)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.q_from_latent(self.encode(frames))


def random_shift(frames: torch.Tensor, padding: int) -> torch.Tensor:
    """Independent replicate-padded random crops, as a lightweight DrQ augmentation."""
    if padding == 0:
        return frames
    padded = F.pad(frames, (padding, padding, padding, padding), mode='replicate')
    height, width = frames.shape[-2:]
    limit = 2 * padding + 1
    crops = []
    for sample in padded:
        top = int(torch.randint(limit, (), device=frames.device))
        left = int(torch.randint(limit, (), device=frames.device))
        crops.append(sample[:, top:top + height, left:left + width])
    return torch.stack(crops)


class StableLearner:
    def __init__(self, cfg: Config, workspace: Path, device: torch.device, *, fresh: bool = False):
        self.cfg = cfg
        self.workspace = workspace
        self.device = device
        self.path = workspace / 'stable_brain.pt'
        if fresh and self.path.exists():
            raise FileExistsError('Fresh start refuses to overwrite a saved stable brain.')
        torch.manual_seed(cfg.seed + 13)
        self.rng = np.random.default_rng(cfg.seed + 817_201)
        self.online = DuelingQNetwork(cfg).to(device)
        self.target = copy.deepcopy(self.online).to(device).eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.representation_target = copy.deepcopy(
            nn.Sequential(self.online.eyes, self.online.trunk)).to(device).eval()
        for parameter in self.representation_target.parameters():
            parameter.requires_grad_(False)
        latent = cfg.stable_latent_dim
        self.jepa_predictor = nn.Sequential(
            nn.Linear(latent + cfg.action_dim, latent * 2), nn.ReLU(),
            nn.Linear(latent * 2, latent)).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.online.parameters()) + list(self.jepa_predictor.parameters()),
            lr=cfg.stable_learning_rate, eps=1e-5)
        self.state = {'episodes': 0, 'environment_steps': 0, 'updates': 0,
                      'created_unix': time.time(), 'target_syncs': 0,
                      'action_counts': [0] * cfg.action_dim}
        if self.path.exists() and not fresh:
            self._load()

    def _load(self) -> None:
        value = torch.load(self.path, map_location='cpu', weights_only=True)
        if value.get('format') != FORMAT or value.get('signature') != signature(self.cfg):
            raise RuntimeError('Saved stable brain uses different architecture/settings.')
        self.online.load_state_dict(value['online'])
        self.target.load_state_dict(value['target'])
        self.representation_target.load_state_dict(value['representation_target'])
        self.jepa_predictor.load_state_dict(value['jepa_predictor'])
        self.optimizer.load_state_dict(value['optimizer'])
        self.state = value['state']
        self.rng.bit_generator.state = value['numpy_rng']
        torch.set_rng_state(value['torch_rng'])
        if self.device.type == 'cuda' and value.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(value['cuda_rng'])

    def epsilon(self) -> float:
        ratio = min(1.0, self.state['environment_steps'] / self.cfg.stable_epsilon_decay_steps)
        return self.cfg.stable_epsilon_start + ratio * (
            self.cfg.stable_epsilon_end - self.cfg.stable_epsilon_start)

    def save(self, *, milestone: bool = False) -> None:
        value = {'format': FORMAT, 'signature': signature(self.cfg), 'settings': settings(self.cfg),
                 'online': self.online.state_dict(), 'target': self.target.state_dict(),
                 'representation_target': self.representation_target.state_dict(),
                 'jepa_predictor': self.jepa_predictor.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'state': dict(self.state),
                 'numpy_rng': self.rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None}
        atomic_torch(self.path, value)
        atomic_json(self.workspace / 'stable_brain.json', {
            'format': FORMAT, 'signature': signature(self.cfg), **self.state,
            'epsilon': self.epsilon(),
            'parameter_count': sum(p.numel() for p in self.online.parameters()) +
                               sum(p.numel() for p in self.jepa_predictor.parameters())})
        if milestone:
            atomic_torch(self.workspace / 'stable_milestones' /
                         f"update_{self.state['updates']:08d}.pt", value)

    @torch.no_grad()
    def q_values(self, stack: np.ndarray) -> torch.Tensor:
        self.online.eval()
        obs = torch.from_numpy(stack[None]).to(self.device)
        return self.online(obs)[0]

    @torch.no_grad()
    def act(self, stack: np.ndarray) -> tuple[int, list[float]]:
        values = self.q_values(stack)
        return int(values.argmax()), [float(value) for value in values]

    def train(self, replay: StableReplay, guard: Guard, updates: int) -> list[dict]:
        if replay.transition_count() < self.cfg.stable_warmup_transitions:
            return []
        records = []
        self.online.train()
        for _ in range(updates):
            guard.check(force=True)
            tick = time.monotonic()
            obs, actions, returns, nxt, one_nxt, dones, discounts = replay.sample(self.cfg.stable_batch_size)
            obs, actions, returns = obs.to(self.device), actions.to(self.device), returns.to(self.device)
            nxt, dones, discounts = nxt.to(self.device), dones.to(self.device), discounts.to(self.device)
            one_nxt = one_nxt.to(self.device)
            obs = random_shift(obs, self.cfg.stable_random_shift)
            nxt = random_shift(nxt, self.cfg.stable_random_shift)
            one_nxt = random_shift(one_nxt, self.cfg.stable_random_shift)
            with torch.no_grad():
                # Double DQN: online network selects, lagged target network evaluates.
                next_actions = self.online(nxt).argmax(dim=1, keepdim=True)
                next_values = self.target(nxt).gather(1, next_actions).squeeze(1)
                targets = returns + discounts * (1.0 - dones) * next_values
            latent = self.online.encode(obs)
            predictions = self.online.q_from_latent(latent).gather(1, actions[:, None]).squeeze(1)
            td = targets - predictions
            q_loss = F.smooth_l1_loss(predictions, targets)
            with torch.no_grad():
                target_latent = self.representation_target(one_nxt.float() / 255.0)
            conditioned = torch.cat((latent, F.one_hot(actions, self.cfg.action_dim).float()), dim=1)
            predicted_latent = self.jepa_predictor(conditioned)
            jepa_loss = 2.0 - 2.0 * (F.normalize(predicted_latent, dim=1) *
                                     F.normalize(target_latent, dim=1)).sum(dim=1).mean()
            loss = q_loss + self.cfg.stable_jepa_weight * jepa_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite stable-control loss.')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(),
                                                   self.cfg.stable_grad_clip,
                                                   error_if_nonfinite=True)
            self.optimizer.step()
            with torch.no_grad():
                source = list(self.online.eyes.parameters()) + list(self.online.trunk.parameters())
                for target_parameter, source_parameter in zip(
                        self.representation_target.parameters(), source, strict=True):
                    target_parameter.mul_(self.cfg.stable_jepa_tau).add_(
                        source_parameter, alpha=1.0 - self.cfg.stable_jepa_tau)
            self.state['updates'] += 1
            synced = self.state['updates'] % self.cfg.stable_target_interval == 0
            if synced:
                self.target.load_state_dict(self.online.state_dict())
                self.state['target_syncs'] += 1
            record = {'update': self.state['updates'], 'loss': float(loss.detach()),
                      'q_loss': float(q_loss.detach()), 'jepa_loss': float(jepa_loss.detach()),
                      'mean_abs_td': float(td.abs().mean()),
                      'mean_q': float(predictions.detach().mean()),
                      'mean_target': float(targets.mean()), 'gradient_norm': float(norm),
                      'target_synced': synced, 'seconds': time.monotonic() - tick}
            with (self.workspace / 'stable_training.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            records.append(record)
        return records


def run_episode(cfg: Config, workspace: Path, learner: StableLearner, replay: StableReplay,
                guard: Guard, *, seed: int, learning: bool, replay_delay: float = 0.0) -> dict:
    env = AsteroidsAdapter(cfg)
    frame = env.reset(seed)
    eye = frame_to_eye(frame, cfg)
    stack = deque([eye.copy() for _ in range(cfg.self_frame_stack)], maxlen=cfg.self_frame_stack)
    eyes, actions, rewards, dones = [eye], [], [], []
    greedy = 0
    hits = 0
    started = time.monotonic()
    for decision in range(cfg.episode_steps):
        enough = replay.transition_count() >= cfg.stable_warmup_transitions
        exploring = learning and (not enough or learner.rng.random() < learner.epsilon())
        if exploring:
            action = int(learner.rng.integers(cfg.action_dim))
            q_values = [0.0] * cfg.action_dim
        else:
            action, q_values = learner.act(np.stack(stack))
            greedy += 1
        frame, reward, done, info = env.step(action)
        eye = frame_to_eye(frame, cfg)
        stack.append(eye)
        eyes.append(eye)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        hits = info['total_hits']
        learner.state['action_counts'][action] += int(learning)
        if learning:
            learner.state['environment_steps'] += 1
        publish(workspace, frame, eye)
        progress(workspace, 'stable-learning' if learning else 'stable-watch',
                 episode=learner.state['episodes'] + 1, decision=decision + 1, score=hits,
                 lifetime_experience=learner.state['environment_steps'],
                 training_updates=learner.state['updates'], epsilon=learner.epsilon(),
                 activity='exploring' if exploring else 'acting from learned real returns',
                 q_values=q_values, simulated_seconds=info['simulated_seconds'],
                 wall_seconds=time.monotonic() - started,
                 message='Learning trainable visual control locally from pixels and real outcomes.')
        if replay_delay:
            time.sleep(replay_delay)
        if done:
            break
    result = {'seed': seed, 'steps': len(actions), 'hits': hits,
              'return': float(sum(rewards)), 'greedy_decisions': greedy,
              'learning': learning, 'terminated': bool(info['terminated']),
              'truncated': bool(info['truncated']), 'epsilon': learner.epsilon(),
              'actions': [int(value) for value in np.bincount(actions, minlength=cfg.action_dim)]}
    if learning:
        replay.add(np.stack(eyes), np.asarray(actions), np.asarray(rewards), np.asarray(dones),
                   seed=seed, hits=hits, updates=learner.state['updates'])
        learner.state['episodes'] += 1
        completed = learner.train(replay, guard, cfg.stable_updates_per_episode)
        result['updates_completed'] = len(completed)
        crossed = completed and (learner.state['updates'] % cfg.stable_milestone_updates < len(completed))
        learner.save(milestone=bool(crossed))
    with (workspace / 'stable_episodes.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(result) + '\n')
    return result


def run_session(cfg: Config, workspace: Path, device: torch.device, guard: Guard, *,
                episodes: int, fresh: bool = False, learning: bool = True,
                replay_delay: float = 0.0) -> dict:
    if episodes < 1:
        raise ValueError('episodes must be positive.')
    if fresh and ((workspace / 'stable_brain.pt').exists() or (workspace / 'stable_replay').exists()):
        raise FileExistsError('Fresh start never erases another learner; choose an empty workspace.')
    learner = StableLearner(cfg, workspace, device, fresh=fresh)
    replay = StableReplay(cfg, workspace, learner.rng)
    if not learning and not learner.path.exists():
        raise RuntimeError('Watch requires a saved stable brain.')
    if fresh:
        learner.save(milestone=True)
    results = []
    try:
        for _ in range(episodes):
            seed = cfg.seed + 4_100_000_000 + learner.state['episodes']
            results.append(run_episode(cfg, workspace, learner, replay, guard, seed=seed,
                                       learning=learning, replay_delay=replay_delay))
    except (Stopped, KeyboardInterrupt):
        if learning:
            learner.save()
        raise
    summary = {'learning': learning, 'episodes': len(results),
               'brain_episodes': learner.state['episodes'],
               'environment_steps': learner.state['environment_steps'],
               'training_updates': learner.state['updates'], 'results': results}
    atomic_json(workspace / 'stable_session.json', summary)
    progress(workspace, 'paused' if learning else 'watch-complete', metrics=summary,
             message='Stable learner saved.' if learning else
                     'Frozen stable-policy watch finished without training or replay writes.')
    return summary


def train_replay(cfg: Config, workspace: Path, device: torch.device, guard: Guard, *,
                 updates: int) -> dict:
    """Learn from already collected real experience without adding imagined data."""
    if updates < 1:
        raise ValueError('updates must be positive.')
    learner = StableLearner(cfg, workspace, device)
    if not learner.path.exists():
        raise RuntimeError('Replay training requires a saved stable brain.')
    replay = StableReplay(cfg, workspace, learner.rng)
    if replay.transition_count() < cfg.stable_warmup_transitions:
        raise RuntimeError('Not enough real experience for replay training.')
    start = learner.state['updates']
    completed = 0
    try:
        while completed < updates:
            count = min(50, updates - completed)
            learner.train(replay, guard, count)
            completed += count
            learner.save(milestone=learner.state['updates'] % cfg.stable_milestone_updates < count)
            progress(workspace, 'stable-replay-training', completed=completed, requested=updates,
                     training_updates=learner.state['updates'],
                     lifetime_experience=learner.state['environment_steps'],
                     message='Training visual control from stored real experience; no imagined transitions.')
    except (Stopped, KeyboardInterrupt):
        learner.save()
        raise
    result = {'updates_completed': completed, 'updates_before': start,
              'updates_after': learner.state['updates'],
              'replay_transitions': replay.transition_count()}
    atomic_json(workspace / 'stable_replay_training_session.json', result)
    return result


def _evaluate_episode(cfg: Config, learner: StableLearner, guard: Guard, *,
                      seed: int, policy: str) -> dict:
    env = AsteroidsAdapter(cfg)
    frame = env.reset(seed)
    eye = frame_to_eye(frame, cfg)
    stack = deque([eye.copy() for _ in range(cfg.self_frame_stack)], maxlen=cfg.self_frame_stack)
    rng = np.random.default_rng(seed + 882_001)
    trace, latencies = [], []
    hits, total = 0, 0.0
    permutation = np.asarray([2, 4, 0, 1, 3])
    for _ in range(cfg.episode_steps):
        guard.check()
        tick = time.monotonic()
        if policy == 'random':
            action = int(rng.integers(cfg.action_dim))
        elif policy == 'constant_fire':
            action = 4
        else:
            action, _ = learner.act(np.stack(stack))
            if policy == 'corrupted_action_mapping':
                action = int(permutation[action])
        latencies.append(time.monotonic() - tick)
        trace.append(action)
        frame, reward, done, info = env.step(action)
        stack.append(frame_to_eye(frame, cfg))
        hits = info['total_hits']
        total += reward
        if done:
            break
    return {'seed': seed, 'policy': policy, 'steps': len(trace), 'hits': hits,
            'return': float(total), 'terminated': bool(info['terminated']),
            'truncated': bool(info['truncated']),
            'mean_decision_seconds': float(np.mean(latencies)), 'actions': trace}


def evaluate(cfg: Config, workspace: Path, device: torch.device, guard: Guard, *,
             episodes: int = 12, seed_offset: int = 0, label: str = 'evaluation') -> dict:
    """Matched, frozen evaluation. Its frames never enter replay or training."""
    if episodes < 1:
        raise ValueError('episodes must be positive.')
    trained = StableLearner(cfg, workspace, device)
    if not trained.path.exists():
        raise RuntimeError('A saved stable brain is required.')
    untrained = StableLearner(cfg, workspace / 'stable_untrained_fixture', device, fresh=True)
    seeds = [cfg.seed + 4_600_000_000 + seed_offset + index for index in range(episodes)]
    policies = ('random', 'constant_fire', 'untrained_model', 'trained_model',
                'corrupted_action_mapping')
    results = []
    for policy in policies:
        agent = untrained if policy == 'untrained_model' else trained
        effective = 'trained_model' if policy == 'trained_model' else policy
        for seed in seeds:
            guard.check(force=True)
            results.append(_evaluate_episode(cfg, agent, guard, seed=seed, policy=effective))
            progress(workspace, 'stable-evaluation', policy=policy, trial=len(results),
                     total_trials=len(policies) * episodes,
                     message='Frozen matched-seed evaluation; evaluation data is excluded from learning.')
    summaries = {}
    for policy in policies:
        name = 'trained_model' if policy == 'trained_model' else policy
        rows = [row for row in results if row['policy'] == name]
        summaries[policy] = {
            'episodes': len(rows), 'mean_hits': float(np.mean([row['hits'] for row in rows])),
            'mean_return': float(np.mean([row['return'] for row in rows])),
            'termination_rate': float(np.mean([row['terminated'] for row in rows])),
            'mean_steps': float(np.mean([row['steps'] for row in rows])),
            'mean_decision_seconds': float(np.mean([row['mean_decision_seconds'] for row in rows]))}
    report = {'protocol': 'matched frozen episodes; evaluation excluded from replay and updates',
              'label': label, 'checkpoint_updates': trained.state['updates'],
              'checkpoint_experience': trained.state['environment_steps'],
              'seed_offset': seed_offset, 'seeds': seeds, 'summaries': summaries,
              'results': results}
    destination = workspace / 'stable_evaluations' / f'update_{trained.state["updates"]:08d}_{label}.json'
    atomic_json(destination, report)
    atomic_json(workspace / 'stable_evaluation.json', report)
    progress(workspace, 'stable-evaluated', metrics=summaries,
             message='Frozen matched evaluation complete; control metrics, not loss, decide success.')
    return report
