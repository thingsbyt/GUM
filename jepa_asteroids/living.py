"""Task-free routing, shared predictive memory and sparse skill composition.

This layer grows above the isolated lifelong skill bank.  Existing skill files
remain immutable.  Their self-generated identities supervise a visual router;
the router receives pixels, not external task labels.  A shared slow world model
learns balanced dynamics from every replay, while a sparse composer combines
compatible policies.  High-value action sequences are mined as reusable options.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .lifelong import LifelongSkillBank, Skill, TaskSpec
from .runtime import atomic_json
from .training import atomic_torch


FORMAT = 'wailah-living-core-v1'


class RouterNet(nn.Module):
    def __init__(self, channels: int, experts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(48 * 4 * 4, 96), nn.ReLU(), nn.Linear(96, experts))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.float() / 255.0)


class SharedWorldModel(nn.Module):
    """Fast encoder/predictor with a slow EMA target shared across all skills."""
    def __init__(self, channels: int, max_actions: int, latent: int = 96):
        super().__init__(); self.max_actions = max_actions
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 24, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(48, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent), nn.LayerNorm(latent), nn.Tanh())
        import copy
        self.slow_encoder = copy.deepcopy(self.encoder)
        for parameter in self.slow_encoder.parameters(): parameter.requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(latent + max_actions, latent * 2), nn.ReLU(),
                                       nn.Linear(latent * 2, latent))
        self.reward = nn.Sequential(nn.Linear(latent + max_actions, 64), nn.ReLU(), nn.Linear(64, 1))
        self.done = nn.Sequential(nn.Linear(latent + max_actions, 64), nn.ReLU(), nn.Linear(64, 1))

    def loss(self, obs, actions, rewards, nxt, dones):
        latent = self.encoder(obs.float() / 255.0)
        one_hot = F.one_hot(actions, self.max_actions).float()
        conditioned = torch.cat((latent, one_hot), dim=1)
        with torch.no_grad(): target = self.slow_encoder(nxt.float() / 255.0)
        predicted = self.predictor(conditioned)
        dynamics = 2 - 2 * (F.normalize(predicted, dim=1) * F.normalize(target, dim=1)).sum(1).mean()
        reward = F.smooth_l1_loss(self.reward(conditioned).squeeze(1), rewards)
        done = F.binary_cross_entropy_with_logits(self.done(conditioned).squeeze(1), dones)
        return dynamics + .5 * reward + .1 * done, dynamics, reward, done

    @torch.no_grad()
    def transition_error(self, obs, actions, rewards, nxt, dones):
        """Per-transition surprise used for autonomous novelty detection."""
        latent = self.encoder(obs.float() / 255.0)
        one_hot = F.one_hot(actions, self.max_actions).float()
        conditioned = torch.cat((latent, one_hot), dim=1)
        target = self.slow_encoder(nxt.float() / 255.0)
        predicted = self.predictor(conditioned)
        dynamics = 2 - 2 * (F.normalize(predicted, dim=1) * F.normalize(target, dim=1)).sum(1)
        reward = F.smooth_l1_loss(self.reward(conditioned).squeeze(1), rewards, reduction='none')
        done = F.binary_cross_entropy_with_logits(
            self.done(conditioned).squeeze(1), dones, reduction='none')
        return dynamics + .5 * reward + .1 * done

    @torch.no_grad()
    def update_slow(self, tau: float = .995):
        for slow, fast in zip(self.slow_encoder.parameters(), self.encoder.parameters(), strict=True):
            slow.mul_(tau).add_(fast, alpha=1 - tau)


@dataclass
class ExpertData:
    task_id: str
    spec: TaskSpec
    replay: dict[str, np.ndarray]
    skill: Skill


class OptionLibrary:
    """Mines repeated high-return action chunks without action semantics."""
    def __init__(self): self.options: dict[str, list[dict]] = {}

    def discover(self, expert: ExpertData, *, length: int = 4, limit: int = 12) -> list[dict]:
        actions = expert.replay['actions']; rewards = expert.replay['rewards']
        scores = defaultdict(list)
        for start in range(max(0, len(actions) - length + 1)):
            sequence = tuple(int(x) for x in actions[start:start + length])
            scores[sequence].append(float(rewards[start:start + length].sum()))
        ranked = []
        for sequence, values in scores.items():
            if len(values) < 3: continue
            ranked.append({'actions': list(sequence), 'count': len(values),
                           'mean_reward': float(np.mean(values)),
                           'value': float(np.mean(values) + .02 * np.log1p(len(values)))})
        ranked.sort(key=lambda row: (row['value'], row['count']), reverse=True)
        self.options[expert.task_id] = ranked[:limit]
        return self.options[expert.task_id]


class RehearsalScheduler:
    """Selects memories using forgetting, uncertainty and time-since-rehearsal."""
    def __init__(self): self.state = {}
    def update(self, task_id: str, *, best: float, current: float,
               uncertainty: float, step: int) -> None:
        old = self.state.get(task_id, {'last_rehearsal': 0})
        self.state[task_id] = {'best': float(best), 'current': float(current),
                               'uncertainty': float(uncertainty), 'step': int(step),
                               'last_rehearsal': old['last_rehearsal']}
    def choose(self, step: int, count: int = 1) -> list[str]:
        scored = []
        for task_id, value in self.state.items():
            forgetting = max(0.0, value['best'] - value['current'])
            age = max(0, step - value['last_rehearsal']) / max(1, step)
            scored.append((2 * forgetting + value['uncertainty'] + .25 * age, task_id))
        result = [task_id for _, task_id in sorted(scored, reverse=True)[:count]]
        for task_id in result: self.state[task_id]['last_rehearsal'] = step
        return result


class LivingSystem:
    """Shared predictive memory plus task-free sparse composition over a skill bank."""
    def __init__(self, workspace: Path, device: torch.device | str = 'cpu'):
        self.workspace = Path(workspace); self.device = torch.device(device)
        self.root = self.workspace / 'living'; self.root.mkdir(parents=True, exist_ok=True)
        self.bank = LifelongSkillBank(self.workspace, self.device)
        self.experts = self._load_experts()
        if not self.experts: raise RuntimeError('LivingSystem needs at least one committed skill.')
        shapes = {x.spec.observation_shape for x in self.experts}
        if len(shapes) != 1: raise RuntimeError('Current living core requires a common observation shape.')
        self.ids = [x.task_id for x in self.experts]
        channels = self.experts[0].spec.observation_shape[0]
        self.max_actions = max(x.spec.action_dim for x in self.experts)
        self.router = RouterNet(channels, len(self.experts)).to(self.device)
        self.world = SharedWorldModel(channels, self.max_actions).to(self.device)
        self.router_path = self.root / 'router.pt'; self.world_path = self.root / 'world.pt'
        self.options = OptionLibrary(); self.rehearsal = RehearsalScheduler()
        self.novelty_baseline: dict | None = None
        if self.router_path.exists():
            saved = torch.load(self.router_path, map_location='cpu', weights_only=True)
            current = self.router.state_dict()
            # Preserve the learned visual trunk when the expert set grows; the
            # expanded output layer starts fresh and learns the new identity.
            current.update({key: value for key, value in saved.items()
                            if key in current and current[key].shape == value.shape})
            self.router.load_state_dict(current)
        if self.world_path.exists():
            saved = torch.load(self.world_path, map_location='cpu', weights_only=True)
            current = self.world.state_dict()
            current.update({key: value for key, value in saved.items()
                            if key in current and current[key].shape == value.shape})
            self.world.load_state_dict(current)

    def _load_experts(self) -> list[ExpertData]:
        result = []
        for task_id, record in sorted(self.bank.registry['tasks'].items()):
            path = self.bank.skills / task_id / 'skill.pt'; replay_path = self.bank.skills / task_id / 'replay.npz'
            if not path.exists() or not replay_path.exists(): continue
            spec_value = record['spec']; spec_value['observation_shape'] = tuple(spec_value['observation_shape'])
            spec = TaskSpec(**spec_value); skill = Skill(path.parent, spec, self.device)
            with np.load(replay_path, allow_pickle=False) as raw:
                replay = {key: raw[key].copy() for key in ('obs','actions','rewards','next_obs','dones')}
            result.append(ExpertData(task_id, spec, replay, skill))
        return result

    def fit_router(self, updates: int = 500, batch: int = 128) -> dict:
        optimizer = torch.optim.Adam(self.router.parameters(), lr=7e-4)
        rng = np.random.default_rng(818_771); history = []
        self.router.train()
        for _ in range(updates):
            per = max(1, batch // len(self.experts)); xs, ys = [], []
            for label, expert in enumerate(self.experts):
                index = rng.integers(len(expert.replay['obs']), size=per)
                xs.append(expert.replay['obs'][index]); ys.extend([label] * per)
            x = torch.from_numpy(np.concatenate(xs)).to(self.device)
            y = torch.tensor(ys, device=self.device)
            logits = self.router(x); loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            history.append(float(loss.detach()))
        atomic_torch(self.router_path, self.router.state_dict())
        accuracy = self.router_accuracy(samples_per_expert=512)
        return {'updates': updates, 'loss_first_50': float(np.mean(history[:50])),
                'loss_last_50': float(np.mean(history[-50:])), 'replay_accuracy': accuracy}

    @torch.no_grad()
    def router_accuracy(self, samples_per_expert: int = 256) -> float:
        self.router.eval(); rng = np.random.default_rng(919_881); correct = total = 0
        for label, expert in enumerate(self.experts):
            count = min(samples_per_expert, len(expert.replay['obs']))
            index = rng.choice(len(expert.replay['obs']), size=count, replace=False)
            pred = self.router(torch.from_numpy(expert.replay['obs'][index]).to(self.device)).argmax(1)
            correct += int((pred == label).sum()); total += count
        return correct / max(1, total)

    def train_world(self, updates: int = 800, batch: int = 128) -> dict:
        optimizer = torch.optim.Adam(self.world.parameters(), lr=4e-4)
        rng = np.random.default_rng(717_661); history = []
        self.world.train()
        for _ in range(updates):
            per = max(1, batch // len(self.experts)); chunks = defaultdict(list)
            for expert in self.experts:
                index = rng.integers(len(expert.replay['obs']), size=per)
                for key in ('obs','actions','rewards','next_obs','dones'):
                    chunks[key].append(expert.replay[key][index])
            obs = torch.from_numpy(np.concatenate(chunks['obs'])).to(self.device)
            action = torch.from_numpy(np.concatenate(chunks['actions'])).long().to(self.device)
            reward = torch.from_numpy(np.concatenate(chunks['rewards'])).float().to(self.device)
            nxt = torch.from_numpy(np.concatenate(chunks['next_obs'])).to(self.device)
            done = torch.from_numpy(np.concatenate(chunks['dones']).astype(np.float32)).to(self.device)
            loss, dynamics, reward_loss, done_loss = self.world.loss(obs, action, reward, nxt, done)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world.parameters(), 10.0); optimizer.step()
            self.world.update_slow(); history.append((float(loss.detach()), float(dynamics.detach()),
                                                       float(reward_loss.detach()), float(done_loss.detach())))
        atomic_torch(self.world_path, self.world.state_dict())
        return {'updates': updates, 'loss_first_50': float(np.mean([x[0] for x in history[:50]])),
                'loss_last_50': float(np.mean([x[0] for x in history[-50:]])),
                'dynamics_last_50': float(np.mean([x[1] for x in history[-50:]])),
                'reward_last_50': float(np.mean([x[2] for x in history[-50:]]))}

    @torch.no_grad()
    def routing(self, observation: np.ndarray, action_dim: int, top_k: int = 2) -> list[tuple[str,float]]:
        value = torch.from_numpy(np.asarray(observation)[None]).to(self.device)
        logits = self.router(value)[0]
        mask = torch.tensor([x.spec.action_dim == action_dim for x in self.experts], device=self.device)
        logits = logits.masked_fill(~mask, -torch.inf); probabilities = logits.softmax(0)
        count = min(top_k, int(mask.sum()))
        weights, indices = probabilities.topk(count)
        return [(self.ids[int(i)], float(w)) for i, w in zip(indices, weights)]

    @torch.no_grad()
    def act(self, observation: np.ndarray, action_dim: int, top_k: int = 2) -> tuple[int,dict]:
        routes = self.routing(observation, action_dim, top_k=top_k)
        combined = torch.zeros(action_dim, device=self.device)
        for task_id, weight in routes:
            expert = next(x for x in self.experts if x.task_id == task_id)
            obs = torch.from_numpy(np.asarray(observation)[None]).to(self.device)
            q = expert.skill.online(obs)[0]
            combined += weight * (q - q.mean()) / (q.std() + 1e-6)
        action = int(combined.argmax())
        return action, {'routes': [{'expert': x, 'weight': w} for x, w in routes],
                        'q': [float(x) for x in combined]}

    def discover_options(self) -> dict[str,list[dict]]:
        return {x.task_id: self.options.discover(x) for x in self.experts}

    @torch.no_grad()
    def calibrate_continual_signals(self, samples_per_expert: int = 256) -> dict:
        """Learn normal router/world surprise from the agent's own stored experience."""
        self.router.eval(); self.world.eval(); rng = np.random.default_rng(303_117)
        surprises, uncertainties, by_expert = [], [], {}
        for expert in self.experts:
            count = min(samples_per_expert, len(expert.replay['obs']))
            index = rng.choice(len(expert.replay['obs']), size=count, replace=False)
            obs = torch.from_numpy(expert.replay['obs'][index]).to(self.device)
            actions = torch.from_numpy(expert.replay['actions'][index]).long().to(self.device)
            rewards = torch.from_numpy(expert.replay['rewards'][index]).float().to(self.device)
            nxt = torch.from_numpy(expert.replay['next_obs'][index]).to(self.device)
            dones = torch.from_numpy(expert.replay['dones'][index].astype(np.float32)).to(self.device)
            surprise = self.world.transition_error(obs, actions, rewards, nxt, dones).cpu().numpy()
            uncertainty = (1 - self.router(obs).softmax(1).max(1).values).cpu().numpy()
            surprises.extend(float(x) for x in surprise)
            uncertainties.extend(float(x) for x in uncertainty)
            by_expert[expert.task_id] = {
                'world_surprise_mean': float(np.mean(surprise)),
                'router_uncertainty_mean': float(np.mean(uncertainty))}
        self.novelty_baseline = {
            'world_surprise_mean': float(np.mean(surprises)),
            'world_surprise_std': float(np.std(surprises) + 1e-6),
            'world_surprise_p99': float(np.quantile(surprises, .99)),
            'router_uncertainty_p99': float(np.quantile(uncertainties, .99)),
            'by_expert': by_expert}
        return self.novelty_baseline

    @torch.no_grad()
    def assess_experience(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                          nxt: np.ndarray, dones: np.ndarray, action_dim: int) -> dict:
        """Decide whether an unlabeled experience stream belongs to known skills."""
        if self.novelty_baseline is None:
            self.calibrate_continual_signals()
        obs_t = torch.from_numpy(np.asarray(obs)).to(self.device)
        action_t = torch.from_numpy(np.asarray(actions)).long().to(self.device)
        reward_t = torch.from_numpy(np.asarray(rewards)).float().to(self.device)
        next_t = torch.from_numpy(np.asarray(nxt)).to(self.device)
        done_t = torch.from_numpy(np.asarray(dones).astype(np.float32)).to(self.device)
        errors = self.world.transition_error(obs_t, action_t, reward_t, next_t, done_t)
        logits = self.router(obs_t)
        compatible = torch.tensor([x.spec.action_dim == action_dim for x in self.experts],
                                  device=self.device)
        if not bool(compatible.any()):
            digest = hashlib.sha256(np.asarray(obs[:min(len(obs), 32)]).tobytes()).hexdigest()[:10]
            return {'novel': True, 'world_surprise': float(errors.mean()),
                    'world_threshold': float('nan'), 'router_uncertainty': 1.0,
                    'uncertainty_threshold': 0.0, 'suggested_expert_id': f'auto-{digest}',
                    'reason': 'no action-compatible expert'}
        probabilities = logits.masked_fill(~compatible, -torch.inf).softmax(1)
        confidence = probabilities.max(1).values
        mean_error = float(errors.mean()); uncertainty = float((1 - confidence).mean())
        baseline = self.novelty_baseline
        error_threshold = max(baseline['world_surprise_p99'],
                              baseline['world_surprise_mean'] + 3 * baseline['world_surprise_std'])
        uncertainty_threshold = max(.20, 2 * baseline['router_uncertainty_p99'])
        novel = mean_error > error_threshold or uncertainty > uncertainty_threshold
        digest = hashlib.sha256(np.asarray(obs[:min(len(obs), 32)]).tobytes()).hexdigest()[:10]
        return {'novel': bool(novel), 'world_surprise': mean_error,
                'world_threshold': float(error_threshold), 'router_uncertainty': uncertainty,
                'uncertainty_threshold': float(uncertainty_threshold),
                'suggested_expert_id': f'auto-{digest}' if novel else None}

    def grow_if_novel(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                      nxt: np.ndarray, dones: np.ndarray, action_dim: int,
                      *, updates: int = 64) -> dict:
        """Create and bootstrap a protected branch when experience is unfamiliar."""
        assessment = self.assess_experience(obs, actions, rewards, nxt, dones, action_dim)
        if not assessment['novel']:
            return {**assessment, 'grown': False}
        task_id = assessment['suggested_expert_id']
        template = self.experts[0].spec
        spec = TaskSpec(task_id, tuple(np.asarray(obs).shape[1:]), action_dim,
                        replay_capacity=max(2_000, min(50_000, len(obs) * 8)),
                        batch_size=min(64, max(4, len(obs))), latent_dim=template.latent_dim,
                        learning_rate=template.learning_rate, discount=template.discount,
                        target_interval=template.target_interval, seed=template.seed + len(self.experts) + 1)
        skill = self.bank.open(spec)
        for index in range(len(obs)):
            skill.observe(obs[index], int(actions[index]), float(rewards[index]),
                          nxt[index], bool(dones[index]))
        losses = skill.learn(updates); skill.save()
        return {**assessment, 'grown': True, 'task_id': task_id,
                'transitions_absorbed': len(obs), 'optimizer_updates': len(losses)}

    def rehearsal_plan(self, step: int = 1_000) -> dict:
        """Prioritize protected memories without letting them share policy parameters."""
        if self.novelty_baseline is None:
            self.calibrate_continual_signals()
        for expert in self.experts:
            record = self.bank.registry['tasks'][expert.task_id]
            score = float(record.get('last_score') or 0.0)
            best = float(record.get('best_score') or score)
            uncertainty = self.novelty_baseline['by_expert'][expert.task_id]['router_uncertainty_mean']
            self.rehearsal.update(expert.task_id, best=best, current=score,
                                  uncertainty=uncertainty, step=step)
        return {'selected_next': self.rehearsal.choose(step, count=min(2, len(self.experts))),
                'priorities': self.rehearsal.state}

    def save_manifest(self, router: dict, world: dict, options: dict,
                      novelty: dict, rehearsal: dict) -> dict:
        manifest = {'format': FORMAT, 'created_unix': time.time(), 'experts': self.ids,
                    'parameter_count': sum(p.numel() for p in self.router.parameters()) +
                                       sum(p.numel() for p in self.world.parameters()),
                    'router': router, 'world': world, 'options': options,
                    'novelty_baseline': novelty, 'rehearsal': rehearsal,
                    'source_skill_hashes': {x.task_id: hashlib.sha256(x.skill.path.read_bytes()).hexdigest()
                                            for x in self.experts}}
        atomic_json(self.root / 'manifest.json', manifest); return manifest

    def build(self, router_updates: int = 500, world_updates: int = 800) -> dict:
        before = {x.task_id: hashlib.sha256(x.skill.path.read_bytes()).hexdigest() for x in self.experts}
        router = self.fit_router(router_updates); world = self.train_world(world_updates)
        options = self.discover_options(); novelty = self.calibrate_continual_signals()
        rehearsal = self.rehearsal_plan()
        manifest = self.save_manifest(router, world, options, novelty, rehearsal)
        after = {x.task_id: hashlib.sha256(x.skill.path.read_bytes()).hexdigest() for x in self.experts}
        manifest['skill_checkpoints_unchanged'] = before == after
        atomic_json(self.root / 'manifest.json', manifest); return manifest
