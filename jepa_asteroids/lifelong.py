"""A growable, task-agnostic skill bank for continual reinforcement learning.

The bank uses parameter isolation for retention: every task owns a policy,
target network, optimizer and bounded replay memory.  A compatible older skill
may initialize a new skill's visual features, but committed old skills are
never edited while another task learns.  Revisiting a task resumes its own
policy and replay.  A score gate commits only non-regressing candidates.

This is intentionally an interface, not an environment-specific trick.  A
task adapter supplies uint8 observations, discrete actions, rewards and done
flags.  No LLM, action names or simulator state are required.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .runtime import atomic_json, replace_file
from .training import atomic_torch


FORMAT = 'wailah-lifelong-skill-bank-v1'
SKILL_FORMAT = 'wailah-lifelong-discrete-skill-v1'
_TASK_ID = re.compile(r'^[a-z0-9][a-z0-9._-]{0,63}$')


@dataclass(frozen=True)
class TaskSpec:
    """Serializable contract between an environment adapter and one skill."""
    task_id: str
    observation_shape: tuple[int, int, int]
    action_dim: int
    replay_capacity: int = 20_000
    batch_size: int = 32
    latent_dim: int = 128
    learning_rate: float = 3e-4
    discount: float = .99
    target_interval: int = 250
    seed: int = 7

    def validate(self) -> 'TaskSpec':
        if not _TASK_ID.fullmatch(self.task_id):
            raise ValueError('task_id must be lowercase letters/numbers plus ._- and at most 64 characters.')
        if len(self.observation_shape) != 3 or min(self.observation_shape) < 1:
            raise ValueError('observation_shape must be positive channel-height-width.')
        if self.observation_shape[0] > 8:
            raise ValueError('Channel-first observations may have at most eight channels.')
        if self.action_dim < 2 or self.replay_capacity < 1 or self.batch_size < 2:
            raise ValueError('A task needs at least two actions and a positive replay/batch size.')
        if self.latent_dim < 16 or self.learning_rate <= 0 or not 0 <= self.discount <= 1:
            raise ValueError('Invalid latent size, learning rate or discount.')
        if self.target_interval < 1:
            raise ValueError('target_interval must be positive.')
        return self

    @property
    def compatibility(self) -> tuple[tuple[int, int, int], int, int]:
        return self.observation_shape, self.action_dim, self.latent_dim


def _spec_dict(spec: TaskSpec) -> dict:
    value = asdict(spec)
    value['observation_shape'] = list(spec.observation_shape)
    return value


def _atomic_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, **values)
    replace_file(temp, path)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SkillNetwork(nn.Module):
    """Small adaptive visual policy suitable for unrelated discrete tasks."""
    def __init__(self, spec: TaskSpec):
        super().__init__()
        channels = spec.observation_shape[0]
        self.eyes = nn.Sequential(
            nn.Conv2d(channels, 24, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(48, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)))
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, spec.latent_dim), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(spec.latent_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(spec.latent_dim, 64), nn.ReLU(),
                                       nn.Linear(64, spec.action_dim))
        self.apply(self._init)
        nn.init.orthogonal_(self.value[-1].weight, gain=.01)
        nn.init.orthogonal_(self.advantage[-1].weight, gain=.01)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(self.eyes(obs.float() / 255.0))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.features(obs)
        value = self.value(latent)
        advantage = self.advantage(latent)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class TaskReplay:
    """Bounded reservoir replay owned by exactly one task."""
    def __init__(self, path: Path, spec: TaskSpec, rng: np.random.Generator):
        self.path, self.spec, self.rng = path, spec, rng
        shape = (0, *spec.observation_shape)
        self.obs = np.empty(shape, dtype=np.uint8)
        self.next_obs = np.empty(shape, dtype=np.uint8)
        self.actions = np.empty((0,), dtype=np.int64)
        self.rewards = np.empty((0,), dtype=np.float32)
        self.dones = np.empty((0,), dtype=np.bool_)
        self.seen = 0
        if path.exists():
            with np.load(path, allow_pickle=False) as value:
                stored = tuple(int(x) for x in value['observation_shape'])
                if stored != spec.observation_shape or int(value['action_dim']) != spec.action_dim:
                    raise RuntimeError('Replay task contract differs from the requested skill.')
                self.obs = value['obs'].copy(); self.next_obs = value['next_obs'].copy()
                self.actions = value['actions'].copy(); self.rewards = value['rewards'].copy()
                self.dones = value['dones'].copy(); self.seen = int(value['seen'])

    def __len__(self) -> int:
        return len(self.actions)

    def add(self, obs: np.ndarray, action: int, reward: float,
            next_obs: np.ndarray, done: bool) -> None:
        obs = self._observation(obs); next_obs = self._observation(next_obs)
        if not 0 <= int(action) < self.spec.action_dim:
            raise ValueError('Action is outside this task action space.')
        self.seen += 1
        if len(self) < self.spec.replay_capacity:
            index = len(self)
            self.obs = np.concatenate((self.obs, obs[None]))
            self.next_obs = np.concatenate((self.next_obs, next_obs[None]))
            self.actions = np.append(self.actions, int(action))
            self.rewards = np.append(self.rewards, np.float32(reward))
            self.dones = np.append(self.dones, bool(done))
        else:
            index = int(self.rng.integers(self.seen))
            if index >= self.spec.replay_capacity:
                return
            self.obs[index] = obs; self.next_obs[index] = next_obs
            self.actions[index] = action; self.rewards[index] = reward; self.dones[index] = done

    def _observation(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        if value.shape != self.spec.observation_shape or value.dtype != np.uint8:
            raise ValueError(f'Observation must be uint8 {self.spec.observation_shape}.')
        return np.ascontiguousarray(value)

    def sample(self, batch: int):
        if len(self) < batch:
            raise RuntimeError('Not enough task replay to sample a batch.')
        index = self.rng.integers(len(self), size=batch)
        return (torch.from_numpy(self.obs[index]), torch.from_numpy(self.actions[index]),
                torch.from_numpy(self.rewards[index]), torch.from_numpy(self.next_obs[index]),
                torch.from_numpy(self.dones[index].astype(np.float32)))

    def save(self) -> None:
        _atomic_npz(self.path, format=np.array(SKILL_FORMAT),
                    observation_shape=np.asarray(self.spec.observation_shape, dtype=np.int64),
                    action_dim=np.int64(self.spec.action_dim), seen=np.int64(self.seen),
                    obs=self.obs, actions=self.actions, rewards=self.rewards,
                    next_obs=self.next_obs, dones=self.dones)


class Skill:
    """One plastic branch. Other committed branches are outside its optimizer."""
    def __init__(self, root: Path, spec: TaskSpec, device: torch.device,
                 *, parent_checkpoint: Path | None = None):
        self.root, self.spec, self.device = root, spec.validate(), device
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = root / 'skill.pt'
        torch.manual_seed(spec.seed + int(hashlib.sha256(spec.task_id.encode()).hexdigest()[:8], 16))
        self.rng = np.random.default_rng(spec.seed + 701)
        self.online = SkillNetwork(spec).to(device)
        self.target = SkillNetwork(spec).to(device).eval()
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters(): p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=spec.learning_rate, eps=1e-5)
        self.state = {'updates': 0, 'transitions': 0, 'encounters': 0,
                      'created_unix': time.time(), 'parent_task': None}
        self.replay = TaskReplay(root / 'replay.npz', spec, self.rng)
        if self.path.exists():
            self._load()
        elif parent_checkpoint is not None:
            self._transfer(parent_checkpoint)

    def _load(self) -> None:
        value = torch.load(self.path, map_location='cpu', weights_only=True)
        if value.get('format') != SKILL_FORMAT or value.get('spec') != _spec_dict(self.spec):
            raise RuntimeError('Saved skill contract differs from TaskSpec.')
        self.online.load_state_dict(value['online']); self.target.load_state_dict(value['target'])
        self.optimizer.load_state_dict(value['optimizer']); self.state = value['state']
        self.rng.bit_generator.state = value['numpy_rng']; torch.set_rng_state(value['torch_rng'])

    def _transfer(self, checkpoint: Path) -> None:
        value = torch.load(checkpoint, map_location='cpu', weights_only=True)
        parent_spec = value['spec']
        if tuple(parent_spec['observation_shape']) != self.spec.observation_shape:
            return
        incoming = value['online']
        current = self.online.state_dict()
        copied = {k: v for k, v in incoming.items()
                  if (k.startswith('eyes.') or k.startswith('trunk.'))
                  and k in current and current[k].shape == v.shape}
        current.update(copied); self.online.load_state_dict(current)
        self.target.load_state_dict(self.online.state_dict())
        self.state['parent_task'] = parent_spec['task_id']

    def _payload(self) -> dict:
        return {'format': SKILL_FORMAT, 'spec': _spec_dict(self.spec),
                'online': self.online.state_dict(), 'target': self.target.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'state': dict(self.state),
                'numpy_rng': self.rng.bit_generator.state, 'torch_rng': torch.get_rng_state()}

    def save(self) -> None:
        atomic_torch(self.path, self._payload()); self.replay.save()

    @torch.no_grad()
    def act(self, observation: np.ndarray, *, epsilon: float = 0.0) -> int:
        if not 0 <= epsilon <= 1: raise ValueError('epsilon must be between zero and one.')
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.spec.action_dim))
        value = self.replay._observation(observation)
        q = self.online(torch.from_numpy(value[None]).to(self.device))
        return int(q.argmax(1))

    def observe(self, observation: np.ndarray, action: int, reward: float,
                next_observation: np.ndarray, done: bool) -> None:
        self.replay.add(observation, action, reward, next_observation, done)
        self.state['transitions'] += 1

    def learn(self, updates: int) -> list[float]:
        losses = []
        if len(self.replay) < self.spec.batch_size:
            return losses
        self.online.train()
        for _ in range(updates):
            obs, action, reward, nxt, done = self.replay.sample(self.spec.batch_size)
            obs, action, reward = obs.to(self.device), action.to(self.device), reward.to(self.device)
            nxt, done = nxt.to(self.device), done.to(self.device)
            with torch.no_grad():
                next_action = self.online(nxt).argmax(1, keepdim=True)
                next_q = self.target(nxt).gather(1, next_action).squeeze(1)
                target = reward + self.spec.discount * (1.0 - done) * next_q
            prediction = self.online(obs).gather(1, action[:, None]).squeeze(1)
            loss = F.smooth_l1_loss(prediction, target)
            self.optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
            self.optimizer.step(); self.state['updates'] += 1
            if self.state['updates'] % self.spec.target_interval == 0:
                self.target.load_state_dict(self.online.state_dict())
            losses.append(float(loss.detach()))
        return losses

    def rollback(self) -> None:
        if self.path.exists(): self._load()


class LifelongSkillBank:
    """Persistent registry, transfer router and regression-gated consolidator."""
    def __init__(self, workspace: Path, device: torch.device | str = 'cpu'):
        self.root = Path(workspace) / 'lifelong'
        self.skills = self.root / 'skills'
        self.path = self.root / 'registry.json'
        self.device = torch.device(device)
        if self.path.exists():
            self.registry = json.loads(self.path.read_text(encoding='utf-8'))
            if self.registry.get('format') != FORMAT:
                raise RuntimeError('Unknown lifelong registry format.')
        else:
            self.registry = {'format': FORMAT, 'created_unix': time.time(),
                             'tasks': {}, 'events': []}
            self._save()

    def _save(self) -> None:
        atomic_json(self.path, self.registry)

    def _compatible_parent(self, spec: TaskSpec) -> Path | None:
        choices = []
        for task_id, record in self.registry['tasks'].items():
            other = record['spec']
            compatible = (tuple(other['observation_shape']) == spec.observation_shape and
                          int(other['latent_dim']) == spec.latent_dim)
            checkpoint = self.skills / task_id / 'skill.pt'
            if compatible and checkpoint.exists():
                choices.append((float(record.get('best_score', -1e30)), checkpoint))
        return max(choices, default=(0.0, None), key=lambda item: item[0])[1]

    def open(self, spec: TaskSpec) -> Skill:
        spec.validate(); tasks = self.registry['tasks']; now = time.time()
        if spec.task_id in tasks:
            if tasks[spec.task_id]['spec'] != _spec_dict(spec):
                raise RuntimeError('Existing task_id has a different TaskSpec.')
            parent = None
            tasks[spec.task_id]['revisits'] = int(tasks[spec.task_id].get('revisits', 0)) + 1
        else:
            parent = self._compatible_parent(spec)
            tasks[spec.task_id] = {'spec': _spec_dict(spec), 'created_unix': now,
                                   'revisits': 0, 'encounters': 0,
                                   'best_score': None, 'last_score': None,
                                   'parent_task': None, 'context_count': 0,
                                   'context_centroid': None}
        skill = Skill(self.skills / spec.task_id, spec, self.device,
                      parent_checkpoint=parent)
        tasks[spec.task_id]['encounters'] += 1
        tasks[spec.task_id]['parent_task'] = skill.state.get('parent_task')
        skill.state['encounters'] = tasks[spec.task_id]['encounters']
        self.registry['events'].append({'event': 'encounter', 'task_id': spec.task_id,
                                        'unix': now, 'revisit': tasks[spec.task_id]['revisits'] > 0})
        self._save()
        return skill

    def consolidate(self, skill: Skill, score: float, *, tolerance: float = 0.0) -> dict:
        """Commit an improving candidate or roll its weights back to the last good skill."""
        if tolerance < 0: raise ValueError('tolerance cannot be negative.')
        task_id = skill.spec.task_id; record = self.registry['tasks'][task_id]
        old_hashes = {p.parent.name: _hash(p) for p in self.skills.glob('*/skill.pt')
                      if p.parent.name != task_id}
        best = record['best_score']
        accepted = best is None or float(score) + tolerance >= float(best)
        improved = best is None or float(score) > float(best)
        if accepted:
            skill.save()
            record['last_score'] = float(score)
            if improved: record['best_score'] = float(score)
        else:
            skill.replay.save()
            skill.rollback()
        unchanged = all(_hash(self.skills / name / 'skill.pt') == digest
                        for name, digest in old_hashes.items())
        if not unchanged:
            raise RuntimeError('A sibling skill changed during consolidation.')
        event = {'event': 'consolidation', 'task_id': task_id, 'unix': time.time(),
                 'score': float(score), 'previous_best': best, 'accepted': accepted,
                 'improved': improved, 'other_skills_unchanged': unchanged,
                 'updates': int(skill.state['updates']),
                 'transitions': int(skill.state['transitions'])}
        self.registry['events'].append(event); self._save()
        return event

    def record_context(self, task_id: str, observation: np.ndarray) -> None:
        record = self.registry['tasks'][task_id]
        feature = self._context_feature(observation)
        count = int(record.get('context_count', 0))
        if count:
            centroid = np.asarray(record['context_centroid'], dtype=np.float64)
            feature = centroid + (feature - centroid) / (count + 1)
        record['context_count'] = count + 1
        record['context_centroid'] = feature.tolist(); self._save()

    def route(self, observation: np.ndarray, action_dim: int, *, max_distance: float = .18) -> str | None:
        """Recognize visually distinct contexts; ambiguous reward tasks require explicit IDs."""
        feature = self._context_feature(observation); choices = []
        for task_id, record in self.registry['tasks'].items():
            if int(record['spec']['action_dim']) != int(action_dim) or record['context_centroid'] is None:
                continue
            centroid = np.asarray(record['context_centroid'], dtype=np.float64)
            distance = float(np.linalg.norm(feature - centroid) / math.sqrt(len(feature)))
            choices.append((distance, task_id))
        if not choices: return None
        distance, task_id = min(choices)
        return task_id if distance <= max_distance else None

    @staticmethod
    def _context_feature(observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation)
        if value.dtype != np.uint8 or value.ndim != 3:
            raise ValueError('Context routing requires a channel-first uint8 image.')
        tensor = torch.from_numpy(value[None]).float() / 255.0
        pooled = F.adaptive_avg_pool2d(tensor, (8, 8))[0].flatten().numpy()
        return pooled.astype(np.float64)

    def status(self) -> dict:
        return json.loads(json.dumps(self.registry))

    def quarantine(self, task_id: str, reason: str) -> dict:
        """Recoverably remove a failed branch from active routing and rehearsal."""
        if task_id not in self.registry['tasks']:
            raise KeyError(task_id)
        stamp = time.strftime('%Y%m%d-%H%M%S', time.gmtime())
        source = self.skills / task_id
        destination = self.root / 'quarantine' / f'{task_id}-{stamp}'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.exists(): source.replace(destination)
        record = self.registry['tasks'].pop(task_id)
        event = {'event': 'quarantine', 'task_id': task_id, 'unix': time.time(),
                 'reason': str(reason), 'recoverable_path': str(destination),
                 'record': record}
        self.registry['events'].append(event); self._save(); return event


def run_toy_continual_proof(workspace: Path, device: torch.device | str = 'cpu') -> dict:
    """Deterministic A -> B -> A visual bandit used as a software proof."""
    bank = LifelongSkillBank(workspace, device)

    def images() -> list[np.ndarray]:
        result = []
        for index in range(9):
            image = np.zeros((1, 16, 16), dtype=np.uint8)
            x = 1 + (index % 3) * 5; y = 1 + (index // 3) * 5
            image[:, y:y + 4, x:x + 4] = 255
            image[:, :, (index * 2) % 16] = 80
            result.append(image)
        return result

    observations = images()

    def collect(skill: Skill, offset: int, repeats: int) -> None:
        for _ in range(repeats):
            for state, obs in enumerate(observations):
                for action in range(3):
                    reward = 1.0 if action == (state + offset) % 3 else 0.0
                    skill.observe(obs, action, reward, obs, True)

    def score(skill: Skill) -> float:
        correct = sum(skill.act(obs) == state % 3 for state, obs in enumerate(observations))
        return correct / len(observations)

    spec_a = TaskSpec('visual-bandit-a', (1, 16, 16), 3, replay_capacity=1024,
                      batch_size=27, latent_dim=64, learning_rate=1e-3,
                      discount=0.0, target_interval=20, seed=11)
    spec_b = TaskSpec('visual-bandit-b', (1, 16, 16), 3, replay_capacity=1024,
                      batch_size=27, latent_dim=64, learning_rate=1e-3,
                      discount=0.0, target_interval=20, seed=17)
    # Deliberately stop A early so the later revisit can demonstrate plasticity,
    # rather than saturating the toy task during its first encounter.
    a = bank.open(spec_a); collect(a, 0, 1); a.learn(6)
    a_first = score(a); event_a1 = bank.consolidate(a, a_first)
    a_hash = _hash(a.path)

    b = bank.open(spec_b); collect(b, 1, 4); b.learn(80)
    # B is scored against its own reversed mapping.
    b_score = sum(b.act(obs) == (state + 1) % 3
                  for state, obs in enumerate(observations)) / len(observations)
    event_b = bank.consolidate(b, b_score)
    a_after_b = score(bank.open(spec_a)); a_unchanged = _hash(a.path) == a_hash

    a_again = bank.open(spec_a); collect(a_again, 0, 5); a_again.learn(120)
    a_revisit = score(a_again); event_a2 = bank.consolidate(a_again, a_revisit)
    report = {'format': 'wailah-lifelong-proof-v1', 'sequence': ['A', 'B', 'A'],
              'a_after_first_encounter': a_first, 'b_after_learning': b_score,
              'a_after_learning_b': a_after_b, 'a_checkpoint_unchanged_while_b_learned': a_unchanged,
              'a_after_revisit': a_revisit, 'a_revisit_improvement': a_revisit - a_first,
              'events': [event_a1, event_b, event_a2],
              'claim': 'Software proof of isolated retention and resumable improvement; not a broad-task benchmark.'}
    atomic_json(Path(workspace) / 'lifelong_proof.json', report)
    return report
