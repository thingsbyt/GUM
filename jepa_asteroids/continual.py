"""Persistent replay and online updates around the faithful AD-E2E-JEPA core.

This module is an extension, not part of the paper method. Task feedback is used
only to select observed goal outcomes. The world-model loss still receives only
frozen visual features and aligned actions.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

from .config import Config
from .data import domain, fingerprint, progress
from .model import WorldModel
from .objective import loss_from_embeddings
from .runtime import Guard, atomic_json, replace_file
from .training import (TRAIN_KEYS, atomic_torch, checkpoint_header,
                       method_signature, regularizer, verify_checkpoint)


REPLAY_FORMAT = 'wailah-continual-replay-v1'
BRAIN_FORMAT = 'wailah-continual-brain-v1'
ONLINE_KEYS = ('online_updates_per_episode','online_min_windows','online_replay_episodes',
               'online_recent_fraction','online_epsilon_start','online_epsilon_end',
               'online_epsilon_decay_steps','online_goal_library_size',
               'online_milestone_updates')


def online_settings(cfg: Config) -> dict:
    return {key: getattr(cfg, key) for key in ONLINE_KEYS}


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    replace_file(temp, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npy')
    np.save(temp, value, allow_pickle=False)
    replace_file(temp, path)


@dataclass(frozen=True)
class Goal:
    episode: str
    frame: int
    task_utility: float
    learning_progress: float
    attempts: int
    successes: int


class ReplayBuffer:
    """Bounded recent-plus-reservoir replay with explicit provenance."""
    def __init__(self, cfg: Config, workspace: Path, encoding: dict,
                 rng: np.random.Generator):
        self.cfg = cfg
        self.root = workspace / 'continual'
        self.raw = self.root / 'raw'
        self.features = self.root / 'features'
        self.path = self.root / 'manifest.json'
        self.rng = rng
        expected = {'format': REPLAY_FORMAT, 'domain': domain(cfg),
                    'method_signature': method_signature(cfg), 'encoding': encoding,
                    'online_settings': online_settings(cfg)}
        if self.path.exists():
            self.manifest = json.loads(self.path.read_text(encoding='utf-8'))
            for key, value in expected.items():
                if self.manifest.get(key) != value:
                    raise RuntimeError(f'Continual replay {key} differs; use a new brain workspace.')
        else:
            self.manifest = {**expected, 'seen_episodes': 0,
                             'archive_candidates': 0, 'recent': [], 'archive': []}
            self._save()

    def _save(self) -> None:
        atomic_json(self.path, self.manifest)

    @property
    def entries(self) -> list[dict]:
        return self.manifest['recent'] + self.manifest['archive']

    def _delete_entry(self, entry: dict) -> None:
        for folder, key in ((self.raw, 'raw'), (self.features, 'features')):
            path = folder / entry[key]
            if path.exists():
                path.unlink()

    def add_episode(self, frames: np.ndarray, actions: np.ndarray, feedback: np.ndarray,
                    features: np.ndarray, *, seed: int, policy: str,
                    model_updates: int) -> dict:
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError('Continual replay requires uint8 RGB frames.')
        if actions.shape != (len(frames)-1,) or feedback.shape != (len(actions), 4):
            raise ValueError('Actions and [reward,hits,terminated,truncated] feedback are misaligned.')
        expected = (len(frames), self.cfg.encoder_dim, *self.cfg.grid)
        if features.shape != expected or features.dtype != np.dtype(self.cfg.feature_dtype):
            raise ValueError(f'Frozen feature shape/dtype must be {expected}/{self.cfg.feature_dtype}.')
        number = int(self.manifest['seen_episodes'])
        identity = f'ep_{number:08d}'
        raw_name, feature_name = identity + '.npz', identity + '.npy'
        _atomic_npz(self.raw/raw_name, frames=frames, actions=actions.astype(np.int64),
                    feedback=feedback.astype(np.float32))
        _atomic_npy(self.features/feature_name, features)
        hits = feedback[:, 1] if len(feedback) else np.zeros(0)
        goal_frames = []
        if len(frames) > 1:
            # Outcomes, not controls: successful frames plus a few temporal anchors.
            chosen = set(int(i+1) for i in np.flatnonzero(hits > 0))
            chosen.update(np.linspace(1, len(frames)-1,
                                      min(4, len(frames)-1), dtype=int).tolist())
            for index in sorted(chosen):
                utility = float(2.0*hits[index-1] + index/max(1, len(frames)-1))
                goal_frames.append({'frame': index, 'task_utility': utility,
                                    'learning_progress': 0.0,
                                    'attempts': 0, 'successes': 0})
        entry = {'id': identity, 'raw': raw_name, 'features': feature_name,
                 'frames': len(frames), 'seed': int(seed), 'policy': policy,
                 'model_updates': int(model_updates), 'goal_frames': goal_frames,
                 'terminated': bool(feedback[-1,2]) if len(feedback) else False,
                 'truncated': bool(feedback[-1,3]) if len(feedback) else False,
                 'hits': int(hits.sum()),
                 'return': float(feedback[:,0].sum()) if len(feedback) else 0.0}
        self.manifest['seen_episodes'] = number + 1
        self.manifest['recent'].append(entry)
        self._rebalance()
        self._save()
        return entry

    def _rebalance(self) -> None:
        total = self.cfg.online_replay_episodes
        recent_cap = max(1, min(total, int(math.ceil(total*self.cfg.online_recent_fraction))))
        archive_cap = total - recent_cap
        while len(self.manifest['recent']) > recent_cap:
            candidate = self.manifest['recent'].pop(0)
            seen = int(self.manifest['archive_candidates']) + 1
            self.manifest['archive_candidates'] = seen
            archive = self.manifest['archive']
            if archive_cap == 0:
                self._delete_entry(candidate)
            elif len(archive) < archive_cap:
                archive.append(candidate)
            else:
                position = int(self.rng.integers(seen))
                if position < archive_cap:
                    self._delete_entry(archive[position])
                    archive[position] = candidate
                else:
                    self._delete_entry(candidate)

    def window_count(self) -> int:
        return sum(max(0, e['frames']-self.cfg.sequence_frames+1) for e in self.entries)

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        eligible = [e for e in self.entries if e['frames'] >= self.cfg.sequence_frames]
        if not eligible:
            raise RuntimeError('No replay episode is long enough for a training window.')
        recent_ids = {e['id'] for e in self.manifest['recent']}
        recent = [e for e in eligible if e['id'] in recent_ids]
        archive = [e for e in eligible if e['id'] not in recent_ids]
        features, actions, ids = [], [], []
        for _ in range(batch_size):
            pool = recent if recent and (not archive or self.rng.random() < self.cfg.online_recent_fraction) else archive
            entry = pool[int(self.rng.integers(len(pool)))]
            high = entry['frames'] - self.cfg.sequence_frames + 1
            start = int(self.rng.integers(high))
            cached = np.load(self.features/entry['features'], mmap_mode='r', allow_pickle=False)
            with np.load(self.raw/entry['raw'], allow_pickle=False) as raw:
                action_ids = raw['actions'][start:start+self.cfg.sequence_frames-1].copy()
            features.append(torch.from_numpy(np.array(cached[start:start+self.cfg.sequence_frames], copy=True)))
            actions.append(torch.nn.functional.one_hot(torch.from_numpy(action_ids).long(),
                                                       self.cfg.action_dim).float())
            ids.append(entry['id'])
        return torch.stack(features), torch.stack(actions), ids

    def goals(self) -> list[Goal]:
        result = []
        for entry in self.entries:
            for value in entry['goal_frames']:
                result.append(Goal(entry['id'], int(value['frame']),
                    float(value['task_utility']), float(value['learning_progress']),
                    int(value['attempts']), int(value['successes'])))
        result.sort(key=lambda g: (g.task_utility + g.learning_progress -
                                   0.5*max(0, g.attempts-g.successes), g.episode, g.frame),
                    reverse=True)
        return result[:self.cfg.online_goal_library_size]

    def select_goal(self) -> Goal | None:
        candidates = [g for g in self.goals() if g.attempts < 3 or g.successes > 0]
        if not candidates:
            return None
        # Soft rank sampling preserves exploration without treating prediction error as reward.
        ranks = np.arange(len(candidates), dtype=np.float64)
        probabilities = np.exp(-ranks/max(1.0, len(candidates)/4))
        probabilities /= probabilities.sum()
        return candidates[int(self.rng.choice(len(candidates), p=probabilities))]

    def goal_feature(self, goal: Goal) -> torch.Tensor:
        entry = next(e for e in self.entries if e['id'] == goal.episode)
        values = np.load(self.features/entry['features'], mmap_mode='r', allow_pickle=False)
        return torch.from_numpy(np.array(values[goal.frame:goal.frame+1], copy=True))

    def goal_frame(self, goal: Goal) -> np.ndarray:
        entry = next(e for e in self.entries if e['id'] == goal.episode)
        with np.load(self.raw/entry['raw'], allow_pickle=False) as values:
            return np.array(values['frames'][goal.frame], copy=True)

    def record_learning_progress(self, episode_ids: list[str], value: float) -> None:
        targets = set(episode_ids)
        for entry in self.entries:
            if entry['id'] in targets:
                for goal in entry['goal_frames']:
                    goal['learning_progress'] = 0.9*float(goal['learning_progress']) + 0.1*max(0.0, value)
        self._save()

    def record_goal_result(self, goal: Goal, success: bool) -> None:
        for entry in self.entries:
            if entry['id'] == goal.episode:
                for value in entry['goal_frames']:
                    if int(value['frame']) == goal.frame:
                        value['attempts'] += 1
                        value['successes'] += int(success)
                        self._save()
                        return


class ContinualLearner:
    """World model, optimizer, RNG and counters saved as one resumable brain."""
    def __init__(self, cfg: Config, workspace: Path, device: torch.device,
                 encoding: dict, *, fresh: bool = False):
        self.cfg, self.workspace, self.device, self.encoding = cfg, workspace, device, encoding
        self.path = workspace/'brain.pt'
        if fresh and self.path.exists():
            raise FileExistsError('Fresh start refuses to overwrite an existing brain; choose a new workspace.')
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed + 91_733)
        self.model = WorldModel(cfg).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate,
                                           weight_decay=cfg.weight_decay)
        self.state = {'updates': 0, 'environment_steps': 0, 'episodes': 0,
                      'created_unix': time.time(), 'last_learning_progress': 0.0}
        if self.path.exists() and not fresh:
            saved = torch.load(self.path, map_location='cpu', weights_only=True)
            if saved.get('brain_format') != BRAIN_FORMAT:
                raise ValueError('Not a WAILAH continual-learning brain.')
            verify_checkpoint(saved, cfg, encoding)
            if saved.get('online_settings') != online_settings(cfg):
                raise RuntimeError('Continual-learning settings differ; resume with the original config.')
            self.model.load_state_dict(saved['model'], strict=True)
            self.optimizer.load_state_dict(saved['optimizer'])
            self.state = saved['state']
            self.rng.bit_generator.state = saved['numpy_rng']
            torch.set_rng_state(saved['torch_rng'])
            if device.type == 'cuda' and saved.get('cuda_rng') is not None:
                torch.cuda.set_rng_state_all(saved['cuda_rng'])

    def epsilon(self) -> float:
        ratio = min(1.0, self.state['environment_steps']/self.cfg.online_epsilon_decay_steps)
        return self.cfg.online_epsilon_start + ratio*(self.cfg.online_epsilon_end-self.cfg.online_epsilon_start)

    def save(self, *, milestone: bool = False) -> None:
        header = checkpoint_header(self.cfg, self.encoding, self.state['updates'])
        model_payload = {**header, 'model': self.model.state_dict()}
        atomic_torch(self.workspace/'model.pt', model_payload)
        payload = {**model_payload, 'brain_format': BRAIN_FORMAT,
                   'optimizer': self.optimizer.state_dict(), 'state': dict(self.state),
                   'numpy_rng': self.rng.bit_generator.state,
                   'torch_rng': torch.get_rng_state(),
                   'cuda_rng': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None,
                   'training_settings': {k: getattr(self.cfg, k) for k in TRAIN_KEYS},
                   'online_settings': online_settings(self.cfg)}
        atomic_torch(self.path, payload)
        atomic_json(self.workspace/'brain.json', {'format': BRAIN_FORMAT, **self.state,
                    'epsilon': self.epsilon(), 'method_signature': method_signature(self.cfg),
                    'encoder': self.encoding['encoder']})
        if milestone:
            atomic_torch(self.workspace/'milestones'/f"brain_update_{self.state['updates']:08d}.pt", payload)

    def train(self, replay: ReplayBuffer, guard: Guard, updates: int) -> list[dict]:
        if replay.window_count() < max(self.cfg.batch_size, self.cfg.online_min_windows):
            return []
        results = []
        reg = regularizer(self.cfg)
        self.model.train()
        for _ in range(updates):
            guard.check(force=True)
            tick = time.monotonic()
            features, actions, episode_ids = replay.sample_batch(self.cfg.batch_size)
            features, actions = features.to(self.device), actions.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            z = self.model.projector(features)
            terms = loss_from_embeddings(self.model.predictor, z, actions, self.cfg, reg,
                                         check=guard.check)
            before = float(terms.prediction.detach())
            if not torch.isfinite(terms.total):
                raise FloatingPointError('Non-finite continual loss; the prior brain remains saved.')
            terms.total.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip,
                                                   error_if_nonfinite=True)
            self.optimizer.step()
            with torch.no_grad():
                updated_z = self.model.projector(features)
                after_terms = loss_from_embeddings(self.model.predictor, updated_z, actions,
                    self.cfg, lambda q, d=None: q.new_zeros(()), check=guard.check)
                after = float(after_terms.prediction)
            learning_progress = max(0.0, before-after)
            self.state['updates'] += 1
            self.state['last_learning_progress'] = learning_progress
            replay.record_learning_progress(episode_ids, learning_progress)
            record = {'update': self.state['updates'], 'prediction_before': before,
                      'prediction_after': after, 'learning_progress': learning_progress,
                      'total': float(terms.total.detach()), 'gradient_norm_before_clip': float(norm),
                      'seconds': time.monotonic()-tick, 'episodes': sorted(set(episode_ids))}
            with (self.workspace/'curiosity.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record)+'\n')
            results.append(record)
            del features, actions, z, updated_z, terms, after_terms
        return results

