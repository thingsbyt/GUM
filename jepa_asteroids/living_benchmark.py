"""Unlabeled hybrid-task benchmark for the WAILAH living system."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .lifelong import Skill, TaskSpec
from .lifelong_benchmark import SIZE, _block
from .living import LivingSystem
from .runtime import Guard, atomic_json


class HybridCatchAvoid:
    """Each falling object silently switches between catch and avoid objectives."""
    action_dim = 3
    def __init__(self, horizon: int = 96): self.horizon = horizon
    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed); self.step_count = 0
        self.paddle = int(self.rng.integers(4, SIZE - 4))
        self.correct = self.objects = self.caught = self.dodged = self.errors = 0
        self._spawn(); return self.render()
    def _spawn(self):
        self.object = int(self.rng.integers(2, SIZE - 2)); self.height = 2
        self.mode = 'catch' if self.rng.random() < .5 else 'avoid'
    def step(self, action: int):
        self.paddle = int(np.clip(self.paddle + (action == 2) - (action == 1), 3, SIZE - 4))
        self.height += 3; reward = -.002
        if self.height >= SIZE - 4:
            contact = abs(self.paddle - self.object) <= 3; self.objects += 1
            success = contact if self.mode == 'catch' else not contact
            if success:
                reward += 1.0; self.correct += 1
                if self.mode == 'catch': self.caught += 1
                else: self.dodged += 1
            else:
                reward -= .7; self.errors += 1
            self._spawn()
        self.step_count += 1; done = self.step_count >= self.horizon
        return self.render(), reward, done, {'correct': self.correct, 'objects': self.objects,
            'caught': self.caught, 'dodged': self.dodged, 'errors': self.errors, 'mode': self.mode}
    def render(self):
        frame = np.zeros((1, SIZE, SIZE), dtype=np.uint8)
        if self.mode == 'catch':
            _block(frame, self.object, self.height, 255, 1)
            frame[0, SIZE-3:SIZE-1, self.paddle-3:self.paddle+4] = 180
        else:
            _block(frame, self.object, self.height, 220, 2)
            frame[0, SIZE-3:SIZE-1, self.paddle-3:self.paddle+4] = 255
        return frame


def _evaluate_policy(policy, *, episodes: int = 128, seed_base: int = 9_600_000) -> dict:
    rows = []; route_correct = route_total = 0
    for episode in range(episodes):
        seed = seed_base + episode; rng = np.random.default_rng(seed + 77_701)
        env = HybridCatchAvoid(); obs = env.reset(seed); total = 0.0
        for _ in range(env.horizon):
            mode = env.mode
            if policy == 'random': action = int(rng.integers(3)); meta = None
            else: action, meta = policy(obs)
            if meta and meta.get('routes'):
                route_correct += int(meta['routes'][0]['expert'] == mode); route_total += 1
            obs, reward, done, info = env.step(action); total += reward
            if done: break
        rows.append({'return': total, **info})
    correct = sum(x['correct'] for x in rows); objects = sum(x['objects'] for x in rows)
    return {'episodes': episodes, 'mean_return': float(np.mean([x['return'] for x in rows])),
            'success_rate': correct / max(1, objects),
            'mean_errors': float(np.mean([x['errors'] for x in rows])),
            'router_frame_accuracy': route_correct / max(1, route_total) if route_total else None}


def _single_policy(living: LivingSystem, task_id: str):
    expert = next(x for x in living.experts if x.task_id == task_id)
    def policy(obs): return expert.skill.act(obs), {'routes': [{'expert': task_id, 'weight': 1.0}]}
    return policy


def _train_scratch(root: Path, device: torch.device, guard: Guard | None,
                   episodes: int = 45) -> tuple[Skill,dict]:
    spec = TaskSpec('hybrid-scratch', (1, SIZE, SIZE), 3, replay_capacity=12_000,
                    batch_size=64, latent_dim=128, learning_rate=5e-4,
                    discount=.97, target_interval=100, seed=404)
    skill = Skill(root / 'hybrid_scratch', spec, device); start = time.monotonic()
    for episode in range(episodes):
        if guard is not None: guard.check()
        env = HybridCatchAvoid(); obs = env.reset(8_700_000 + episode)
        epsilon = 1.0 + episode / max(1, episodes - 1) * (.05 - 1.0)
        for _ in range(env.horizon):
            action = skill.act(obs, epsilon=epsilon); nxt, reward, done, _ = env.step(action)
            skill.observe(obs, action, reward, nxt, done); obs = nxt
            if done: break
        skill.learn(8)
    skill.save()
    return skill, {'episodes': episodes, 'updates': episodes * 8,
                   'seconds': time.monotonic() - start}


def run_living_benchmark(workspace: Path, output: Path, device: torch.device | str = 'cpu',
                         guard: Guard | None = None, *, router_updates: int = 600,
                         world_updates: int = 900) -> dict:
    """Build the shared organism and evaluate zero-shot composition against baselines."""
    workspace, output, device = Path(workspace), Path(output), torch.device(device)
    living = LivingSystem(workspace, device); build = living.build(router_updates, world_updates)
    random = _evaluate_policy('random')
    catch_only = _evaluate_policy(_single_policy(living, 'catch'))
    avoid_only = _evaluate_policy(_single_policy(living, 'avoid'))
    # One active expert per frame is the correct sparse limit for two mutually
    # exclusive objectives; the selected expert may change on the very next frame.
    composed = _evaluate_policy(lambda obs: living.act(obs, 3, top_k=1))
    scratch, scratch_training = _train_scratch(output.parent / 'living_scratch', device, guard)
    scratch_result = _evaluate_policy(lambda obs: (scratch.act(obs), None))
    skill_hashes_after = {x.task_id: hashlib.sha256(x.skill.path.read_bytes()).hexdigest()
                          for x in living.experts}
    report = {'format': 'wailah-living-hybrid-benchmark-v1',
              'input': 'unlabeled 32x32 pixel stream; catch/avoid mode is never passed to controller',
              'build': build, 'hybrid': {'random': random, 'catch_only': catch_only,
              'avoid_only': avoid_only, 'scratch_45_episodes': scratch_result,
              'zero_shot_sparse_composer': composed}, 'scratch_training': scratch_training,
              'positive_transfer_vs_scratch': composed['success_rate'] - scratch_result['success_rate'],
              'positive_transfer_vs_random': composed['success_rate'] - random['success_rate'],
              'source_skills_unchanged': skill_hashes_after == build['source_skill_hashes']}
    output.parent.mkdir(parents=True, exist_ok=True); atomic_json(output, report); return report


def render_living_showcase(workspace: Path, output: Path,
                           device: torch.device | str = 'cpu') -> Path:
    """Render an audited episode with the pixel router's live expert choice."""
    from PIL import Image, ImageDraw, ImageFont
    living = LivingSystem(workspace, device); env = HybridCatchAvoid(horizon=96)
    obs = env.reset(9_950_031); frames = []; correct = objects = 0
    for step in range(env.horizon):
        hidden_mode = env.mode
        action, meta = living.act(obs, 3, top_k=1)
        selected = meta['routes'][0]['expert']; confidence = meta['routes'][0]['weight']
        nxt, reward, done, info = env.step(action)
        frames.append((obs.copy(), step, hidden_mode, selected, confidence, action,
                       reward, info['correct'], info['objects']))
        obs = nxt; correct, objects = info['correct'], info['objects']
        if done: break
    scale = 7; game_size = SIZE * scale; side = 238; font = ImageFont.load_default()
    animation = []
    for raw, step, mode, selected, confidence, action, reward, correct, objects in frames:
        canvas = Image.new('RGB', (game_size + side, game_size), (7, 13, 25))
        tint = np.array((40, 210, 230) if mode == 'catch' else (245, 95, 90), dtype=np.float32)
        rgb = (raw[0, ..., None].astype(np.float32) / 255.0 * tint).astype(np.uint8)
        image = Image.fromarray(rgb, mode='RGB').resize((game_size, game_size), Image.Resampling.NEAREST)
        canvas.paste(image, (0, 0)); draw = ImageDraw.Draw(canvas)
        x = game_size + 16; white = (230, 237, 247); muted = (135, 151, 174)
        draw.text((x, 16), 'WAILAH / LIVING', font=font, fill=(96, 220, 170))
        draw.text((x, 39), 'input: pixels only', font=font, fill=muted)
        draw.text((x, 61), f'frame {step + 1:02d} / 96', font=font, fill=white)
        draw.text((x, 86), 'AUDIT (not input)', font=font, fill=muted)
        draw.text((x, 103), f'hidden objective: {mode.upper()}', font=font,
                  fill=tuple(int(x) for x in tint))
        draw.text((x, 129), 'AGENT INFERENCE', font=font, fill=muted)
        draw.text((x, 146), f'expert: {selected.upper()}', font=font,
                  fill=(96, 220, 170) if selected == mode else (245, 95, 90))
        draw.text((x, 163), f'route weight: {confidence:.3f}', font=font, fill=white)
        draw.text((x, 180), f'action: {action}  reward: {reward:+.3f}', font=font, fill=white)
        rate = correct / max(1, objects)
        draw.text((x, 205), f'success: {correct}/{objects}  ({rate:.0%})', font=font, fill=white)
        animation.append(canvas)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation[0].save(output, save_all=True, append_images=animation[1:], duration=90,
                      loop=0, optimize=False, disposal=2)
    return output
