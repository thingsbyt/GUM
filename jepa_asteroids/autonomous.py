"""Automatic observe/act/store/train loop for the Asteroids adapter.

The paper-faithful world model remains unchanged. This extension supplies
exploration, outcome-goal selection, continual replay and orchestration.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import torch

from .config import Config
from .continual import ContinualLearner, Goal, ReplayBuffer
from .data import domain, game_config, progress
from .engine import Asteroids
from .evaluation import publish
from .planning import build_vocabulary, plan_to_goal
from .runtime import Guard, Stopped, atomic_json


def encoding_identity(cfg: Config, encoder) -> dict:
    return {'encoder': encoder.identity, 'storage_dtype': cfg.feature_dtype,
            'domain': domain(cfg)}


def encode_frames(cfg: Config, encoder, frames: np.ndarray, guard: Guard) -> np.ndarray:
    parts = []
    for start in range(0, len(frames), cfg.encoder_batch):
        guard.check()
        with torch.no_grad():
            value = encoder(torch.from_numpy(frames[start:start+cfg.encoder_batch])).float().cpu().numpy()
        parts.append(value)
    result = np.concatenate(parts).astype(cfg.feature_dtype, copy=False)
    expected = (len(frames), cfg.encoder_dim, *cfg.grid)
    if result.shape != expected or not np.isfinite(result).all():
        raise ValueError('Visual encoder returned invalid continual-learning features.')
    return result


def _epsilon_action(learner: ContinualLearner, previous: int) -> int:
    # Persistence is a generic exploration prior, not a game steering rule.
    if learner.rng.random() < 0.5:
        return previous
    return int(learner.rng.integers(learner.cfg.action_dim))


def _project_goal(learner: ContinualLearner, replay: ReplayBuffer, goal: Goal) -> torch.Tensor:
    feature = replay.goal_feature(goal).to(learner.device).unsqueeze(0)
    with torch.no_grad():
        return learner.model.projector(feature)


def run_episode(cfg: Config, workspace: Path, encoder, learner: ContinualLearner,
                replay: ReplayBuffer, guard: Guard, *, seed: int,
                learning: bool = True, replay_delay: float = 0.0) -> dict:
    game = Asteroids(game_config(cfg), seed)
    frames = [game.observe()]
    actions: list[int] = []
    feedback: list[list[float]] = []
    # Features are cached once as observations arrive; exact physics never enters them.
    feature_frames = [encode_frames(cfg, encoder, frames[0][None], guard)[0]]
    vocabulary = build_vocabulary(cfg.candidates, cfg.horizon, cfg.action_dim)
    previous = int(learner.rng.integers(cfg.action_dim))
    planned = 0
    active_goal: Goal | None = None
    goal_z = None
    goal_start_distance = None
    goal_age = 0
    task_start = time.monotonic()
    learner.model.eval()

    for decision in range(cfg.episode_steps):
        guard.check()
        can_plan = (learner.state['updates'] > 0 and len(feature_frames) >= cfg.history
                    and bool(replay.goals()))
        explore = (not can_plan or learner.rng.random() < learner.epsilon())
        if explore:
            action = _epsilon_action(learner, previous)
        else:
            if active_goal is None:
                active_goal = replay.select_goal()
                if active_goal is not None:
                    goal_z = _project_goal(learner, replay, active_goal)
                    goal_age = 0
                    publish(workspace, frames[-1], replay.goal_frame(active_goal))
            if active_goal is None:
                action = _epsilon_action(learner, previous)
            else:
                recent = torch.from_numpy(np.stack(feature_frames[-cfg.history:])).to(learner.device).unsqueeze(0)
                with torch.no_grad():
                    context = learner.model.projector(recent)
                    past_ids = torch.tensor(actions[-(cfg.history-1):], device=learner.device)
                    past = torch.nn.functional.one_hot(past_ids, cfg.action_dim).float().unsqueeze(0)
                    chosen = plan_to_goal(learner.model.predictor, context, past, goal_z,
                                          vocabulary, cfg.action_dim, cfg.candidate_chunk,
                                          check=guard.check)
                    if goal_start_distance is None:
                        goal_start_distance = float((context[:,-1:]-goal_z).square().mean())
                action = int(chosen.actions[0])
                planned += 1
        image, reward, ended, timeout, info = game.step(action)
        previous = action
        actions.append(action)
        frames.append(image)
        feedback.append([reward, float(info['hits']), float(ended), float(timeout)])
        feature_frames.append(encode_frames(cfg, encoder, image[None], guard)[0])
        publish(workspace, image, replay.goal_frame(active_goal) if active_goal else image)
        if learning:
            learner.state['environment_steps'] += 1
        if active_goal is not None:
            goal_age += 1
            if goal_age >= cfg.horizon or ended or timeout:
                with torch.no_grad():
                    actual = learner.model.projector(torch.from_numpy(
                        feature_frames[-1][None,None]).to(learner.device))
                    final_distance = float((actual-goal_z).square().mean())
                if learning:
                    replay.record_goal_result(active_goal, final_distance < float(goal_start_distance))
                active_goal, goal_z, goal_start_distance, goal_age = None, None, None, 0
        progress(workspace, 'learning' if learning else 'watching',
                 episode=learner.state['episodes']+1, decision=decision+1,
                 score=game.hits, lifetime_experience=learner.state['environment_steps'],
                 training_updates=learner.state['updates'], epsilon=learner.epsilon(),
                 activity='exploring' if explore else 'planning to an experienced outcome',
                 simulated_seconds=game.frames/60.0, wall_seconds=time.monotonic()-task_start,
                 message='Executing the game; exact engine state is outside the agent observation.')
        if replay_delay:
            time.sleep(replay_delay)
        if ended or timeout:
            break

    result = {'seed': int(seed), 'steps': len(actions), 'hits': int(game.hits),
              'return': float(game.total_reward), 'terminated': bool(game.terminated),
              'truncated': bool(game.truncated), 'planned_decisions': planned,
              'epsilon': learner.epsilon(), 'learning': learning}
    if learning:
        feature_array = np.asarray(feature_frames, dtype=cfg.feature_dtype)
        replay.add_episode(np.stack(frames), np.asarray(actions, dtype=np.int64),
                           np.asarray(feedback, dtype=np.float32), feature_array,
                           seed=seed,
                           policy='epsilon exploration + learned model goal planning',
                           model_updates=learner.state['updates'])
        learner.state['episodes'] += 1
        updates = learner.train(replay, guard, cfg.online_updates_per_episode)
        result['updates_completed'] = len(updates)
        milestone = (learner.state['updates'] == 0 or
                     learner.state['updates'] % cfg.online_milestone_updates < len(updates))
        learner.save(milestone=milestone)
    with (workspace/'episodes.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(result)+'\n')
    return result


def run_session(cfg: Config, workspace: Path, device: torch.device, guard: Guard,
                encoder, *, episodes: int, fresh: bool = False,
                learning: bool = True, replay_delay: float = 0.0) -> dict:
    if episodes < 1:
        raise ValueError('episodes must be positive.')
    if fresh and ((workspace/'brain.pt').exists() or (workspace/'continual').exists()):
        raise FileExistsError('Fresh start never erases a learner; choose an empty workspace.')
    encoding = encoding_identity(cfg, encoder)
    learner = ContinualLearner(cfg, workspace, device, encoding, fresh=fresh)
    replay = ReplayBuffer(cfg, workspace, encoding, learner.rng)
    if not learning and not learner.path.exists():
        raise RuntimeError('Watch without learning requires a saved brain.')
    if fresh:
        learner.save(milestone=True)
    results = []
    try:
        for _ in range(episodes):
            seed = cfg.seed + 3_000_000_000 + learner.state['episodes']
            results.append(run_episode(cfg, workspace, encoder, learner, replay, guard,
                                       seed=seed, learning=learning,
                                       replay_delay=replay_delay))
    except (Stopped, KeyboardInterrupt):
        if learning:
            learner.save()
        raise
    summary = {'learning': learning, 'episodes': len(results),
               'brain_episodes': learner.state['episodes'],
               'environment_steps': learner.state['environment_steps'],
               'training_updates': learner.state['updates'], 'results': results}
    atomic_json(workspace/'session.json', summary)
    progress(workspace, 'paused' if learning else 'watch-complete', metrics=summary,
             message='Session complete; brain and replay are saved.' if learning else
                     'Frozen watch complete; no experience or weight updates were saved.')
    return summary

