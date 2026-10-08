"""Goal-image trials and an explicit receding-horizon control interface.

The default goal is an ACTUAL held-out future screenshot, as in the paper's
oracle-goal protocol. It is not a reward-derived target. Physics is used only
AFTER candidate ranking, to execute the selected controls and measure outcomes.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .config import Config
from .data import EpisodeWindows,game_config,progress
from .engine import Asteroids
from .planning import build_vocabulary,plan_to_goal
from .rendering import png_bytes
from .runtime import Guard,atomic_bytes,atomic_json
from .training import load_model,verify_checkpoint


def publish(workspace:Path,current:np.ndarray,goal:np.ndarray)->None:
    atomic_bytes(workspace/'current.png',png_bytes(current))
    atomic_bytes(workspace/'goal.png',png_bytes(goal))


def restore_game(cfg:Config,seed:int,actions:np.ndarray)->Asteroids:
    env=Asteroids(game_config(cfg),seed)
    for action in actions:
        env.step(int(action))
    return env


def evaluate(cfg:Config,workspace:Path,device:torch.device,guard:Guard,*,trials:int=8,
             replay_delay:float=0.10)->dict:
    if trials<1:raise ValueError('trials must be positive.')
    dataset=EpisodeWindows(cfg,workspace,'test',window_frames=cfg.history+cfg.horizon)
    model,payload=load_model(cfg,workspace,device)
    verify_checkpoint(payload,cfg,dataset.manifest['encoding'])
    vocabulary=build_vocabulary(cfg.candidates,cfg.horizon,cfg.action_dim)
    # One deterministic, middle-of-episode trial per independent held-out episode.
    names=sorted(dataset.entries)[:trials]
    results=[]
    with torch.no_grad():
        for name in names:
            guard.check(force=True)
            available=[i for i,(n,_) in enumerate(dataset.windows) if n==name]
            if not available:continue
            index=available[len(available)//2];_,start=dataset.windows[index]
            features,actions=dataset[index];entry=dataset.entries[name]
            with np.load(workspace/'raw'/entry['raw'],allow_pickle=False) as raw:
                frames=raw['frames'];all_actions=raw['actions']
            z=model.projector(features.unsqueeze(0).to(device))
            context=z[:,:cfg.history];goal_z=z[:,-1:]
            past=actions[:cfg.history-1].unsqueeze(0).to(device)
            goal=frames[start+cfg.history+cfg.horizon-1]
            current=frames[start+cfg.history-1]
            publish(workspace,current,goal)
            progress(workspace,'evaluate',trial=len(results)+1,total_trials=len(names),
                message='Ranking a fixed button vocabulary using terminal goal-embedding distance only.')
            if device.type=='cuda':torch.cuda.synchronize(device)
            tick=time.monotonic()
            selected=plan_to_goal(model.predictor,context,past,goal_z,vocabulary,
                cfg.action_dim,cfg.candidate_chunk,check=guard.check)
            if device.type=='cuda':torch.cuda.synchronize(device)
            seconds=time.monotonic()-tick
            # Separate diagnostic only. This true sequence NEVER enters selection.
            true_ids=all_actions[start+cfg.history-1:start+cfg.history-1+cfg.horizon]
            diagnostic=plan_to_goal(model.predictor,context,past,goal_z,
                torch.as_tensor(true_ids.copy()).view(1,-1),cfg.action_dim,1,check=guard.check)
            true_cost=float(diagnostic.costs[0]);rank=1+int((selected.costs<true_cost).sum())
            game=restore_game(cfg,entry['seed'],all_actions[:start+cfg.history-1])
            if not np.array_equal(game.observe(),current):
                raise RuntimeError('Replay did not reproduce the recorded starting image.')
            applied=[]
            for action in selected.actions.tolist():
                guard.check()
                image,_,ended,timeout,_=game.step(int(action));applied.append(int(action))
                publish(workspace,image,goal)
                progress(workspace,'evaluate',trial=len(results)+1,total_trials=len(names),
                    chosen_controls=selected.actions.tolist(),applied=len(applied),planning_seconds=seconds,
                    message='Executing the selected controls. No physics was consulted during candidate ranking.')
                if replay_delay:time.sleep(replay_delay)
                if ended or timeout:break
            difference=(game.observe().astype(np.float32)-goal.astype(np.float32))/255
            result={'episode':name,'seed':entry['seed'],'window_start':start,
                'chosen_vocabulary_index':selected.index,'selected_controls':selected.actions.tolist(),
                'executed_controls':applied,'selected_predicted_cost':float(selected.costs[selected.index]),
                'ground_truth_diagnostic_cost':true_cost,'ground_truth_competition_rank':rank,
                'top1_with_ties':rank<=1,'top5_with_ties':rank<=5,'planning_seconds':seconds,
                'terminal_pixel_mse':float(np.square(difference).mean()),'terminated':bool(game.terminated),
                'hits_observed_not_optimized':int(game.hits)}
            results.append(result)
            atomic_json(workspace/'evaluation.json',{'protocol':'held-out future-image goals; no reward scoring',
                'checkpoint_steps':payload['steps'],'results':results})
    if not results:raise RuntimeError('No held-out episodes long enough for a goal trial.')
    summary={'trials':len(results),'top1_with_ties':float(np.mean([r['top1_with_ties'] for r in results])),
        'top5_with_ties':float(np.mean([r['top5_with_ties'] for r in results])),
        'mean_planning_seconds':float(np.mean([r['planning_seconds'] for r in results])),
        'mean_terminal_pixel_mse':float(np.mean([r['terminal_pixel_mse'] for r in results]))}
    report={'protocol':'held-out future-image goals; not an unlimited survival/high-score policy',
        'checkpoint_steps':payload['steps'],'method_signature':payload['method_signature'],
        'summary':summary,'results':results}
    atomic_json(workspace/'evaluation.json',report)
    progress(workspace,'evaluated',metrics=summary,message='Held-out goal trials complete. See evaluation.json for each episode.')
    return report


def play_goal(cfg:Config,workspace:Path,device:torch.device,guard:Guard,encoder,goal_path:Path,
              *,seed:int=3000000007,decisions:int=100)->dict:
    """Game adapter: execute the first control and replan toward a user-supplied image."""
    if decisions<1:raise ValueError('decisions must be positive.')
    model,payload=load_model(cfg,workspace,device)
    expected=payload['encoding']['encoder']
    if encoder.identity!=expected:raise ValueError('Live encoder differs from the training cache encoder.')
    goal=np.array(Image.open(goal_path).convert('RGB'),copy=True)
    if goal.shape!=(cfg.image_height,cfg.image_width,3):
        raise ValueError(f'Goal must be a {cfg.image_width} x {cfg.image_height} RGB game screenshot. No silent crop.')
    game=Asteroids(game_config(cfg),seed);frames=[game.observe()];history_actions=[]
    # Warm-up is explicitly coast, not a hidden controller.
    for _ in range(cfg.history-1):
        image,_,ended,timeout,_=game.step(0);frames.append(image);history_actions.append(0)
        if ended or timeout:raise RuntimeError('Episode ended during context collection.')
    vocabulary=build_vocabulary(cfg.candidates,cfg.horizon,cfg.action_dim)
    with torch.no_grad():
        goal_z=model.projector(encoder(torch.from_numpy(goal[None])).unsqueeze(1))
        context=model.projector(encoder(torch.from_numpy(np.stack(frames))).unsqueeze(0))
        for step in range(decisions):
            guard.check(force=True)
            past=torch.nn.functional.one_hot(torch.tensor(history_actions[-(cfg.history-1):],device=device),cfg.action_dim).float()[None]
            chosen=plan_to_goal(model.predictor,context,past,goal_z,vocabulary,
                               cfg.action_dim,cfg.candidate_chunk,check=guard.check)
            action=int(chosen.actions[0]);frame,_,ended,timeout,_=game.step(action)
            publish(workspace,frame,goal);history_actions.append(action)
            progress(workspace,'play',decision=step+1,control=action,
                message='User-supplied goal pursuit. No survival reward, action-value head or generated goal.')
            if ended or timeout:break
            next_z=model.projector(encoder(torch.from_numpy(frame[None])).unsqueeze(1))
            context=torch.cat([context[:,1:],next_z],dim=1)
    result={'decisions':step+1,'terminated':bool(game.terminated),'goal':str(goal_path),
            'hits_observed_not_optimized':game.hits,'seed':seed}
    atomic_json(workspace/'play.json',result);progress(workspace,'play-ended',**result)
    return result
