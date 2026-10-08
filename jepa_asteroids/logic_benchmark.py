"""Sparse-reward causal-chain puzzle for testing online logic and memory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .living import LivingSystem
from .runtime import atomic_json
from .state_memory import perceptual_state_key


SIZE = 32


class LogicChainGame:
    """A hidden permutation must be completed; only the whole chain is rewarded."""
    action_dim = 5
    def __init__(self, horizon: int = 64): self.horizon = horizon
    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed); self.chain = self.rng.permutation(5).tolist()
        self.progress = self.step_count = self.resets = 0; self.completed = []
        return self.render()
    def step(self, action: int):
        correct = int(action) == self.chain[self.progress]
        reward = 0.0
        if correct:
            self.completed.append(int(action)); self.progress += 1
        else:
            self.progress = 0; self.completed = []; self.resets += 1
        success = self.progress == len(self.chain)
        if success: reward = 1.0
        self.step_count += 1; done = success or self.step_count >= self.horizon
        return self.render(), reward, done, {'success': success, 'steps': self.step_count,
            'resets': self.resets, 'progress': self.progress, 'chain_audit_only': self.chain}
    def render(self):
        frame = np.zeros((1,SIZE,SIZE),dtype=np.uint8)
        # The five unrelated event objects. Their shapes identify controls but do
        # not disclose order. Completed events leave a visible causal trace.
        centers = (4,10,16,22,28)
        for action, x in enumerate(centers):
            value = 230 if action in self.completed else 85
            if action == 0: frame[0,18:29,x-1:x+1] = value
            elif action == 1: frame[0,22:24,x-3:x+4] = value
            elif action == 2:
                for d in range(-3,4): frame[0,23+d,x+d] = value; frame[0,23+d,x-d] = value
            elif action == 3:
                frame[0,19:28,x-3:x-2] = value; frame[0,19:28,x+2:x+3] = value
                frame[0,19:20,x-3:x+3] = value; frame[0,27:28,x-3:x+3] = value
            else:
                frame[0,19:28,x-1:x+1] = value; frame[0,22:24,x-3:x+4] = value
        # Five neutral sockets show how many events remain, not which order is right.
        for index in range(5):
            x = 4 + index*6; frame[0,4:7,x-1:x+2] = 170 if index < self.progress else 35
        return frame


class CausalChainMemory:
    """Infers a causal path, including repeated events, from visual transitions."""
    def __init__(self,action_dim:int=5):self.action_dim=action_dim
    def reset(self, observation: np.ndarray):
        self.root = self._key(observation); self.state_keys = [self.root]
        self.prefix = []; self.tried = []
        self.pending = None
    @staticmethod
    def _key(observation: np.ndarray) -> str:
        return perceptual_state_key(observation)
    def _tried_at(self, depth: int) -> set[int]:
        while len(self.tried) <= depth:self.tried.append(set())
        return self.tried[depth]
    def act(self, observation: np.ndarray) -> int:
        key = self._key(observation)
        try: depth = self.state_keys.index(key)
        except ValueError: depth = 0
        if depth < len(self.prefix):
            action = self.prefix[depth]; self.pending = ('replay',depth,action); return action
        tried=self._tried_at(depth)
        # Prefer unseen events, retaining the original permutation efficiency, but
        # then test previously used events too.  Repetition and loops are legal.
        available=[action for action in range(self.action_dim) if action not in tried]
        available.sort(key=lambda action:(action in self.prefix,action))
        # A hostile or aliased state must degrade into another bounded probe rather
        # than indexing an empty list and aborting the autonomous selector.
        action=available[0] if available else min(range(self.action_dim),key=lambda x:sum(x in row for row in self.tried))
        self.pending = ('hypothesis',depth,action); return action
    def observe(self, next_observation: np.ndarray, done: bool) -> None:
        if self.pending is None:return
        kind, depth, action = self.pending
        if kind != 'hypothesis': return
        nxt = self._key(next_observation)
        if done or nxt != self.root:
            if depth == len(self.prefix):self.prefix.append(action)
            else:self.prefix[depth]=action
            if not done:
                if len(self.state_keys) == depth + 1: self.state_keys.append(nxt)
                else: self.state_keys[depth + 1] = nxt
        else:
            self._tried_at(depth).add(action)


def _evaluate(kind, living: LivingSystem | None = None, *, episodes: int = 512,
              seed_base: int = 10_100_000) -> dict:
    rows = []
    for episode in range(episodes):
        seed = seed_base + episode; rng = np.random.default_rng(seed + 11_707)
        env = LogicChainGame(); obs = env.reset(seed)
        reasoner = CausalChainMemory() if kind == 'causal_memory' else None
        if reasoner: reasoner.reset(obs)
        while True:
            if kind == 'random': action = int(rng.integers(5))
            elif kind == 'living_zero_shot': action, _ = living.act(obs,5,top_k=1)
            else: action = reasoner.act(obs)
            nxt,reward,done,info = env.step(action)
            if reasoner: reasoner.observe(nxt,done)
            obs = nxt
            if done: break
        rows.append(info)
    successes = [row for row in rows if row['success']]
    return {'episodes': episodes, 'success_rate': len(successes)/episodes,
            'mean_steps_on_success': float(np.mean([row['steps'] for row in successes])) if successes else None,
            'worst_steps_on_success': max((row['steps'] for row in successes),default=None),
            'mean_resets': float(np.mean([row['resets'] for row in rows]))}


def _novelty_sample(living: LivingSystem, count: int = 512) -> dict:
    env = LogicChainGame(horizon=count); obs = env.reset(10_333_001)
    rng = np.random.default_rng(808); values = {key: [] for key in ('obs','actions','rewards','next','dones')}
    for _ in range(count):
        action = int(rng.integers(5)); nxt,reward,done,_ = env.step(action)
        values['obs'].append(obs); values['actions'].append(action); values['rewards'].append(reward)
        values['next'].append(nxt); values['dones'].append(done); obs = nxt
        if done: obs = env.reset(int(rng.integers(2**31)))
    return living.assess_experience(np.asarray(values['obs']),np.asarray(values['actions']),
        np.asarray(values['rewards']),np.asarray(values['next']),np.asarray(values['dones']),5)


def run_logic_benchmark(workspace: Path, output: Path) -> dict:
    workspace, output = Path(workspace), Path(output); living = LivingSystem(workspace)
    hashes_before = {x.task_id: hashlib.sha256(x.skill.path.read_bytes()).hexdigest()
                     for x in living.experts}
    random = _evaluate('random'); zero = _evaluate('living_zero_shot',living)
    causal = _evaluate('causal_memory'); novelty = _novelty_sample(living)
    hashes_after = {x.task_id: hashlib.sha256(x.skill.path.read_bytes()).hexdigest()
                    for x in living.experts}
    report = {'format':'wailah-causal-chain-logic-v1',
              'puzzle':'hidden random permutation of five unrelated visual events',
              'reward':'zero for every intermediate event; +1 only after the complete correct chain',
              'wrong_action':'silently resets all progress',
              'evaluation':'512 unseen chains for each method',
              'random':random,'living_zero_shot':zero,'causal_memory':causal,
              'novelty_reaction':novelty,
              'existing_skill_hashes_unchanged':hashes_before==hashes_after,
              'promoted':causal['success_rate']>=.95}
    atomic_json(output,report)
    strategy_path=workspace/'living'/'strategies.json'
    if strategy_path.exists(): strategies=json.loads(strategy_path.read_text(encoding='utf-8'))
    else: strategies={'format':'wailah-strategy-skills-v1','strategies':{}}
    if report['promoted']:
        strategies['strategies']['causal-chain-memory']={
            'status':'promoted','mechanism':'pixel-state hypothesis testing and prefix replay',
            'held_out_success_rate':causal['success_rate'],'evaluation_chains':causal['episodes']}
        atomic_json(strategy_path,strategies)
    return report


def render_logic_showcase(output: Path, *, seed: int = 10_444_019) -> Path:
    from PIL import Image,ImageDraw,ImageFont
    env=LogicChainGame(); obs=env.reset(seed); reasoner=CausalChainMemory(); reasoner.reset(obs); rows=[]
    while True:
        action=reasoner.act(obs); nxt,reward,done,info=env.step(action); reasoner.observe(nxt,done)
        rows.append((obs.copy(),action,reward,list(reasoner.prefix),dict(info))); obs=nxt
        if done: break
    scale=7; panel=SIZE*scale; side=270; font=ImageFont.load_default(); animation=[]
    for index,(raw,action,reward,prefix,info) in enumerate(rows):
        canvas=Image.new('RGB',(panel+side,panel),(7,13,25)); gray=raw[0]
        rgb=np.stack((gray,gray,gray),axis=-1); image=Image.fromarray(rgb,mode='RGB').resize(
            (panel,panel),Image.Resampling.NEAREST); canvas.paste(image,(0,0))
        draw=ImageDraw.Draw(canvas); x=panel+16; white=(230,237,247); muted=(135,151,174); green=(96,220,170)
        draw.text((x,16),'LOGIC / CAUSAL CHAIN',font=font,fill=green)
        draw.text((x,39),'reward only after full chain',font=font,fill=muted)
        draw.text((x,66),f'attempt {index+1} / 64',font=font,fill=white)
        draw.text((x,88),f'testing event: {action}',font=font,fill=white)
        draw.text((x,110),f'observed reward: {reward:+.1f}',font=font,fill=white)
        draw.text((x,137),'INFERRED CAUSAL PREFIX',font=font,fill=muted)
        draw.text((x,156),' -> '.join(map(str,prefix)) or '(none yet)',font=font,fill=green)
        draw.text((x,181),f'resets observed: {info["resets"]}',font=font,fill=white)
        draw.text((x,204),'SUCCESS' if info['success'] else 'hypothesis testing',font=font,
                  fill=green if info['success'] else white)
        animation.append(canvas)
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    animation[0].save(output,save_all=True,append_images=animation[1:],duration=220,
        loop=0,optimize=False,disposal=2); return output
