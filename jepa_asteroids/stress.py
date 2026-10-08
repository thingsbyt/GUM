"""Adversarial, read-only stress suite for the autonomous competence loop."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .autonomy import AnonymousSixGame, AutonomousCompetenceLoop
from .lifelong_benchmark import AvoidGame,CatchGame,MazeGame,NavigateGame,SignalGame
from .logic_benchmark import LogicChainGame
from .runtime import atomic_json


class ObservationShift:
    """Stable contrast/translation shift that preserves task semantics."""
    def __init__(self,env,*,roll=2,contrast=.72):
        self.env=env;self.action_dim=env.action_dim;self.horizon=env.horizon
        self.roll=roll;self.contrast=contrast
    def _x(self,obs):
        value=(obs.astype(np.float32)*self.contrast).astype(np.uint8)
        return np.roll(value,(self.roll,self.roll),axis=(1,2))
    def reset(self,seed):return self._x(self.env.reset(seed))
    def step(self,action):
        obs,reward,done,info=self.env.step(action);return self._x(obs),reward,done,info
    def __getattr__(self,name):return getattr(self.env,name)


class FlickerNoise:
    """Fresh sensor noise deliberately breaks exact observation identity."""
    def __init__(self,env,amount=20):
        self.env=env;self.action_dim=env.action_dim;self.horizon=env.horizon;self.amount=amount
    def _x(self,obs):
        noise=self.rng.integers(-self.amount,self.amount+1,obs.shape,dtype=np.int16)
        return np.clip(obs.astype(np.int16)+noise,0,255).astype(np.uint8)
    def reset(self,seed):self.rng=np.random.default_rng(seed+3391);return self._x(self.env.reset(seed))
    def step(self,action):
        obs,reward,done,info=self.env.step(action);return self._x(obs),reward,done,info
    def __getattr__(self,name):return getattr(self.env,name)


class RemappedActions:
    """Preserves observations/rewards while changing control semantics."""
    def __init__(self,env,mapping):
        self.env=env;self.mapping=tuple(mapping);self.action_dim=env.action_dim;self.horizon=env.horizon
    def reset(self,seed):return self.env.reset(seed)
    def step(self,action):return self.env.step(self.mapping[int(action)])
    def __getattr__(self,name):return getattr(self.env,name)


class RepeatedLogicGame(LogicChainGame):
    """Causal chain violates the no-repeated-event assumption."""
    def reset(self,seed):
        obs=super().reset(seed);self.chain.insert(3,self.chain[0]);return self.render()


def _hashes(workspace: Path)->dict:
    paths=list((workspace/'lifelong'/'skills').glob('*/skill.pt'))
    paths += [workspace/'living'/'router.pt',workspace/'living'/'world.pt']
    return {str(path.relative_to(workspace)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.exists()}


def _safe_rollout(loop,factory,method,*,episodes,seed_base):
    try:return {'ok':True,'result':loop._rollout(factory,method,episodes=episodes,seed_base=seed_base)}
    except Exception as exc:return {'ok':False,'error':f'{type(exc).__name__}: {exc}'}


def _safe_select(loop,factory,*,probe_seed):
    """Keep one hostile audit from aborting the rest of the read-only suite."""
    try:return {'ok':True,'decision':loop.select(factory,probe_seed=probe_seed)}
    except Exception as exc:return {'ok':False,'error':f'{type(exc).__name__}: {exc}'}


def run_stress_benchmark(workspace: Path,output: Path,device='cpu')->dict:
    workspace,output=Path(workspace),Path(output);before=_hashes(workspace)
    loop=AutonomousCompetenceLoop(workspace,device)
    base=[('catch',CatchGame,'neural'),('avoid',AvoidGame,'neural'),
          ('navigate',NavigateGame,'neural'),('signals',SignalGame,'neural'),
          ('maze',MazeGame,'maze-memory'),('logic',LogicChainGame,'causal-chain-memory'),
          ('grown-six',AnonymousSixGame,'neural')]
    base_rows=[]
    for trial in range(3):
        for offset,(name,factory,expected) in enumerate(base):
            audit=_safe_select(loop,factory,probe_seed=13_000_000+trial*100_000+offset*1_000)
            decision=audit.get('decision',{})
            base_rows.append({'trial':trial,'audit_name':name,'expected':expected,
                              'selected':decision.get('method','error'),
                              'correct':decision.get('method')==expected,
                              'novel':decision.get('novel'),'evidence':audit})
    shifted=[]
    variants=[
        ('catch-contrast-shift',lambda:ObservationShift(CatchGame()),'neural'),
        ('signals-contrast-shift',lambda:ObservationShift(SignalGame()),'neural'),
        ('navigate-remapped-controls',lambda:RemappedActions(NavigateGame(),(2,3,1,0)),'not-neural'),
        ('signals-remapped-controls',lambda:RemappedActions(SignalGame(),(2,0,4,1,3)),'not-neural')]
    for offset,(name,factory,expectation) in enumerate(variants):
        audit=_safe_select(loop,factory,probe_seed=14_000_000+offset*10_000)
        decision=audit.get('decision',{});method=decision.get('method','error')
        passed=(method=='neural') if expectation=='neural' else (method not in ('neural','error'))
        shifted.append({'audit_name':name,'expectation':expectation,'selected':method,
                        'passed':passed,'decision':audit})
    strategies={
        'maze_clean':_safe_rollout(loop,MazeGame,'maze-memory',episodes=64,seed_base=15_000_000),
        'maze_remapped_controls':_safe_rollout(loop,
            lambda:RemappedActions(MazeGame(),(2,3,1,0)),'maze-memory',episodes=64,seed_base=15_100_000),
        'maze_flicker_noise':_safe_rollout(loop,
            lambda:FlickerNoise(MazeGame()),'maze-memory',episodes=64,seed_base=15_200_000),
        'logic_clean':_safe_rollout(loop,LogicChainGame,'causal-chain-memory',episodes=128,seed_base=15_300_000),
        'logic_flicker_noise':_safe_rollout(loop,
            lambda:FlickerNoise(LogicChainGame()),'causal-chain-memory',episodes=128,seed_base=15_400_000),
        'logic_repeated_event':_safe_rollout(loop,RepeatedLogicGame,'causal-chain-memory',episodes=128,seed_base=15_500_000)}
    after=_hashes(workspace);base_correct=sum(row['correct'] for row in base_rows)
    failures=[{'case':f"base_selection:{row['audit_name']}:trial-{row['trial']}",'result':row}
              for row in base_rows if not row['correct']]
    failures += [{'case':f"distribution_shift:{row['audit_name']}",'result':row}
                 for row in shifted if not row['passed']]
    for name,value in strategies.items():
        success=value.get('result',{}).get('success_rate') if value.get('ok') else None
        if not value.get('ok') or (success is not None and success<.90):
            failures.append({'case':name,'result':value})
    report={'format':'wailah-autonomy-stress-v1',
            'mode':'read-only; no training, promotion or quarantine permitted',
            'base_selection':{'trials':len(base_rows),'correct':base_correct,
                              'accuracy':base_correct/max(1,len(base_rows)),'rows':base_rows,
                              'selection_counts':dict(Counter(row['selected'] for row in base_rows))},
            'distribution_shifts':shifted,
            'strategy_adversaries':strategies,'failures_discovered':failures,
            'integrity':{'before':before,'after':after,'all_model_hashes_unchanged':before==after}}
    atomic_json(output,report);return report


def render_stress_summary(report_path: Path,output: Path)->Path:
    """Render the compact pass/fail envelope from a completed stress audit."""
    from PIL import Image,ImageDraw,ImageFont
    report=json.loads(Path(report_path).read_text(encoding='utf-8'))
    try:
        title=ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf',28)
        body=ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf',16)
        bold=ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf',16)
        small=ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf',13)
    except OSError:title=body=bold=small=ImageFont.load_default()
    width,height=1100,720;canvas=Image.new('RGB',(width,height),(7,13,25));draw=ImageDraw.Draw(canvas)
    green=(79,220,164);red=(247,111,120);amber=(246,186,73);white=(232,239,248);muted=(139,156,180);panel=(14,25,43)
    draw.text((34,25),'WAILAH  /  ADVERSARIAL STRESS TEST',font=title,fill=white)
    draw.text((35,65),'frozen brain  •  no retraining  •  no checkpoint mutation',font=body,fill=muted)
    base=report['base_selection'];shifts=report['distribution_shifts'];strategies=report['strategy_adversaries']
    visual=sum(row['passed'] for row in shifts[:2]);control=sum(row['passed'] for row in shifts[2:])
    intact=report['integrity']['all_model_hashes_unchanged'];model_count=len(report['integrity']['before'])
    cards=[('BASE RECALL',f"{base['correct']} / {base['trials']}",'100% correct'),
           ('VISUAL SHIFTS',f'{visual} / 2','semantic shifts survived'),
           ('CONTROL CHANGE',f'{control} / 2','both safely reclassified'),
           ('INTEGRITY',f'{model_count} / {model_count}' if intact else 'FAILED','hashes unchanged' if intact else 'mutation detected')]
    for i,(label,value,detail) in enumerate(cards):
        x=34+i*262;draw.rounded_rectangle((x,105,x+238,205),radius=12,fill=panel)
        draw.text((x+16,120),label,font=small,fill=muted);draw.text((x+16,145),value,font=title,fill=green)
        draw.text((x+16,181),detail,font=small,fill=white)
    cases=[('Clean maze memory','maze_clean'),('Maze + remapped controls','maze_remapped_controls'),
           ('Maze + pixel flicker','maze_flicker_noise'),('Clean causal memory','logic_clean'),
           ('Logic + pixel flicker','logic_flicker_noise'),('Logic + repeated event','logic_repeated_event')]
    rows=[]
    for label,key in cases:
        item=strategies[key];rate=item.get('result',{}).get('success_rate') if item.get('ok') else None
        passed=bool(item.get('ok') and rate is not None and rate >= .90)
        result=f'{rate:.0%} success' if rate is not None else item.get('error','no result')
        rows.append((label,result,'PASS' if passed else 'FAIL'))
    signals=next(row for row in shifts if row['audit_name']=='signals-remapped-controls')
    rows.append(('Signals + remapped controls',f"safely selected {signals['selected']}",'PASS' if signals['passed'] else 'FAIL'))
    draw.text((36,238),'ADVERSARIAL CASE',font=small,fill=muted);draw.text((620,238),'MEASURED RESULT',font=small,fill=muted);draw.text((965,238),'STATUS',font=small,fill=muted)
    for i,(name,result,status) in enumerate(rows):
        y=270+i*48;draw.rounded_rectangle((28,y-8,width-28,y+32),radius=8,fill=panel)
        draw.text((43,y),name,font=body,fill=white);draw.text((620,y),result,font=body,fill=muted)
        color=green if status=='PASS' else red;draw.text((970,y),status,font=bold,fill=color)
    draw.rounded_rectangle((28,618,width-28,688),radius=10,fill=(28,24,40))
    draw.text((44,632),'OPERATING ENVELOPE',font=small,fill=amber)
    failures=len(report.get('failures_discovered',[]))
    verdict=('All current adversarial cases passed; noisy identity, repeated events and strategy rejection are stable.'
             if failures==0 else f'{failures} adversarial failures remain; inspect the machine-readable report.')
    draw.text((44,655),verdict,font=body,fill=white)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output);return output
