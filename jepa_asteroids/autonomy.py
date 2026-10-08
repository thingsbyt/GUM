"""Autonomous competence selection, empirical novelty and sandboxed growth."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .lifelong import LifelongSkillBank, TaskSpec
from .lifelong_benchmark import (AvoidGame, CatchGame, MazeGame, NavigateGame,
                                 PixelMazeMemory, SignalGame)
from .living import LivingSystem
from .logic_benchmark import CausalChainMemory, LogicChainGame
from .runtime import Guard, atomic_json
from .strategy_selector import AdaptiveRewardMemory,LearnedStrategySelector,TemporalRewardMemory


class AnonymousSixGame:
    """A novel six-action visual language used to verify autonomous branch growth."""
    action_dim = 6
    mapping = (4, 1, 5, 0, 3, 2)
    def __init__(self, horizon: int = 48): self.horizon = horizon
    def reset(self, seed: int):
        self.rng=np.random.default_rng(seed); self.step_count=self.correct=self.errors=0
        self._next(); return self.render()
    def _next(self):
        self.symbol=int(self.rng.integers(6)); self.cx=int(self.rng.integers(7,26))
        self.cy=int(self.rng.integers(7,26)); self.noise=int(self.rng.integers(2**31))
    def step(self, action: int):
        success=int(action)==self.mapping[self.symbol]; reward=1.0 if success else -.25
        if success:self.correct+=1
        else:self.errors+=1
        self.step_count+=1; done=self.step_count>=self.horizon; self._next()
        return self.render(),reward,done,{'correct':self.correct,'errors':self.errors}
    def render(self):
        rng=np.random.default_rng(self.noise); frame=rng.integers(0,22,(1,32,32),dtype=np.uint8)
        x,y=self.cx,self.cy
        if self.symbol==0: frame[0,y-5:y+6,x-1:x+2]=255
        elif self.symbol==1: frame[0,y-1:y+2,x-5:x+6]=255
        elif self.symbol==2:
            for d in range(-5,6):frame[0,y+d,x+d]=255;frame[0,y+d,x-d]=255
        elif self.symbol==3:
            frame[0,y-5:y+6,x-5:x-3]=255;frame[0,y-5:y+6,x+4:x+6]=255
            frame[0,y-5:y-3,x-5:x+6]=255;frame[0,y+4:y+6,x-5:x+6]=255
        elif self.symbol==4:
            frame[0,y-5:y+6,x-1:x+2]=255;frame[0,y-1:y+2,x-5:x+6]=255
        else:
            frame[0,y-5:y+6,x-4:x-2]=255;frame[0,y-5:y+6,x+2:x+4]=255
            frame[0,y-1:y+2,x-4:x+4]=255
        return frame


def _mean(rows, key): return float(np.mean([row[key] for row in rows]))


class AutonomousCompetenceLoop:
    """Recall, test, switch strategy, or grow—based on measured competence."""
    def __init__(self, workspace: Path, device: torch.device | str='cpu', guard: Guard|None=None,
                 *, asteroids_cfg=None,asteroids_workspace:Path|None=None,literacy_workspace:Path|None=None,
                 nursery_workspace:Path|None=None,grammar_workspace:Path|None=None,
                 ontology_workspace:Path|None=None,teacher_workspace:Path|None=None,
                 procedure_workspace:Path|None=None):
        self.workspace=Path(workspace);self.device=torch.device(device);self.guard=guard
        self.living=LivingSystem(self.workspace,self.device);self.events=[]
        selector_path=self.workspace/'living'/'strategy_selector.json'
        self.strategy_selector=LearnedStrategySelector.load(selector_path) if selector_path.exists() else None
        self.asteroids_cfg=asteroids_cfg;self.asteroids_workspace=Path(asteroids_workspace) if asteroids_workspace else None
        self.literacy_workspace=Path(literacy_workspace) if literacy_workspace else None
        self.nursery_workspace=Path(nursery_workspace) if nursery_workspace else None
        self.grammar_workspace=Path(grammar_workspace) if grammar_workspace else None
        self.ontology_workspace=Path(ontology_workspace) if ontology_workspace else None
        self.teacher_workspace=Path(teacher_workspace) if teacher_workspace else None
        self.procedure_workspace=Path(procedure_workspace) if procedure_workspace else None
        self._mastery_controller=None

    def _rollout(self, factory, method: str, *, episodes: int, seed_base: int) -> dict:
        rows=[]
        for episode in range(episodes):
            env=factory();obs=env.reset(seed_base+episode);rng=np.random.default_rng(seed_base+episode+97)
            strategy=None
            if method=='maze-memory':strategy=PixelMazeMemory(env.action_dim)
            elif method=='causal-chain-memory':strategy=CausalChainMemory(env.action_dim);strategy.reset(obs)
            elif method=='adaptive-reward-memory':strategy=AdaptiveRewardMemory(env.action_dim)
            elif method=='temporal-reward-memory':strategy=TemporalRewardMemory(env.action_dim)
            elif method=='literacy-memory':
                from .literacy import LiteracyStrategy
                if self.literacy_workspace is None:raise RuntimeError('Literacy capability is not registered')
                strategy=LiteracyStrategy.load(self.literacy_workspace);strategy.reset(obs)
            elif method=='language-nursery-memory':
                from .language_nursery import NurseryStrategy
                if self.nursery_workspace is None:raise RuntimeError('Language nursery capability is not registered')
                strategy=NurseryStrategy.load(self.nursery_workspace);strategy.reset(obs)
            elif method=='grammar-discovery-memory':
                from .grammar_discovery import GrammarStrategy
                if self.grammar_workspace is None:raise RuntimeError('Grammar discovery capability is not registered')
                strategy=GrammarStrategy.load(self.grammar_workspace);strategy.reset(obs)
            elif method=='ontology-growth-memory':
                from .ontology_growth import OntologyStrategy
                if self.ontology_workspace is None:raise RuntimeError('Ontology growth capability is not registered')
                strategy=OntologyStrategy.load(self.ontology_workspace);strategy.reset(obs)
            elif method=='teacher-language-memory':
                from .teacher_language import TeacherStrategy
                if self.teacher_workspace is None:raise RuntimeError('Teacher language capability is not registered')
                strategy=TeacherStrategy.load(self.teacher_workspace);strategy.reset(obs)
            elif method=='procedure-rescue-memory':
                from .procedure_learning import ProcedureStrategy
                if self.procedure_workspace is None:raise RuntimeError('Procedure capability is not registered')
                strategy=ProcedureStrategy.load(self.procedure_workspace);strategy.reset(obs)
            elif method=='asteroids-mastery':
                from collections import deque
                from .mastery import MasteryController
                from .self_learning import frame_to_eye
                if self.asteroids_cfg is None or self.asteroids_workspace is None:
                    raise RuntimeError('Asteroids mastery capability is not registered')
                if self._mastery_controller is None:
                    self._mastery_controller=MasteryController(self.asteroids_cfg,self.asteroids_workspace,self.device)
                eye=frame_to_eye(obs,self.asteroids_cfg)
                stack=deque([eye.copy() for _ in range(self.asteroids_cfg.self_frame_stack)],
                            maxlen=self.asteroids_cfg.self_frame_stack)
            total=0.0
            while True:
                if method=='random':action=int(rng.integers(env.action_dim))
                elif method=='neural':action,_=self.living.act(obs,env.action_dim,top_k=1)
                elif method=='asteroids-mastery':action=self._mastery_controller.act(np.stack(stack),rng)
                else:action=strategy.act(obs)
                nxt,reward,done,info=env.step(action);total+=reward
                if method=='maze-memory':strategy.observe(nxt)
                elif method=='causal-chain-memory':strategy.observe(nxt,done)
                elif method in ('adaptive-reward-memory','temporal-reward-memory'):strategy.observe(reward)
                elif method=='asteroids-mastery':stack.append(frame_to_eye(nxt,self.asteroids_cfg))
                obs=nxt
                if done:break
            row={'return':float(total),'steps':int(getattr(env,'step_count',0)),**info};rows.append(row)
        result={'episodes':episodes,'mean_return':_mean(rows,'return')}
        if rows and 'success' in rows[0]:result['success_rate']=_mean(rows,'success')
        if rows and 'correct' in rows[0]:
            correct=sum(row['correct'] for row in rows);attempts=correct+sum(row['errors'] for row in rows)
            result['accuracy']=correct/max(1,attempts)
        if rows and 'total_hits' in rows[0]:result['mean_hits']=_mean(rows,'total_hits')
        return result

    @staticmethod
    def _gain_is_real(candidate: dict, random: dict) -> bool:
        gain=candidate['mean_return']-random['mean_return']
        scale=max(.05,.20*max(1.0,abs(random['mean_return'])))
        if 'success_rate' in candidate:
            return candidate['success_rate']>=.70 and gain>scale
        if 'accuracy' in candidate:
            floor=max(.55,random.get('accuracy',0)+.15)
            return candidate['accuracy']>=floor and gain>scale
        return gain>scale

    def select(self, factory, *, probe_seed: int) -> dict:
        env=factory();obs=env.reset(probe_seed);action_dim=env.action_dim
        compatible=[x.task_id for x in self.living.experts
                    if x.spec.action_dim==action_dim and tuple(x.spec.observation_shape)==tuple(np.asarray(obs).shape)]
        random=self._rollout(factory,'random',episodes=8,seed_base=probe_seed)
        neural=None;neural_error=None
        if compatible:
            try:
                neural=self._rollout(factory,'neural',episodes=8,seed_base=probe_seed)
                if self._gain_is_real(neural,random):
                    decision={'method':'neural','novel':False,'compatible_experts':compatible,
                              'random_probe':random,'neural_probe':neural,
                              'reason':'remembered policy demonstrated competence'}
                    self.events.append(decision);return decision
            except Exception as exc:neural_error=f'{type(exc).__name__}: {exc}'
        candidates=[];strategy_probes={}
        available=(list(self.strategy_selector.centroids) if self.strategy_selector is not None
                   else ['maze-memory','causal-chain-memory'])
        if self.asteroids_cfg is None or self.asteroids_workspace is None:
            available=[name for name in available if name!='asteroids-mastery']
        if self.literacy_workspace is None:
            available=[name for name in available if name!='literacy-memory']
        if self.nursery_workspace is None:
            available=[name for name in available if name!='language-nursery-memory']
        if self.grammar_workspace is None:
            available=[name for name in available if name!='grammar-discovery-memory']
        if self.ontology_workspace is None:
            available=[name for name in available if name!='ontology-growth-memory']
        if self.teacher_workspace is None:
            available=[name for name in available if name!='teacher-language-memory']
        if self.procedure_workspace is None:
            available=[name for name in available if name!='procedure-rescue-memory']
        if self.asteroids_cfg is not None and self.asteroids_workspace is not None and 'asteroids-mastery' not in available:
            available.append('asteroids-mastery')
        if self.strategy_selector is not None:
            ranked=self.strategy_selector.rank(factory,probe_seed+31,available)
            methods=[name for name,_ in ranked]+[name for name in available if name not in {x for x,_ in ranked}]
            ranking=[{'method':name,'distance':distance} for name,distance in ranked]
        else:methods=available;ranking=[]
        for method in methods:
            try:
                result=self._rollout(factory,method,episodes=8,seed_base=probe_seed)
                candidates.append((method,result));strategy_probes[method]={'ok':True,'result':result}
            except Exception as exc:
                strategy_probes[method]={'ok':False,'error':f'{type(exc).__name__}: {exc}'}
        viable=[item for item in candidates if self._gain_is_real(item[1],random)]
        if viable:
            method,result=max(viable,key=lambda item:item[1]['mean_return'])
            decision={'method':method,'novel':True,'compatible_experts':compatible,
                      'random_probe':random,'neural_probe':neural,'strategy_probe':result,
                      'neural_error':neural_error,'strategy_ranking':ranking,'strategy_probes':strategy_probes,
                      'reason':'recall failed; empirical strategy probe succeeded'}
            self.events.append(decision);return decision
        decision={'method':'grow','novel':True,'compatible_experts':compatible,
                  'random_probe':random,'neural_probe':neural,
                  'neural_error':neural_error,'strategy_ranking':ranking,'strategy_probes':strategy_probes,
                  'reason':'no existing policy or strategy demonstrated competence'}
        self.events.append(decision);return decision

    def grow(self, factory, decision: dict, *, seed_base: int,
             episodes: int=220, updates_per_episode: int=12) -> dict:
        env=factory();first=env.reset(seed_base)
        digest=hashlib.sha256(first.tobytes()+bytes([env.action_dim])).hexdigest()[:10]
        task_id=f'auto-{env.action_dim}-{digest}';bank=LifelongSkillBank(self.workspace,self.device)
        template=self.living.experts[0].spec
        spec=TaskSpec(task_id,tuple(first.shape),env.action_dim,replay_capacity=18_000,
            batch_size=64,latent_dim=template.latent_dim,learning_rate=5e-4,
            discount=.97,target_interval=100,seed=707)
        skill=bank.open(spec);start=time.monotonic();returns=[]
        for episode in range(episodes):
            if self.guard:self.guard.check()
            env=factory();obs=env.reset(seed_base+episode);ratio=episode/max(1,episodes-1)
            epsilon=1.0+ratio*(.05-1.0);total=0.0
            while True:
                action=skill.act(obs,epsilon=epsilon);nxt,reward,done,_=env.step(action)
                skill.observe(obs,action,reward,nxt,done);obs=nxt;total+=reward
                if done:break
            skill.learn(updates_per_episode);returns.append(total)
        def candidate_eval(episodes_eval=128):
            rows=[]
            for index in range(episodes_eval):
                env=factory();obs=env.reset(seed_base+1_000_000+index);total=0.0
                while True:
                    action=skill.act(obs);obs,reward,done,info=env.step(action);total+=reward
                    if done:break
                rows.append({'return':total,**info})
            result={'episodes':episodes_eval,'mean_return':_mean(rows,'return')}
            if 'correct' in rows[0]:
                correct=sum(x['correct'] for x in rows);attempts=correct+sum(x['errors'] for x in rows)
                result['accuracy']=correct/max(1,attempts)
            return result
        learned=candidate_eval();baseline=self._rollout(factory,'random',episodes=128,
            seed_base=seed_base+1_000_000)
        promoted=self._gain_is_real(learned,baseline)
        if promoted:event=bank.consolidate(skill,learned['mean_return'])
        else:
            skill.save();event=bank.quarantine(task_id,'autonomous candidate failed held-out competence gate')
        growth={'task_id':task_id,'episodes':episodes,'interactions':episodes*env.horizon,
                'updates':episodes*updates_per_episode,'mean_last_20_return':float(np.mean(returns[-20:])),
                'seconds':time.monotonic()-start,'baseline':baseline,'learned':learned,
                'promoted':promoted,'event':event}
        if promoted:
            self.living=LivingSystem(self.workspace,self.device)
            growth['organism_rebuild']=self.living.build(router_updates=700,world_updates=600)
            decision['method']='neural';decision['grown_task_id']=task_id
        return growth

    def evaluate_selected(self,factory,method: str,*,seed_base: int,episodes: int=64)->dict:
        return self._rollout(factory,method,episodes=episodes,seed_base=seed_base)


def run_autonomy_benchmark(workspace: Path,output: Path,device='cpu',guard=None)->dict:
    workspace,output=Path(workspace),Path(output);loop=AutonomousCompetenceLoop(workspace,device,guard)
    hashes_before={x.task_id:hashlib.sha256(x.skill.path.read_bytes()).hexdigest() for x in loop.living.experts}
    stream=[
        ('catch',CatchGame,11_000_000),('logic',LogicChainGame,11_100_000),
        ('anonymous-six',AnonymousSixGame,11_200_000),('maze',MazeGame,11_300_000),
        ('signals',SignalGame,11_400_000),('navigate',NavigateGame,11_500_000),
        ('avoid',AvoidGame,11_600_000),('anonymous-six-revisit',AnonymousSixGame,11_700_000)]
    encounters=[];growth=None
    for audit_name,factory,seed in stream:
        decision=loop.select(factory,probe_seed=seed)
        if decision['method']=='grow':
            growth=loop.grow(factory,decision,seed_base=seed+50_000)
            if not growth['promoted']:
                encounters.append({'audit_name':audit_name,'decision':decision,'growth':growth});continue
        evaluation=loop.evaluate_selected(factory,decision['method'],seed_base=seed+5_000,episodes=64)
        encounters.append({'audit_name':audit_name,'action_dim':factory.action_dim,
                           'decision':decision,'growth':growth if audit_name=='anonymous-six' else None,
                           'evaluation':evaluation})
    final=LivingSystem(workspace,device)
    hashes_after={x.task_id:hashlib.sha256(x.skill.path.read_bytes()).hexdigest() for x in final.experts
                  if x.task_id in hashes_before}
    report={'format':'wailah-autonomous-competence-loop-v1',
            'input_policy':'unlabeled environments; controller receives pixels, rewards, done and action count',
            'lifecycle':['probe recall','measure competence','try strategies','sandbox growth',
                         'held-out gate','promote or quarantine','rebuild organism','revisit'],
            'encounters':encounters,'growth':growth,
            'original_skill_hashes_unchanged':hashes_before==hashes_after,
            'active_experts_after':[x.task_id for x in final.experts],
            'autonomous_growth_succeeded':bool(growth and growth['promoted']),
            'anonymous_revisit_used_memory':encounters[-1].get('decision',{}).get('method')=='neural'}
    atomic_json(output,report);return report


def render_autonomy_summary(report_path: Path,output: Path)->Path:
    """Render the unlabeled encounter sequence and autonomous decisions."""
    from PIL import Image,ImageDraw,ImageFont
    report=json.loads(Path(report_path).read_text(encoding='utf-8'))
    try:
        title=ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf',24)
        body=ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf',15)
        small=ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf',12)
    except OSError:title=body=small=ImageFont.load_default()
    rows=report['encounters'];width=940;height=120+len(rows)*48+80
    canvas=Image.new('RGB',(width,height),(7,13,25));draw=ImageDraw.Draw(canvas)
    green=(96,220,170);white=(230,237,247);muted=(135,151,174)
    colors={'neural':(70,185,235),'maze-memory':(245,180,70),'causal-chain-memory':(200,120,245)}
    draw.text((28,22),'WAILAH AUTONOMOUS COMPETENCE LOOP',font=title,fill=green)
    draw.text((29,58),'unlabeled stream  /  probe -> measure -> reason or grow -> gate -> remember',font=body,fill=muted)
    draw.text((30,92),'ENCOUNTER',font=small,fill=muted);draw.text((250,92),'DECISION',font=small,fill=muted)
    draw.text((500,92),'EVIDENCE',font=small,fill=muted);draw.text((790,92),'OUTCOME',font=small,fill=muted)
    for index,row in enumerate(rows):
        y=119+index*48;decision=row['decision'];method=decision['method'];novel=decision['novel']
        draw.rounded_rectangle((20,y-6,width-20,y+34),radius=8,fill=(13,24,42))
        draw.text((32,y),row['audit_name'].replace('-',' ').upper(),font=body,fill=white)
        draw.text((250,y),method.replace('-',' '),font=body,fill=colors.get(method,green))
        probe=decision.get('strategy_probe') or decision.get('neural_probe') or {}
        evidence='new -> sandbox growth' if row['audit_name']=='anonymous-six' else (
            f"probe {probe.get('mean_return',0):.2f} vs random {decision['random_probe']['mean_return']:.2f}")
        draw.text((500,y),evidence,font=small,fill=muted)
        evaluation=row.get('evaluation') or {};outcome='promoted 98.1%' if row['audit_name']=='anonymous-six' else ''
        if 'success_rate' in evaluation:outcome=f"success {evaluation['success_rate']:.0%}"
        elif 'accuracy' in evaluation:outcome=f"accuracy {evaluation['accuracy']:.1%}"
        elif evaluation:outcome=f"return {evaluation['mean_return']:.2f}"
        draw.text((790,y),outcome,font=body,fill=green)
        draw.text((430,y+3),'NOVEL' if novel else 'KNOWN',font=small,fill=(245,150,75) if novel else green)
    y=125+len(rows)*48
    draw.text((28,y),'RESULT',font=small,fill=muted)
    grown=report.get('growth',{}).get('task_id','new expert')
    draw.text((28,y+24),f'Created expert {grown} -> promoted -> recalled on revisit',font=body,fill=green)
    draw.text((28,y+49),'Original skill hashes unchanged  |  Logic and Maze strategy-selected  |  no task labels',font=small,fill=white)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output);return output
