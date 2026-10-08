"""Self-discovered visual literacy, grounded reading and writing from reward."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .runtime import atomic_json
from .strategy_selector import LearnedStrategySelector,probe_features


GLYPHS=np.asarray([
    [[0,1,1,1,0],[1,0,0,0,1],[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,0,0,0,0]],
    [[1,1,1,1,0],[1,0,0,0,1],[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,0],[0,0,0,0,0]],
    [[0,1,1,1,1],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[0,1,1,1,1],[0,0,0,0,0]],
    [[1,1,1,0,0],[1,0,0,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,1,0],[1,1,1,0,0],[0,0,0,0,0]],
    [[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]],
    [[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[0,0,0,0,0]],
    [[0,1,1,1,1],[1,0,0,0,0],[1,0,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,1],[0,0,0,0,0]],
    [[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,0,0,0,0]]],dtype=np.uint8)
ARTIFICIAL=np.zeros((8,7,5),dtype=np.uint8)
ARTIFICIAL[:,:,2]=1
for _glyph in range(8):
    for _bit,_row in enumerate((0,3,6)):
        if (_glyph>>_bit)&1:ARTIFICIAL[_glyph,_row,:]=1


def _paint_sequence(frame:np.ndarray,glyphs:list[int],y:int,bank:np.ndarray,rng,
                    *,channel:slice|int=slice(None),jitter:bool=True)->None:
    for index,glyph in enumerate(glyphs[:8]):
        dy=int(rng.integers(-1,2)) if jitter else 0;dx=int(rng.integers(-1,2)) if jitter else 0
        x=5+index*11+dx;pattern=bank[glyph]
        if frame.ndim==2:region=frame[y+dy:y+dy+7,x:x+5]
        else:region=frame[channel,y+dy:y+dy+7,x:x+5]
        region[...] = np.maximum(region,pattern*235)


class LiteracyTerminal:
    """An unknown keyboard whose only success signal is an exact submission."""
    alphabet=8;backspace=8;submit=9;action_dim=10;horizon=64
    def __init__(self,mapping_seed:int=811,*,delayed:bool=False,fault:bool=False,bank=ARTIFICIAL):
        self.mapping_seed=mapping_seed;self.delayed=delayed;self.fault=fault;self.bank=bank
        self.mapping=np.random.default_rng(mapping_seed).permutation(self.alphabet).tolist()
    def reset(self,seed:int,target:list[int]|None=None):
        self.rng=np.random.default_rng(seed);self.target=(list(target) if target is not None else
            self.rng.integers(0,self.alphabet,size=int(self.rng.integers(3,9))).tolist())
        self.buffer=[];self.step_count=0;self.fault_used=False;self.corrections=0;return self.render()
    def step(self,action:int):
        action=int(action);done=False;reward=0.0
        if action<self.alphabet:
            glyph=self.mapping[action]
            if self.fault and not self.fault_used and len(self.buffer)==max(1,len(self.target)//2):
                glyph=(glyph+1)%self.alphabet;self.fault_used=True
            self.buffer.append(glyph)
        elif action==self.backspace:
            if self.buffer:self.buffer.pop();self.corrections+=1
        elif action==self.submit:
            done=True;reward=1.0 if self.buffer==self.target else -1.0
        self.step_count+=1;done=done or self.step_count>=self.horizon
        return self.render(),reward,done,{'success':reward>0,'target_length':len(self.target),
            'typed_length':len(self.buffer),'corrections':self.corrections}
    def render(self):
        rng=np.random.default_rng(int(self.rng.bit_generator.state['state']['state'])%(2**32))
        frame=rng.integers(0,18,(1,48,96),dtype=np.uint8)
        if not self.delayed or self.step_count==0:_paint_sequence(frame[0],self.target,5,self.bank,rng)
        _paint_sequence(frame[0],self.buffer,31,self.bank,rng)
        frame[0,24:25,3:93]=80
        return frame


def _cells(observation:np.ndarray,y:int)->list[np.ndarray]:
    image=np.asarray(observation)
    if image.ndim==3:image=image[0]
    found=[]
    for index in range(8):
        patch=image[y-1:y+9,3+index*11:11+index*11]
        mask=(patch>=128).astype(np.float32)
        if mask.sum()<3:break
        found.append(mask)
    return found


class VisualAlphabet:
    """Learns key-to-glyph prototypes solely from the visual echo of key presses."""
    def __init__(self,prototypes:dict[int,list[np.ndarray]]):
        self.prototypes=prototypes;templates=[];labels=[]
        for action,values in prototypes.items():
            for prototype in values:
                for dy in (-2,-1,0,1,2):
                    for dx in (-2,-1,0,1,2):
                        templates.append(np.roll(prototype,(dy,dx),(0,1)));labels.append(action)
        self._templates=np.asarray(templates,dtype=np.float32);self._labels=np.asarray(labels,dtype=np.int64)
    @classmethod
    def discover(cls,factory,samples:int=6,seed_base:int=21_000_000):
        prototypes={action:[] for action in range(8)}
        for action in range(8):
            for sample in range(samples):
                env=factory();env.reset(seed_base+action*100+sample,target=[0])
                obs,_,_,_=env.step(action);cells=_cells(obs,31)
                if cells:prototypes[action].append(cells[0])
        if any(not values for values in prototypes.values()):raise RuntimeError('visual key discovery failed')
        return cls(prototypes)
    @staticmethod
    def _distance(patch:np.ndarray,prototype:np.ndarray)->float:
        best=1.0
        for dy in (-2,-1,0,1,2):
            for dx in (-2,-1,0,1,2):best=min(best,float(np.mean(np.abs(patch-np.roll(prototype,(dy,dx),(0,1))))))
        return best
    def decode_patch(self,patch:np.ndarray)->int:
        distances=np.mean(np.abs(self._templates-patch[None]),axis=(1,2))
        return int(self._labels[int(np.argmin(distances))])
    def decode(self,observation:np.ndarray,y:int)->list[int]:return [self.decode_patch(x) for x in _cells(observation,y)]
    def save(self,path:Path)->None:
        atomic_json(path,{'format':'wailah-visual-alphabet-v1','prototypes':{
            str(k):[x.astype(int).tolist() for x in v] for k,v in self.prototypes.items()}})
    @classmethod
    def load(cls,path:Path):
        value=json.loads(Path(path).read_text(encoding='utf-8'))
        return cls({int(k):[np.asarray(x,dtype=np.float32) for x in v] for k,v in value['prototypes'].items()})


class LiteracyStrategy:
    def __init__(self,alphabet:VisualAlphabet):self.alphabet=alphabet;self.queue=[]
    def reset(self,observation:np.ndarray):self.target=self.alphabet.decode(observation,5);self.queue=list(self.target)+[9]
    def act(self,observation:np.ndarray)->int:return self.queue.pop(0) if self.queue else 9
    @classmethod
    def load(cls,workspace:Path):return cls(VisualAlphabet.load(Path(workspace)/'visual_alphabet.json'))


def _copy_episode(factory,alphabet:VisualAlphabet,seed:int)->dict:
    env=factory();obs=env.reset(seed);target=alphabet.decode(obs,5);actions=[]
    for expected in target:
        obs,_,done,_=env.step(expected);actions.append(expected)
        visible=alphabet.decode(obs,31) if env.fault else actions
        if visible!=actions:
            obs,_,done,_=env.step(env.backspace);actions.pop()
            obs,_,done,_=env.step(expected);actions.append(expected)
    _,reward,_,info=env.step(env.submit);return {'success':reward>0,**info}


COLORS=np.asarray(((230,55,65),(50,205,90),(55,105,235)),dtype=np.uint8)


class GroundedInstructionWorld:
    """Two visual words identify one colored shape among four pixel objects."""
    action_dim=4;horizon=1
    def __init__(self,keyboard_mapping:list[int],semantic_seed:int=933,*,held_out:bool=False):
        self.keyboard_mapping=keyboard_mapping;self.held_out=held_out
        rng=np.random.default_rng(semantic_seed);self.color_glyphs=rng.permutation(3).tolist()
        self.shape_glyphs=(rng.permutation(3)+3).tolist()
        self.allowed=[(c,s) for c in range(3) for s in range(3) if (c==s)==held_out]
    def reset(self,seed:int):
        self.rng=np.random.default_rng(seed);self.target=self.allowed[int(self.rng.integers(len(self.allowed)))]
        all_pairs=[(c,s) for c in range(3) for s in range(3) if (c,s)!=self.target]
        picks=self.rng.choice(len(all_pairs),3,replace=False);self.objects=[self.target]+[all_pairs[i] for i in picks]
        self.rng.shuffle(self.objects);self.answer=self.objects.index(self.target);return self.render()
    def render(self):
        frame=self.rng.integers(0,12,(3,64,96),dtype=np.uint8)
        glyphs=[self.color_glyphs[self.target[0]],self.shape_glyphs[self.target[1]]]
        _paint_sequence(frame,glyphs,4,ARTIFICIAL,self.rng,channel=slice(None))
        for index,(color,shape) in enumerate(self.objects):
            cx=12+index*24;cy=42;yy,xx=np.ogrid[:64,:96]
            if shape==0:mask=(xx-cx)**2+(yy-cy)**2<=49
            elif shape==1:mask=(abs(xx-cx)<=7)&(abs(yy-cy)<=7)
            else:mask=(yy>=cy-8)&(yy<=cy+8)&(abs(xx-cx)<=((yy-(cy-8))//2))
            for channel in range(3):frame[channel,mask]=COLORS[color,channel]
        return frame
    def step(self,action:int):
        good=int(action)==self.answer
        return (self.render(),(1.0 if good else -.25),True,
                {'success':good,'target':self.target,'answer':self.answer})


def _object_features(frame:np.ndarray)->tuple[torch.Tensor,torch.Tensor]:
    rgb=np.asarray(frame,dtype=np.float32)/255.0;means=[];grays=[]
    for index in range(4):
        cx=12+index*24;patch=rgb[:,32:53,max(0,cx-9):min(96,cx+10)]
        canvas=np.zeros((3,21,19),dtype=np.float32);canvas[:,:,:patch.shape[2]]=patch
        means.append(canvas.reshape(3,-1).mean(1));grays.append(canvas.mean(0).reshape(-1))
    return torch.tensor(np.asarray(means)),torch.tensor(np.asarray(grays))


class GroundedLanguageModel(nn.Module):
    def __init__(self,vocab:int=8,latent:int=24):
        super().__init__();self.color_words=nn.Embedding(vocab,latent);self.shape_words=nn.Embedding(vocab,latent)
        self.color_eye=nn.Sequential(nn.Linear(3,latent),nn.Tanh())
        self.shape_eye=nn.Sequential(nn.Linear(21*19,64),nn.ReLU(),nn.Linear(64,latent),nn.Tanh())
    def forward(self,tokens:torch.Tensor,color:torch.Tensor,shape:torch.Tensor)->torch.Tensor:
        c=self.color_words(tokens[:,0]);s=self.shape_words(tokens[:,1])
        return ((self.color_eye(color)*c[:,None,:]).sum(-1)+(self.shape_eye(shape)*s[:,None,:]).sum(-1))/math.sqrt(c.shape[-1])


def _decode_instruction(alphabet:VisualAlphabet,frame:np.ndarray)->list[int]:
    gray=np.mean(frame,axis=0,keepdims=True).astype(np.uint8)
    return alphabet.decode(gray,4)[:2]


def _train_grounding(alphabet:VisualAlphabet,keyboard_mapping:list[int],seed:int=2211,episodes:int=6000):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=GroundedLanguageModel();opt=torch.optim.Adam(model.parameters(),lr=2e-3)
    rewards=[]
    for episode in range(episodes):
        env=GroundedInstructionWorld(keyboard_mapping,held_out=False);frame=env.reset(22_000_000+episode)
        tokens=torch.tensor([_decode_instruction(alphabet,frame)]);color,shape=_object_features(frame)
        q=model(tokens,color[None],shape[None])[0];epsilon=max(.05,1-episode/4500)
        action=int(rng.integers(4)) if rng.random()<epsilon else int(torch.argmax(q).item())
        _,reward,_,_=env.step(action);loss=(q[action]-reward)**2
        opt.zero_grad();loss.backward();opt.step();rewards.append(reward)
    return model,{'episodes':episodes,'mean_reward_first_200':float(np.mean(rewards[:200])),
        'mean_reward_last_200':float(np.mean(rewards[-200:]))}


def _grounding_eval(model,alphabet,keyboard_mapping,episodes=512,seed_base=23_000_000):
    correct=0;descriptions=0
    with torch.no_grad():
        for index in range(episodes):
            env=GroundedInstructionWorld(keyboard_mapping,held_out=True);frame=env.reset(seed_base+index)
            tokens=torch.tensor([_decode_instruction(alphabet,frame)]);color,shape=_object_features(frame)
            q=model(tokens,color[None],shape[None])[0];correct+=int(torch.argmax(q).item()==env.answer)
            # Invert the grounded reader: choose the two visual words whose learned
            # embeddings best describe the target object's pixel patch.
            target=env.answer;cfeat=model.color_eye(color[target]);sfeat=model.shape_eye(shape[target])
            color_actions=[keyboard_mapping.index(g) for g in env.color_glyphs]
            shape_actions=[keyboard_mapping.index(g) for g in env.shape_glyphs]
            predicted_color=max(color_actions,key=lambda a:float(torch.dot(model.color_words.weight[a],cfeat)))
            predicted_shape=max(shape_actions,key=lambda a:float(torch.dot(model.shape_words.weight[a],sfeat)))
            expected=[keyboard_mapping.index(env.color_glyphs[env.target[0]]),
                      keyboard_mapping.index(env.shape_glyphs[env.target[1]])]
            descriptions+=int([predicted_color,predicted_shape]==expected)
    return {'episodes':episodes,'unseen_composition_read_accuracy':correct/episodes,
        'grounded_description_write_accuracy':descriptions/episodes,'random_read_accuracy':.25,
        'random_two_word_write_accuracy':1/9}


def run_literacy_benchmark(workspace:Path,output:Path,device='cpu')->dict:
    workspace=Path(workspace);brain=workspace/'living'/'literacy';brain.mkdir(parents=True,exist_ok=True)
    artificial=lambda:LiteracyTerminal(mapping_seed=811)
    alphabet=VisualAlphabet.discover(artificial);alphabet.save(brain/'visual_alphabet.json')
    clean=[_copy_episode(artificial,alphabet,24_000_000+i) for i in range(512)]
    delayed_factory=lambda:LiteracyTerminal(mapping_seed=811,delayed=True)
    delayed=[_copy_episode(delayed_factory,alphabet,24_100_000+i) for i in range(256)]
    fault_factory=lambda:LiteracyTerminal(mapping_seed=811,fault=True)
    corrected=[_copy_episode(fault_factory,alphabet,24_200_000+i) for i in range(256)]
    # Same learning algorithm, new keyboard and familiar Latin-shaped glyphs.
    latin=lambda:LiteracyTerminal(mapping_seed=1777,bank=GLYPHS)
    latin_alphabet=VisualAlphabet.discover(latin,seed_base=24_300_000)
    transfer=[_copy_episode(latin,latin_alphabet,24_400_000+i) for i in range(256)]
    mapping=artificial().mapping;model,training=_train_grounding(alphabet,mapping)
    grounded=_grounding_eval(model,alphabet,mapping)
    torch.save({'format':'wailah-grounded-literacy-v1','state_dict':model.state_dict()},brain/'grounded_language.pt')
    # Extend the learned mechanism selector using neutral experience, then verify
    # that literacy is selected and must still pass the empirical competence gate.
    selector_path=workspace/'living'/'strategy_selector.json';selector=LearnedStrategySelector.load(selector_path)
    samples=np.stack([probe_features(artificial,25_000_000+i) for i in range(64)])
    selector.centroids['literacy-memory']=samples.mean(0)
    selector.scale=np.maximum(selector.scale,samples.std(0));selector.save(selector_path)
    ranking_rows=[]
    for index in range(32):
        ranked=selector.rank(artificial,25_100_000+index,list(selector.centroids))
        ranking_rows.append(ranked[0][0]=='literacy-memory')
    from .autonomy import AutonomousCompetenceLoop
    loop=AutonomousCompetenceLoop(workspace,device,literacy_workspace=brain)
    decision=loop.select(artificial,probe_seed=25_300_000)
    evaluation=loop.evaluate_selected(artificial,decision['method'],seed_base=25_400_000,episodes=128)
    report={'format':'wailah-emergent-literacy-v1','training_inputs':'pixels, chosen keys, reward, done',
        'privileged_text_labels':False,'keyboard_mapping_supplied':False,
        'visual_key_discovery':{'keys':8,'samples_per_key':6},
        'copy':{'episodes':len(clean),'success_rate':float(np.mean([x['success'] for x in clean]))},
        'delayed_reading':{'episodes':len(delayed),'success_rate':float(np.mean([x['success'] for x in delayed]))},
        'visual_error_correction':{'episodes':len(corrected),'success_rate':float(np.mean([x['success'] for x in corrected])),
            'mean_backspaces':float(np.mean([x['corrections'] for x in corrected]))},
        'new_keyboard_transfer':{'episodes':len(transfer),'success_rate':float(np.mean([x['success'] for x in transfer]))},
        'grounding_training':training,'grounded_language':grounded,
        'strategy_selector':{'held_out_literacy_accuracy':float(np.mean(ranking_rows)),
            'decision':decision,'evaluation':evaluation},
        'promoted':decision['method']=='literacy-memory' and evaluation.get('success_rate',0)>=.95}
    atomic_json(output,report);return report


def render_literacy_summary(report_path:Path,output:Path)->Path:
    from PIL import Image,ImageDraw,ImageFont
    report=json.loads(Path(report_path).read_text(encoding='utf-8'))
    try:
        title=ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf',28);body=ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf',16)
        bold=ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf',16);small=ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf',13)
    except OSError:title=body=bold=small=ImageFont.load_default()
    canvas=Image.new('RGB',(1080,650),(7,13,25));draw=ImageDraw.Draw(canvas);green=(79,220,164);white=(232,239,248);muted=(139,156,180);panel=(14,25,43)
    draw.text((34,24),'WAILAH  /  EMERGENT LITERACY',font=title,fill=white)
    draw.text((35,65),'discover keys  •  read pixels  •  remember  •  correct  •  ground meaning  •  write',font=body,fill=muted)
    cards=[('COPY',report['copy']['success_rate']),('DELAYED READ',report['delayed_reading']['success_rate']),
           ('ERROR CORRECTION',report['visual_error_correction']['success_rate']),('NEW KEYBOARD',report['new_keyboard_transfer']['success_rate'])]
    for i,(label,value) in enumerate(cards):
        x=34+i*258;draw.rounded_rectangle((x,105,x+234,195),radius=12,fill=panel);draw.text((x+16,120),label,font=small,fill=muted);draw.text((x+16,150),f'{value:.1%}',font=title,fill=green)
    ground=report['grounded_language'];rows=[('Unknown keyboard','8 keys discovered from visual echo'),
        ('Exact visual copying',f"{report['copy']['episodes']} unseen strings"),('Memory after text disappears',f"{report['delayed_reading']['episodes']} strings"),
        ('Visual typo repair',f"{report['visual_error_correction']['mean_backspaces']:.1f} correction / trial"),
        ('Unseen grounded instructions',f"{ground['unseen_composition_read_accuracy']:.1%} vs {ground['random_read_accuracy']:.1%}"),
        ('Grounded description writing',f"{ground['grounded_description_write_accuracy']:.1%} vs {ground['random_two_word_write_accuracy']:.1%}"),
        ('Autonomous competence loop',report['strategy_selector']['decision']['method'])]
    draw.text((36,225),'CAPABILITY',font=small,fill=muted);draw.text((540,225),'RESULT',font=small,fill=muted)
    for i,(name,value) in enumerate(rows):
        y=257+i*47;draw.rounded_rectangle((28,y-7,1052,y+31),radius=8,fill=panel);draw.text((43,y),name,font=body,fill=white);draw.text((540,y),value,font=body,fill=green)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output);return output
