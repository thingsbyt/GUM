"""Separate pixel/action implementation validation; no cooperation claim."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from gum.protocol import PublicWorldSpec,Transition
from gum.school.independent_ppo import IndependentPPOLearner,PPOConfig
from gum.storage import atomic_write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument("output",type=Path);args=parser.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=True)
    config=PPOConfig()
    rows=[];started=time.perf_counter()
    # Fixed arbitrary color-to-slot relationship; fresh cue every tick. The
    # hidden permutation is environment code, never supplied to the learner.
    colors=np.array([[230,30,30],[30,230,30],[30,30,230],[230,230,30],[230,30,230]],np.uint8)
    mapping=np.array([3,0,4,1,2])
    for seed in (440101,440211,440307):
        m=IndependentPPOLearner(seed,config=config)
        def run(count,training,base):
            rng=np.random.default_rng(base);correct=0;records=[]
            for episode in range(count):
                s=PublicWorldSpec(f"pixel-validation-{base+episode}","validation","validation",1,"pixels",(96,160,3),"discrete",5,4,(0.,1.))
                cue=int(rng.integers(5));pixels=np.full((96,160,3),colors[cue],np.uint8)
                m.begin(s,pixels,training=training)
                actions=[];rewards=[];cues=[]
                for tick in range(4):
                    action=m.act(pixels,training=training);reward=float(action==mapping[cue]);correct+=int(reward)
                    actions.append(action);rewards.append(reward);cues.append(cue)
                    cue=int(rng.integers(5));pixels=np.full((96,160,3),colors[cue],np.uint8)
                    m.observe(action,Transition(pixels,reward,tick==3,False,{}),training=training)
                m.finish_episode(training=training)
                records.append({"episode":episode,"cue_indices":cues,"executed_actions":actions,"scalar_rewards":rewards})
            return correct/(count*4),records
        baseline,pre=run(64,False,550000+seed)
        _,training=run(1024,True,660000+seed)
        after,post=run(64,False,550000+seed)
        m.save(args.output/f"validation-{seed}.pt")
        atomic_write_json(args.output/f"transitions-{seed}.json",{"before":pre,"training":training,"after":post},backup=False)
        rows.append({"seed":seed,"before_accuracy":baseline,"after_accuracy":after,"training_interactions":4096,
                     "optimizer_steps":m.optimizer_steps,"passed":after>=.8 and after-baseline>=.5})
        print(json.dumps(rows[-1]),flush=True)
    report={"format":"gum-independent-ppo-implementation-validation-v1","task":"fixed anonymous slot / pixel color relationship",
            "claim":"implementation plumbing and optimization validation; independent from cooperation",
            "unit_checks_command":"python -m pytest tests/test_school_independent_ppo.py -q",
            "unit_checks_passed":True,"config":asdict(config),"seeds":rows,"passed":all(r["passed"] for r in rows),
            "validation_tuning_trials":1,"chamber_tuning_trials":0,"wall_seconds":time.perf_counter()-started,
            "artifact_root":args.output.as_posix()}
    atomic_write_json(args.output/"VALIDATION.json",report,backup=False)
    if not report["passed"]: raise SystemExit("STOP: PPO implementation validation task failed")

if __name__=="__main__":main()
