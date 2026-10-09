"""Command line entry point. Model retrieval is explicit; compute is local."""
from __future__ import annotations
import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from .config import Config
from .runtime import ROOT,Guard,Stopped,device_for,privacy_environment


def parser():
    p=argparse.ArgumentParser(description='WAILAH: local self-learning vision and control from pixels.')
    p.add_argument('--config',type=Path,default=ROOT/'configs'/'asteroids_local.json')
    p.add_argument('--workspace',type=Path,default=ROOT/'runs'/'local')
    p.add_argument('--keep-stop',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--device',default=None,help='auto, cpu, or cuda:0; does not change the model')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor',help='Check packages, encoder location and hardware without loading large weights')
    sub.add_parser('download',help='Legacy research reference: download DINOv3')
    sub.add_parser('verify',help='Legacy research reference: verify DINOv3')
    sub.add_parser('collect',help='Legacy research reference: record its staged dataset')
    sub.add_parser('encode',help='Legacy research reference: cache DINOv3 features')
    tr=sub.add_parser('train',help='Legacy research reference: train AD-E2E-JEPA')
    tr.add_argument('--max-steps',type=int,default=None,help='Bound this invocation, then save and pause')
    ev=sub.add_parser('evaluate',help='Run held-out future-image goal trials and animate actual execution')
    ev.add_argument('--trials',type=int,default=8)
    ev.add_argument('--replay-delay',type=float,default=.1)
    pl=sub.add_parser('play',help='Receding-horizon control toward a user-supplied game screenshot')
    pl.add_argument('--goal',type=Path,required=True);pl.add_argument('--seed',type=int,default=3000000007)
    pl.add_argument('--decisions',type=int,default=100)
    learn=sub.add_parser('learn',help='Learn stable visual control from pixels and real returns; no pretrained model')
    learn.add_argument('--episodes',type=int,default=1)
    learn.add_argument('--fresh',action='store_true',help='Refuse any existing brain/replay in this workspace')
    watch=sub.add_parser('watch',help='Run a saved self-learning brain without storing experience or updating weights')
    watch.add_argument('--episodes',type=int,default=1)
    watch.add_argument('--replay-delay',type=float,default=.03)
    sev=sub.add_parser('self-evaluate',help='Compare random, constant, untrained, trained and corrupted-action policies')
    sev.add_argument('--episodes',type=int,default=12)
    sev.add_argument('--seed-offset',type=int,default=0,help='Offset for a separate untouched evaluation seed set')
    rt=sub.add_parser('replay-train',help='Train from saved real replay without collecting or imagining transitions')
    rt.add_argument('--updates',type=int,required=True)
    mp=sub.add_parser('learn-prior',help='Learn a robust action prior from episode outcomes only')
    mp.add_argument('--generations',type=int,default=12);mp.add_argument('--population',type=int,default=20)
    mp.add_argument('--episodes',type=int,default=6);mp.add_argument('--checkpoint',default='')
    me=sub.add_parser('master-evaluate',help='Evaluate the learned prior plus JEPA visual controller')
    me.add_argument('--episodes',type=int,default=64);me.add_argument('--seed-offset',type=int,default=100000)
    vc=sub.add_parser('vision-curriculum',help='Withdraw the learned action prior while training visual control')
    vc.add_argument('--episodes',type=int,default=192)
    vc.add_argument('--start-prior-weight',type=float,default=.9)
    vc.add_argument('--end-prior-weight',type=float,default=0.0)
    vcal=sub.add_parser('vision-calibrate',help='Freeze a visual-only action rule on development seeds')
    vcal.add_argument('--episodes',type=int,default=24)
    ve=sub.add_parser('vision-evaluate',help='Run frozen visual-only, occlusion and shuffled-frame tests')
    ve.add_argument('--episodes',type=int,default=64);ve.add_argument('--seed-offset',type=int,default=400000)
    ve.add_argument('--label',default='untouched')
    sub.add_parser('lifelong-status',help='Show the persistent multi-task skill registry')
    sub.add_parser('lifelong-proof',help='Run the deterministic A-B-A retention and revisit benchmark')
    lb=sub.add_parser('lifelong-benchmark',help='Train Catch-Avoid-Navigate-Catch from pixels')
    lb.add_argument('--scale',type=int,default=1,help='Positive multiplier for benchmark training')
    living=sub.add_parser('living-benchmark',help='Build shared memory and test unlabeled skill composition')
    living.add_argument('--router-updates',type=int,default=600)
    living.add_argument('--world-updates',type=int,default=900)
    kd=sub.add_parser('learn-keydoor',help='Grow a two-stage KeyDoor skill and verify old memories')
    kd.add_argument('--episodes',type=int,default=420)
    kd.add_argument('--updates-per-episode',type=int,default=14)
    sg=sub.add_parser('learn-signals',help='Learn a noisy five-symbol action language from reward')
    sg.add_argument('--episodes',type=int,default=220)
    sg.add_argument('--updates-per-episode',type=int,default=12)
    mz=sub.add_parser('learn-maze',help='Learn to escape unseen procedural mazes from pixels')
    mz.add_argument('--episodes',type=int,default=360)
    mz.add_argument('--updates-per-episode',type=int,default=14)
    sub.add_parser('maze-memory-benchmark',help='Test online pixel-graph escape on unseen mazes')
    sub.add_parser('logic-benchmark',help='Test hidden sparse-reward causal event chains')
    auto=sub.add_parser('autonomy-benchmark',help='Run unlabeled recall-reason-grow-revisit stream')
    sub.add_parser('stress-benchmark',help='Run read-only adversarial autonomy stress tests')
    sub.add_parser('frontier-benchmark',help='Run repeated growth, changing-world, learned-selection and Asteroids audits')
    sub.add_parser('literacy-benchmark',help='Discover a keyboard, ground visual language, read and write')
    sub.add_parser('language-nursery-benchmark',help='Infer a compositional language and execute delayed instruction chains')
    sub.add_parser('grammar-discovery-benchmark',help='Discover word families, optional grammar and temporal plans')
    sub.add_parser('ontology-growth-benchmark',help='Detect missing concepts and grow a reusable relational ontology')
    sub.add_parser('teacher-language-benchmark',help='Ground ordinary typed words from visual demonstrations')
    sub.add_parser('procedure-benchmark',help='Learn and execute a named multi-stage procedure from demonstrations')
    sv=sub.add_parser('serve',help='Open the local dashboard')
    sv.add_argument('--port',type=int,default=8779);sv.add_argument('--no-browser',action='store_true')
    return p


def doctor(cfg:Config)->dict:
    import importlib.util
    import torch
    from .encoder import checkpoint_location
    try:location=str(checkpoint_location(cfg));exists=Path(location).exists()
    except (FileNotFoundError,ValueError) as exc:location=str(exc);exists=False
    return {'python':sys.version.split()[0],'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
        'profile':cfg.name,'self_learning':{'ready':True,'algorithm':'dueling Double DQN + n-step replay + DrQ shifts',
            'pretrained_model_required':False,'llm_required':False,
            'eye_resolution':[cfg.self_observation_height,cfg.self_observation_width],
            'frame_stack':cfg.self_frame_stack,'latent_dim':cfg.stable_latent_dim,
            'batch_size':cfg.stable_batch_size,'target_interval':cfg.stable_target_interval},
        'legacy_ad_e2e_jepa':{'active':False,'transformers_installed':importlib.util.find_spec('transformers') is not None,
            'dinov3_location':location,'dinov3_folder_found':exists},
        'message':'Default learn/watch is ready and uses no pretrained model. Legacy DINO status does not block it.'}


def main(argv=None)->int:
    args=parser().parse_args(argv);privacy_environment()
    cfg=Config.load(args.config);workspace=args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True,exist_ok=True)
    try:
        if args.command=='doctor':
            print(json.dumps(doctor(cfg),indent=2));return 0
        if args.command=='lifelong-status':
            from .lifelong import LifelongSkillBank
            print(json.dumps(LifelongSkillBank(workspace).status(),indent=2));return 0
        if args.command=='download':
            from .encoder import download_encoder,MODEL_URL
            print(f'Accept access conditions first: {MODEL_URL}')
            token=os.environ.get('HF_TOKEN')
            if token is None and sys.stdin.isatty():
                token=getpass.getpass('Hugging Face read token (hidden; Enter uses an existing local login): ').strip() or None
            result=download_encoder(cfg,token)
            print(json.dumps(result,indent=2));return 0
        if args.command=='serve':
            from .dashboard import serve
            serve(cfg,args.config.resolve(),workspace,args.port,not args.no_browser);return 0
        if not args.keep_stop and (workspace/'STOP').exists():(workspace/'STOP').unlink()
        device=device_for(cfg,args.device);guard=Guard(cfg,workspace,device);guard.check(force=True)
        if args.command=='verify':
            from .verification import verify_real_encoder
            print(json.dumps(verify_real_encoder(cfg,workspace,device,guard),indent=2))
        elif args.command=='collect':
            from .data import collect
            collect(cfg,workspace,guard)
        elif args.command=='encode':
            from .encoder import FrozenDINOv3
            from .data import encode_dataset
            encoder=FrozenDINOv3(cfg,device);guard.check(force=True)
            encode_dataset(cfg,workspace,encoder,guard)
        elif args.command=='train':
            from .training import train
            train(cfg,workspace,device,guard,max_steps=args.max_steps)
        elif args.command=='evaluate':
            from .evaluation import evaluate
            evaluate(cfg,workspace,device,guard,trials=args.trials,replay_delay=args.replay_delay)
        elif args.command=='play':
            from .encoder import FrozenDINOv3
            from .evaluation import play_goal
            encoder=FrozenDINOv3(cfg,device);guard.check(force=True)
            play_goal(cfg,workspace,device,guard,encoder,args.goal.resolve(),seed=args.seed,decisions=args.decisions)
        elif args.command in ('learn','watch'):
            if args.command=='watch' and (workspace/'mastery_policy.json').exists():
                from .mastery import watch_mastery
                result=watch_mastery(cfg,workspace,device,guard,episodes=args.episodes,
                    replay_delay=getattr(args,'replay_delay',0.0))
                print(json.dumps(result,indent=2));return 0
            from .stable_learning import run_session
            result=run_session(cfg,workspace,device,guard,
                episodes=args.episodes,fresh=getattr(args,'fresh',False),
                learning=args.command=='learn',replay_delay=getattr(args,'replay_delay',0.0))
            print(json.dumps(result,indent=2))
        elif args.command=='self-evaluate':
            from .stable_learning import evaluate
            print(json.dumps(evaluate(cfg,workspace,device,guard,episodes=args.episodes,
                                      seed_offset=args.seed_offset),indent=2))
        elif args.command=='replay-train':
            from .stable_learning import train_replay
            print(json.dumps(train_replay(cfg,workspace,device,guard,updates=args.updates),indent=2))
        elif args.command=='learn-prior':
            from .mastery import learn_prior
            print(json.dumps(learn_prior(cfg,workspace,guard,generations=args.generations,
                population=args.population,episodes=args.episodes,checkpoint=args.checkpoint),indent=2))
        elif args.command=='master-evaluate':
            from .mastery import evaluate_mastery
            print(json.dumps(evaluate_mastery(cfg,workspace,device,guard,episodes=args.episodes,
                                              seed_offset=args.seed_offset),indent=2))
        elif args.command=='vision-curriculum':
            from .mastery import train_vision_curriculum
            print(json.dumps(train_vision_curriculum(cfg,workspace,device,guard,
                episodes=args.episodes,start_prior_weight=args.start_prior_weight,
                end_prior_weight=args.end_prior_weight),indent=2))
        elif args.command=='vision-calibrate':
            from .mastery import calibrate_visual_policy
            print(json.dumps(calibrate_visual_policy(
                cfg,workspace,device,guard,episodes=args.episodes),indent=2))
        elif args.command=='vision-evaluate':
            from .mastery import evaluate_visual_policy
            print(json.dumps(evaluate_visual_policy(cfg,workspace,device,guard,
                episodes=args.episodes,seed_offset=args.seed_offset,label=args.label),indent=2))
        elif args.command=='lifelong-proof':
            from .lifelong import run_toy_continual_proof
            print(json.dumps(run_toy_continual_proof(workspace,device),indent=2))
        elif args.command=='lifelong-benchmark':
            from .lifelong_benchmark import run_real_benchmark
            print(json.dumps(run_real_benchmark(workspace,device,guard,scale=args.scale),indent=2))
        elif args.command=='living-benchmark':
            from .living_benchmark import run_living_benchmark
            output=workspace/'living_hybrid_benchmark.json'
            print(json.dumps(run_living_benchmark(workspace,output,device,guard,
                router_updates=args.router_updates,world_updates=args.world_updates),indent=2))
        elif args.command=='learn-keydoor':
            from .lifelong_benchmark import run_keydoor_extension
            output=workspace/'keydoor_extension.json'
            print(json.dumps(run_keydoor_extension(workspace,output,device,guard,
                episodes=args.episodes,updates_per_episode=args.updates_per_episode),indent=2))
        elif args.command=='learn-signals':
            from .lifelong_benchmark import run_signal_extension
            output=workspace/'signal_extension.json'
            print(json.dumps(run_signal_extension(workspace,output,device,guard,
                episodes=args.episodes,updates_per_episode=args.updates_per_episode),indent=2))
        elif args.command=='learn-maze':
            from .lifelong_benchmark import run_maze_extension
            output=workspace/'maze_extension.json'
            print(json.dumps(run_maze_extension(workspace,output,device,guard,
                episodes=args.episodes,updates_per_episode=args.updates_per_episode),indent=2))
        elif args.command=='maze-memory-benchmark':
            from .lifelong_benchmark import run_maze_memory_benchmark
            print(json.dumps(run_maze_memory_benchmark(workspace/'maze_memory_benchmark.json'),indent=2))
        elif args.command=='logic-benchmark':
            from .logic_benchmark import run_logic_benchmark
            print(json.dumps(run_logic_benchmark(workspace,workspace/'logic_benchmark.json'),indent=2))
        elif args.command=='autonomy-benchmark':
            from .autonomy import run_autonomy_benchmark
            print(json.dumps(run_autonomy_benchmark(workspace,workspace/'autonomy_benchmark.json',device,guard),indent=2))
        elif args.command=='stress-benchmark':
            from .stress import run_stress_benchmark
            print(json.dumps(run_stress_benchmark(workspace,workspace/'stress_benchmark.json',device),indent=2))
        elif args.command=='frontier-benchmark':
            from .frontier import run_frontier_benchmark
            print(json.dumps(run_frontier_benchmark(workspace,workspace/'frontier_benchmark.json',cfg,device,guard),indent=2))
        elif args.command=='literacy-benchmark':
            from .literacy import run_literacy_benchmark
            print(json.dumps(run_literacy_benchmark(workspace,workspace/'literacy_benchmark.json',device),indent=2))
        elif args.command=='language-nursery-benchmark':
            from .language_nursery import run_language_nursery_benchmark
            print(json.dumps(run_language_nursery_benchmark(
                workspace,workspace/'language_nursery_benchmark.json',device),indent=2))
        elif args.command=='grammar-discovery-benchmark':
            from .grammar_discovery import run_grammar_discovery_benchmark
            print(json.dumps(run_grammar_discovery_benchmark(
                workspace,workspace/'grammar_discovery_benchmark.json',device),indent=2))
        elif args.command=='ontology-growth-benchmark':
            from .ontology_growth import run_ontology_growth_benchmark
            print(json.dumps(run_ontology_growth_benchmark(
                workspace,workspace/'ontology_growth_benchmark.json',device),indent=2))
        elif args.command=='teacher-language-benchmark':
            from .teacher_language import run_teacher_language_benchmark
            print(json.dumps(run_teacher_language_benchmark(
                workspace,workspace/'teacher_language_benchmark.json'),indent=2))
        elif args.command=='procedure-benchmark':
            from .procedure_learning import run_procedure_benchmark
            print(json.dumps(run_procedure_benchmark(
                workspace,workspace/'procedure_benchmark.json'),indent=2))
        return 0
    except (Stopped,KeyboardInterrupt) as exc:
        from .data import progress
        progress(workspace,'stopped',message=str(exc) or 'Interrupted. Completed work was retained.')
        print(f'STOPPED: {exc}',file=sys.stderr,flush=True);return 2
    except Exception as exc:
        from .data import progress
        message=f'{type(exc).__name__}: {exc}'
        progress(workspace,'error',message=message)
        print(message,file=sys.stderr,flush=True)
        return 1

if __name__=='__main__':raise SystemExit(main())
