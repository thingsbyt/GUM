"""Stop only the original serial worker after its completed registered case."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import urlopen
import psutil
from gum.school.escape_ppo_study import outcome_summary, learning_digest
from gum.school.escape_team import EscapeTeam
from gum.storage import atomic_write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument("root",type=Path);parser.add_argument("pid",type=int);args=parser.parse_args()
    root=args.root.resolve();process=psutil.Process(args.pid)
    command=process.cmdline()
    if not any(arg.endswith("run_escape_ppo_study.py") for arg in command) or "--only-team" in command:
        raise ValueError("target is not original serial study worker")
    url=json.loads((root.parent/(root.name+"-viewer.json")).read_text())["url"].replace('/?','/state?')
    while True:
        with urlopen(url,timeout=10) as response: state=json.load(response)
        if state["method"]=="PPO":
            if state["checkpoint"]!=0 or state["phase"]!="frozen evaluation":
                raise RuntimeError("refuse to interrupt duplicate training")
            raw=json.loads((root/"RESULTS_IN_PROGRESS.json").read_text())
            row=raw["current"]
            if row["team"]!=1 or row["method"]!="gum" or len(row["training"])!=256 or len(row["evaluations"]["256"]["episodes"])!=32:
                raise RuntimeError("first case is not complete")
            team=EscapeTeam.load(root/row["root"])
            if team.training_episodes!=256 or team.evaluation_episodes!=128:
                raise RuntimeError("first case checkpoint disagrees")
            row["training_summary"]=outcome_summary(row["training"])
            row["resources"]["optimizer_steps"]=1024
            row["resources"]["optimization_epochs"]=1024
            row["resources"]["peak_cuda_allocated_bytes"]=0
            row["recovery_verified"]=True
            atomic_write_json(root/"FIRST_CASE_COMPLETED.json",{"team":row,"learning_digest":learning_digest(team)},backup=False)
            # No primary training is active; the duplicate evaluation prefix
            # remains archived exactly as completed, with no selected outcomes.
            cpu=process.cpu_times()
            process.terminate();process.wait(timeout=20)
            atomic_write_json(root/"EXECUTION_HANDOFF.json",{"stopped_process_id":args.pid,
                "phase_at_stop":state,"primary_training_interrupted":False,"duplicate_training_performed":False,
                "original_process_cpu_seconds":cpu.user+cpu.system,
                "uncompleted_duplicate_evaluation_ticks":state["tick"]},backup=False)
            print("Original GUM case complete; serial duplicate evaluation stopped safely.",flush=True)
            return
        time.sleep(.25)

if __name__=="__main__":main()
