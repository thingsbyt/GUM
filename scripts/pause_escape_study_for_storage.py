"""Suspend only owned study workers inside live ticks, away from file writes."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import urlopen
import psutil
from gum.storage import atomic_write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument("root",type=Path);parser.add_argument("record",type=Path);parser.add_argument("--resume-and-verify",action="store_true");args=parser.parse_args()
    if args.resume_and_verify:
        from gum.school.escape_team import EscapeTeam
        import torch
        torch.set_num_threads(1)
        record=json.loads(args.record.read_text())
        try:
            roots=list(Path("D:/Codex-GUM-evidence/2026-10-10/independent-ppo-development-v1").rglob("EPISODE_LEDGER.jsonl"))
            if len(roots)!=8: raise RuntimeError("eight relocated artifact roots required")
            checks=[]
            for ledger in roots:
                team=EscapeTeam.load(ledger.parent)
                checks.append({"root":str(ledger.parent),"completed_episodes":team.completed_episodes,"archives":team.verify_archives()})
            record["relocation_checks"]=checks
            record["artifact_relocation_verified"]=True
            atomic_write_json(args.record,record,backup=False)
        finally:
            for row in record["workers"]: psutil.Process(row["pid"]).resume()
        print("All relocated archive/checkpoint hashes valid; same eight live processes resumed.",flush=True)
        return
    root=args.root.resolve();paused=[]
    try:
        for process in psutil.process_iter(["pid","cmdline"]):
            command=process.info["cmdline"] or []
            if not any(a.endswith("run_escape_ppo_study.py") for a in command) or "--replay-only" in command: continue
            script_index=next(i for i,a in enumerate(command) if a.endswith("run_escape_ppo_study.py"))
            output=Path(command[script_index+1]).resolve()
            if output!=root and root not in output.parents: continue
            info=output.parent/(output.name+"-viewer.json")
            url=json.loads(info.read_text())["url"].replace('/?','/state?')
            while True:
                with urlopen(url,timeout=10) as response: state=json.load(response)
                if 10<=state["tick"]<=40:
                    process.suspend();paused.append({"pid":process.pid,"output":str(output),"state_at_pause":state})
                    break
                time.sleep(.03)
        if len(paused)!=8: raise RuntimeError(f"expected eight workers, got {len(paused)}")
        atomic_write_json(args.record,{"workers":paused,"paused_for_storage_only":True},backup=False)
        print(f"Suspended {len(paused)} owned workers at live ticks without replacing learners.",flush=True)
    except BaseException:
        for row in paused: psutil.Process(row["pid"]).resume()
        raise

if __name__=="__main__":main()
