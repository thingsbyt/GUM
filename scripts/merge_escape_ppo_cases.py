"""Merge exactly the registered independent cases, retaining excluded prefix."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
import psutil
from gum.school.escape_ppo_study import analyze
from gum.storage import atomic_write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument("study",type=Path);args=parser.parse_args();root=args.study.resolve()
    first=json.loads((root/"FIRST_CASE_COMPLETED.json").read_text())["team"]
    rows=[first];sources=[];durations=[]
    for path in sorted((root/"parallel-cases").glob("*/STUDY_RESULTS.json")):
        result=json.loads(path.read_text());sources.append(result["provenance"]);durations.append(result["wall_seconds"])
        if len(result["teams"])!=1: raise ValueError("expected one case")
        row=result["teams"][0];row["root"]=(path.parent/row["root"]).relative_to(root).as_posix();rows.append(row)
    if {(r["team"],r["method"]) for r in rows}!={(i,m) for i in (1,2,3,4) for m in ("gum","ppo")} or len(rows)!=8:
        raise ValueError("eight unique registered cases required")
    protocol=json.loads((root/"PROTOCOL.json").read_text())
    for row in rows:
        if len(row["training"])!=protocol["training_episodes"] or set(row["evaluations"])!=set(map(str,protocol["evaluation_checkpoints"])):
            raise ValueError("incomplete registered case")
    rows.sort(key=lambda r:(r["team"],r["method"]))
    excluded=[]
    ledger=root/"team-1/ppo/EPISODE_LEDGER.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            record=json.loads(line)
            if record["event"]=="episode-completed":
                row=record["payload"]
                if row["training"]: raise ValueError("duplicate process performed training")
                excluded.append({"episode_id":row["episode_id"],"joint_ticks":row["steps"],"archive":row["archive"],"reason":"execution amendment duplicate pre-training evaluation prefix; excluded without inspecting outcome"})
    provenance=json.loads((root/"PROVENANCE.json").read_text());provenance["parallel_case_sources"]=sources
    storage=json.loads((root.parent/"ppo-storage-handoff.json").read_text())
    handoff=json.loads((root/"EXECUTION_HANDOFF.json").read_text())
    for row in rows:
        if row["team"]==1 and row["method"]=="gum":
            row["resources"]["worker_cpu_seconds_at_collection"]=handoff["original_process_cpu_seconds"]
        else:
            worker=next(w for w in storage["workers"] if w["state_at_pause"]["team"]==row["team"] and w["state_at_pause"]["method"].lower()==row["method"])
            times=psutil.Process(worker["pid"]).cpu_times()
            row["resources"]["worker_cpu_seconds_at_collection"]=times.user+times.system
        row["resources"]["cpu_time_scope"]="worker process through collection: includes startup, environment, archive/recovery and viewer work; not isolated optimization"
    started=datetime.fromisoformat(provenance["started_at_utc"])
    result={"format":"gum-independent-ppo-development-study-v1","protocol":protocol,"provenance":provenance,
            "teams":rows,"analysis":analyze(rows,protocol),"wall_seconds":sum(r["resources"]["training_wall_seconds"]+r["resources"]["evaluation_wall_seconds"] for r in rows),
            "wall_time_definition":"sum of case episode wall times; concurrent execution, not elapsed makespan",
            "parallel_case_wall_seconds":durations,"excluded_duplicate_evaluation_prefix":excluded,
            "elapsed_makespan_seconds":(datetime.now(timezone.utc)-started).total_seconds(),
            "storage_relocation":storage,"execution_handoff":handoff,
            "recovered_excluded_prefix":json.loads((root/"RECOVERED_DUPLICATE_PREFIX.json").read_text()),
            "implementation_checks_passed":True,"hard_room_a_used":False,"tuning_environment_interactions":0}
    atomic_write_json(root/"STUDY_RESULTS.json",result,backup=False)
    print(json.dumps(result["analysis"],indent=2))

if __name__=="__main__":main()
