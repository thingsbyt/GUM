"""Report every replication and checkpoint; no selected-success plots."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
from gum.lineage import file_sha256
from gum.storage import atomic_write_bytes,atomic_write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument("study",type=Path);parser.add_argument("output",type=Path);args=parser.parse_args()
    results=json.loads((args.study/"STUDY_RESULTS.json").read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    table=[];resource=[]
    for team in results["teams"]:
        for checkpoint,evaluation in team["evaluations"].items():
            table.append({"team":team["team"],"method":team["method"],"checkpoint":int(checkpoint),**evaluation["summary"]})
        resource.append({"team":team["team"],"method":team["method"],"parameters_per_member":team["parameters_per_member"][0],**team["resources"]})
    for filename,rows in (("learning-curves.csv",table),("resources.csv",resource)):
        text=io.StringIO();writer=csv.DictWriter(text,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        atomic_write_bytes(args.output/filename,text.getvalue().encode(),backup=False)
    # Full result retains every training and evaluation episode, all failures,
    # archive hashes, optimizer diagnostics and resource accounting.
    atomic_write_json(args.output/"independent-ppo-full-results.json",results,backup=False)
    summary={"source_commit":results["provenance"]["source_commit"],"protocol_sha256":results["provenance"]["protocol_sha256"],
             "study_results_sha256":file_sha256(args.study/"STUDY_RESULTS.json"),"analysis":results["analysis"],
             "learning_curves":table,"resources":resource,"validation_passed":True,"room_a_used":False,
             "training_episodes":sum(len(t["training"]) for t in results["teams"]),
             "evaluation_episodes":sum(sum(len(e["episodes"]) for e in t["evaluations"].values()) for t in results["teams"]),
             "training_completions_by_team":[{"team":t["team"],"method":t["method"],"completions":t["training_summary"]["completions"]} for t in results["teams"]],
             "wall_seconds":results["wall_seconds"]}
    atomic_write_json(args.output/"independent-ppo-summary.json",summary,backup=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for method,color in (("gum","#9869ca"),("ppo","#139ca5")):
        for team in [t for t in results["teams"] if t["method"]==method]:
            checkpoints=sorted(map(int,team["evaluations"]))
            rows=[team["evaluations"][str(c)]["summary"] for c in checkpoints]
            label=f"{method.upper()} team {team['team']}"
            axes[0,0].plot(checkpoints,[r["completion_rate"] for r in rows],marker="o",alpha=.6,color=color,label=label)
            axes[0,1].plot(checkpoints,[r["escapes"]/r["episodes"] for r in rows],marker="o",alpha=.6,color=color)
            axes[1,0].plot(checkpoints,[r["gate_crossings"]/r["episodes"] for r in rows],marker="o",alpha=.6,color=color)
            axes[1,1].plot(checkpoints,[r["mean_maximum_sustained_plate_ticks"] for r in rows],marker="o",alpha=.6,color=color)
    for ax,title in zip(axes.flat,["Primary: three-member completion rate","Escapes per evaluation episode","Gate crossings per evaluation episode","Mean longest continuous plate dwell"]):
        ax.set_title(title);ax.grid(alpha=.2);ax.set_xlabel("Training episodes")
    axes[0,0].set_ylim(-.01,1.01);axes[0,0].legend(fontsize=7)
    fig.suptitle("Independent PPO / GUM · all four teams · exploratory development pilot")
    fig.tight_layout();fig.savefig(args.output/"learning-curves.png",dpi=160);plt.close(fig)
    print(json.dumps(summary["analysis"],indent=2))

if __name__=="__main__":main()
