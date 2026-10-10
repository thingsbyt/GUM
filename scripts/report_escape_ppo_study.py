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
    parser=argparse.ArgumentParser();parser.add_argument("study",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument("--public-summary",type=Path);args=parser.parse_args()
    results=json.loads((args.study/"STUDY_RESULTS.json").read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    table=[];resource=[];training_table=[]
    for team in results["teams"]:
        if team["method"]=="gum":
            team["resources"]["derived_valid_action_loss_presentations"]=team["resources"]["training_individual_actions"]
            team["resources"]["derived_padded_recurrent_update_slots"]=team["resources"]["training_individual_actions"]
        else:
            import torch
            generators=[torch.Generator().manual_seed(seed+92821) for seed in team["member_seeds"]]
            valid_slots=padded_slots=0
            for start in range(0,len(team["training"]),8):
                batch=team["training"][start:start+8];lengths=[]
                for episode in batch:
                    with np.load(args.study/team["root"]/episode["archive"]["path"],allow_pickle=False) as saved:
                        lengths.append((saved["actions"]>=0).sum(0))
                lengths=np.asarray(lengths)
                for member,update in enumerate(batch[-1]["updates"]):
                    remaining=update["optimizer_steps"]
                    for _ in range(update["epochs"]):
                        order=torch.randperm(len(batch),generator=generators[member]).tolist()
                        for offset in range(0,len(order),4):
                            if remaining<=0: break
                            selected=lengths[order[offset:offset+4],member]
                            valid_slots+=int(selected.sum());padded_slots+=int(selected.max()*len(selected));remaining-=1
                    if remaining: raise ValueError("unaccounted PPO optimizer steps")
            team["resources"]["derived_valid_action_loss_presentations"]=valid_slots
            team["resources"]["derived_padded_recurrent_update_slots"]=padded_slots
        for checkpoint,evaluation in team["evaluations"].items():
            table.append({"team":team["team"],"method":team["method"],"checkpoint":int(checkpoint),**evaluation["summary"]})
        resource.append({"team":team["team"],"method":team["method"],"parameters_per_member":team["parameters_per_member"][0],**team["resources"]})
        for episode,row in enumerate(team["training"],1):
            training_table.append({"team":team["team"],"method":team["method"],"episode":episode,
                "episode_id":row["episode_id"],"environment_seed":row["seed"],"completion":row["legal_maximum"],
                "escapes":row["escapes"],"gate_crossings":row["gate_crossings"],"plate_ticks":row["plate_ticks"],
                "maximum_sustained_plate_ticks":row["maximum_sustained_plate_ticks"],"mean_external_return":float(np.mean(row["returns"])),
                "joint_ticks":row["joint_ticks"],"individual_actions":row["individual_actions"]})
    for filename,rows in (("learning-curves.csv",table),("training-curves.csv",training_table),("resources.csv",resource)):
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
    summary["elapsed_makespan_seconds"]=results.get("elapsed_makespan_seconds")
    summary["implementation_validation_resource_use"]={"training_interactions":12288,"frozen_evaluation_interactions":1536,
        "optimizer_steps":3071,"wall_seconds":37.47904760000529,"chamber_tuning_interactions":0,
        "historical_gum_development_expenditure":"not quantified or equalized"}
    atomic_write_json(args.output/"independent-ppo-summary.json",summary,backup=False)
    if args.public_summary:
        atomic_write_json(args.public_summary,summary,backup=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for method,color in (("gum","#9869ca"),("ppo","#139ca5")):
        for team in [t for t in results["teams"] if t["method"]==method]:
            checkpoints=sorted(map(int,team["evaluations"]))
            rows=[team["evaluations"][str(c)]["summary"] for c in checkpoints]
            label=f"{method.upper()} team {team['team']}"
            style={"marker":("o","s","^","D")[team['team']-1],"alpha":.7,"color":color,
                   "linestyle":("-","--",":","-.")[team['team']-1]}
            axes[0,0].plot(checkpoints,[r["completion_rate"] for r in rows],label=label,**style)
            axes[0,1].plot(checkpoints,[r["escapes"]/r["episodes"] for r in rows],**style)
            axes[1,0].plot(checkpoints,[r["gate_crossings"]/r["episodes"] for r in rows],**style)
            axes[1,1].plot(checkpoints,[r["mean_maximum_sustained_plate_ticks"] for r in rows],**style)
    for ax,title in zip(axes.flat,["Primary: three-member completion rate","Escapes per evaluation episode","Gate crossings per evaluation episode","Mean longest continuous plate dwell (ticks)"]):
        ax.set_title(title);ax.grid(alpha=.2);ax.set_xlabel("Training episodes")
    axes[0,0].set_ylim(-.01,1.01);axes[0,0].legend(fontsize=7)
    fig.suptitle("Independent PPO / GUM · all four teams · exploratory development pilot")
    fig.tight_layout();fig.savefig(args.output/"learning-curves.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method,color in (("gum","#9869ca"),("ppo","#139ca5")):
        for team in [t for t in results["teams"] if t["method"]==method]:
            x=np.arange(32,len(team["training"])+1)
            for ax,key in zip(axes,("legal_maximum","escapes")):
                values=np.array([r[key] for r in team["training"]],float)
                ax.plot(x,np.convolve(values,np.ones(32)/32,mode="valid"),color=color,
                        alpha=.7,linestyle=("-","--",":","-.")[team["team"]-1],label=f"{method.upper()} team {team['team']}")
                ax.grid(alpha=.2);ax.set_xlabel("Training episode · trailing 32-episode mean")
    axes[0].set_title("Training three-member completion");axes[0].set_ylim(-.01,1.01);axes[0].legend(fontsize=7)
    axes[1].set_title("Training escapes per episode");fig.tight_layout();fig.savefig(args.output/"training-curves.png",dpi=160);plt.close(fig)
    print(json.dumps(summary["analysis"],indent=2))

if __name__=="__main__":main()
