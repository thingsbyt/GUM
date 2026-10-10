"""Recover the observed excluded prefix from frozen weights/RNG and state hash.

This is explicitly reconstructed recovery evidence, not new training, a
completed episode, or a primary comparison sample.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from gum.school.escape_team import EscapeTeam
from gum.school.escape_chamber import DEVELOPMENT_ADAPTER,make_escape_chamber
from gum.storage import atomic_write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument("study",type=Path);args=parser.parse_args();root=args.study.resolve()
    torch.set_num_threads(1)
    handoff=json.loads((root/"EXECUTION_HANDOFF.json").read_text())
    state=handoff["phase_at_stop"]
    team=EscapeTeam.load(root/"team-1/ppo")
    if team.training_episodes or team.evaluation_episodes: raise RuntimeError("duplicate must be untouched birth state")
    seed=json.loads((root/"PROTOCOL.json").read_text())["evaluation_seed_bases"][0]+state["episode"]-1
    world=make_escape_chamber(DEVELOPMENT_ADAPTER,seed=seed,horizon=80,reward_table=team.reward_table)
    observations=world.reset();spec=world.public_spec()
    for member,observation in zip(team.members,observations): member.learner.begin(spec,observation,training=False)
    names=("observations","next_observations","actions","rewards","terminated","truncated","active_before","active_after","state_hashes")
    rows={name:[] for name in names}
    for _ in range(state["tick"]):
        actions=[None if world.escaped[i] else m.learner.act(observations[i],training=False) for i,m in enumerate(team.members)]
        step=world.step(actions)
        values=(np.stack(observations),np.stack(step.observations),[-1 if a is None else a for a in actions],step.rewards,
                step.terminated,step.truncated,step.active_before,step.active_after,world.audit_state()["state_sha256"].removeprefix("sha256:"))
        for name,value in zip(names,values): rows[name].append(value)
        for i,m in enumerate(team.members):
            if step.active_before[i]: m.learner.observe(actions[i],step.transition_for(i),training=False)
        observations=step.observations
    if world.audit_state()["state_sha256"].removeprefix("sha256:") != state["state_sha256"]:
        raise RuntimeError("recovered prefix does not match observed authoritative hash")
    archive=team._archive_episode(state["episode_id"],seed=seed,training=False,environment_adapter=DEVELOPMENT_ADAPTER,**rows)
    atomic_write_json(root/"RECOVERED_DUPLICATE_PREFIX.json",{"format":"gum-excluded-evaluation-prefix-recovery-v1",
        "reconstructed_from_frozen_birth_checkpoint":True,"matches_observed_authoritative_state_hash":True,
        "observed_joint_ticks":state["tick"],"observed_individual_actions":int(sum(sum(a>=0 for a in row) for row in rows["actions"])),
        "not_a_completed_episode":True,"excluded_from_primary_results":True,"archive":archive,
        "caveat":"Covers the last observed authoritative prefix; any subsequent policy inference before process suspension is not reconstructed."},backup=False)
    print("Excluded observed prefix recovered and verified by authoritative state hash.")

if __name__=="__main__":main()
