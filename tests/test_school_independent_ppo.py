import json
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from gum.protocol import PublicWorldSpec, Transition
from gum.school.independent_ppo import PPOConfig, IndependentPPOLearner, ElapsedRewardGUMLearner, clipped_actor_loss, generalized_advantages
from gum.school.recurrent_meta import RecurrentMetaConfig
from gum.school.escape_ppo_study import create_baseline_team, learning_digest
from gum.school.escape_team import EscapeTeam
from gum.school.escape_chamber import DEVELOPMENT_ADAPTER

pytestmark = pytest.mark.neural

def spec():
    return PublicWorldSpec("check-1","check","check",1,"pixels",(12,12,3),"discrete",5,8,(0.,4.))

def learner(seed=42,batch=2):
    return IndependentPPOLearner(seed,config=PPOConfig(pooled_size=4,visual_width=8,hidden_size=8,batch_episodes=batch,minibatch_episodes=2))

def episode(agent,length=3,training=True):
    pixels=np.zeros((12,12,3),np.uint8)
    agent.begin(spec(),pixels,training=training)
    for tick in range(length):
        action=agent.act(pixels,training=training)
        agent.observe(action,Transition(pixels,float(action==2),tick==length-1,False,{"hidden": object()}),training=training)
    return agent.finish_episode(training=training)

def test_clipped_surrogate_both_advantage_signs():
    logp=torch.tensor([np.log(1.5),np.log(.5),np.log(1.5),np.log(.5)],requires_grad=True)
    old=torch.zeros(4,requires_grad=True)
    advantage=torch.tensor([1.,1.,-1.,-1.])
    loss=clipped_actor_loss(logp,old,advantage,.2)
    assert float(loss.detach()) == pytest.approx(-((1.2+.5-1.5-.8)/4))
    loss.backward()
    assert old.grad is None
    assert logp.grad[0]==0 and logp.grad[3]==0

def test_gae_value_targets_and_no_boundary_leak():
    a,target=generalized_advantages([0,0,1],[.2,.3,.4],.9,1.)
    assert target==pytest.approx([.81,.9,1.])
    assert a==pytest.approx(np.array([.81,.9,1.])-[.2,.3,.4])
    _,other=generalized_advantages([3],[1],.9,.95)
    assert other==pytest.approx([3])
    a,_=generalized_advantages([0,1],[.5,.2],.9,0.)
    assert a==pytest.approx([-.32,.8])

@pytest.mark.parametrize("kind",["ppo","gum"])
def test_sequential_escapes_credit_elapsed_joint_ticks(kind):
    gamma=.9
    members=[learner(i,batch=99) if kind=="ppo" else ElapsedRewardGUMLearner(i,config=RecurrentMetaConfig(action_count=5,discount=gamma,batch_episodes=99)) for i in range(4)]
    if kind=="ppo":
        members=[IndependentPPOLearner(i,config=PPOConfig(discount=gamma,batch_episodes=99)) for i in range(4)]
    pixels=np.zeros((12,12,3),np.uint8)
    for m in members: m.begin(spec(),pixels,training=True)
    # Members escape at ticks 1,3,5. Inactive tick 2/4 has zero reward.
    # Final tick 5 pays escape plus completion; the holder remains active.
    escape_ticks=[1,3,5,99]
    rewards=[1.,0.,1.,0.,2.]
    for tick,reward in enumerate(rewards,1):
        for i,m in enumerate(members):
            if tick<=escape_ticks[i]:
                action=m.act(pixels,training=True)
                m.observe(action,Transition(pixels,reward,tick==5,False,{}),training=True)
            else: m.credit_delayed_reward(reward,training=True)
    expected=[1+gamma**2+2*gamma**4,1+2*gamma**2,2,2]
    for m,value in zip(members,expected):
        recorded=m.records[-1]["reward"] if kind=="ppo" else m._rewards[-1]
        assert recorded==pytest.approx(value)
    assert len(members[0].records if kind=="ppo" else members[0]._rewards)==1

def test_whole_sequence_padding_old_logps_and_updates():
    torch.set_num_threads(1)
    agent=learner(batch=99)
    episode(agent,2);episode(agent,4)
    old=[r["old_logp"].clone() for r in agent.pending]
    batch=agent.batch(agent.pending)
    logits,_,_=agent.policy(batch["features"],batch["previous_action"],batch["previous_reward"],batch["boundary"])
    short=agent.batch([agent.pending[0]])
    alone,_,_=agent.policy(short["features"],short["previous_action"],short["previous_reward"],short["boundary"])
    assert torch.allclose(logits[0,:2],alone[0],atol=1e-7)
    assert batch["mask"].sum()==6
    before={k:v.clone() for k,v in agent.policy.state_dict().items()}
    update=agent.update()
    assert update["old_logp_fixed"] and update["old_logp_reconstruction_max_error"]<2e-5
    assert agent.optimizer_steps>0
    assert any(not torch.equal(v,before[k]) for k,v in agent.policy.state_dict().items())
    assert [len(v) for v in old]==[2,4]

def test_executed_action_and_mode_checks():
    m=learner();pixels=np.zeros((12,12,3),np.uint8);m.begin(spec(),pixels,training=True)
    with pytest.raises(ValueError): m.act(pixels,training=False)
    action=m.act(pixels,training=True)
    with pytest.raises(ValueError): m.observe((action+1)%5,Transition(pixels,0,False,False,{}),training=True)

def test_checkpoint_optimizer_pending_rng_and_frozen_evaluation(tmp_path):
    torch.set_num_threads(1)
    m=learner();episode(m);episode(m);episode(m)
    path=tmp_path/"brain.pt";m.save(path);restored=IndependentPPOLearner.load(path)
    assert len(restored.pending)==1 and restored.optimizer_steps==m.optimizer_steps
    episode(m);episode(restored)
    for name,value in m.policy.state_dict().items(): assert torch.equal(value,restored.policy.state_dict()[name])
    m.save(path);before=torch.load(path,weights_only=True)
    episode(m,training=False);m.save(path);after=torch.load(path,weights_only=True)
    for name,value in before["policy"].items(): assert torch.equal(value,after["policy"][name])
    assert torch.equal(before["training_rng"],after["training_rng"])
    assert torch.equal(before["update_rng"],after["update_rng"])

@pytest.mark.parametrize("method",["ppo","gum"])
def test_four_member_recovery_after_pointer_interruption(tmp_path,monkeypatch,method):
    torch.set_num_threads(1)
    team=create_baseline_team(tmp_path/method,method,(101,211,307,401))
    before=(team.root/"ESCAPE_TEAM.json").read_bytes()
    team.run_episode(seed=5,training=True,horizon=3,environment_adapter=DEVELOPMENT_ADAPTER)
    expected=learning_digest(team)
    (team.root/"ESCAPE_TEAM.json").write_bytes(before)
    restored=EscapeTeam.load(team.root)
    assert restored.completed_episodes==1 and learning_digest(restored)==expected
    assert json.loads((team.root/"ESCAPE_TEAM.json").read_text())["recovered_after_incomplete_pointer_update"]
    assert len({id(m.learner.optimizer) for m in restored.members})==4
