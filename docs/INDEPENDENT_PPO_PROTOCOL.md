# Independent PPO development pilot

The machine-readable authority is
`evidence/gum-school/studies/independent-ppo-v1/PROTOCOL.json`, committed before
any decision-relevant development training or evaluation. This is a four-team
exploratory feasibility pilot. Historical study `26af943` remains evidence;
its value-gradient correction is not active in this experiment.

Four separately initialized policies and Adam optimizers train from their own
public pixels, previous actions, rewards and recurrent history. Each method
gets the same development geometry, anonymous five-action interface, reward
table, environment schedules, 80-tick finite-horizon episodes and maximum
20,480 training joint ticks per team. Sharing and messaging are disabled.
Each independent team is one replication. Both methods are evaluated on 32
matched instances before training and at fixed episodes 64, 128 and 256.
The fixed final checkpoint determines the result; no best-checkpoint selection.

Primary outcome is the legal maximum of three escapes. A method passes the
pilot development criterion only if at least three of four teams achieve at
least 10% completion with a gain of at least 10 percentage points, and the
exploratory bootstrap lower bound on mean team gain exceeds zero. With 32
instances, the operational floor is four completions and a gain of four.
Every team, training episode and evaluation checkpoint is reported. Confidence
intervals resample whole teams. Four teams cannot establish population-level
reliability; degenerate zero intervals do not prove impossibility.

## Baseline and differences

This is an inspectable standard implementation of [PPO](https://arxiv.org/abs/1707.06347)
with [GAE](https://arxiv.org/abs/1506.02438). Independent PPO is distinct from
centralized or parameter-shared variants discussed in
[cooperative PPO research](https://arxiv.org/abs/2103.01955).
Full sequences and episode resets follow the recurrent concerns illustrated
by the [SB3 recurrent implementation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html).
The custom implementation fits the anonymous-action history, inactive reward
tail and existing complete-episode archive directly. It is not a port of SB3
and does not claim its benchmark performance.

PPO pools pixels to 16x16, learns a 64-unit visual layer and 64-unit GRU, and
uses ordinary categorical actor/value heads. GUM retains 8x8 pooling, a
32-unit visual layer, a 64-unit recurrent state, and per-action 16-unit memories.
PPO includes one-hot previous actions in its GRU input; GUM routes action
history through its per-action memories. PPO uses orthogonal initialization.

PPO uses gamma .99, lambda .95, clipping .2, learning rate .0003, entropy .01,
gradient clipping .5, Adam epsilon .00001, eight-episode rollouts and up to
four epochs in four-episode minibatches. Approximate KL above .045 stops the
update. Advantages, targets, old values and old log probabilities stay fixed.
Whole episodes are shuffled, hidden state is reconstructed from zero every
pass, and right padding is excluded from every objective and statistic.
GUM retains its original Monte Carlo actor-critic loss, gamma .97, learning
rate .001, entropy .02, gradient clipping 1 and one episode/update.

The GUM reference is the audited novelty-off/action-mixer-off configuration:
zero novelty reward, zero episodic mixer, with its declared 20% uniform
training mixture. PPO uses no mixture. Both use temperature 1 in frozen
evaluation; this differs from GUM's historical .25 evaluation temperature.
These are complete-method comparisons, not an isolated PPO-objective ablation.

Escaped members stop acting, but continue receiving later external team
rewards. Both methods count all inactive ticks, including zero-reward ticks,
and fold a reward delayed by k ticks into the last actual action with gamma^k.
That action's PPO target includes the discounted tail; no post-escape action
or actor sample is invented. GUM's original loss remains, but historical
undiscounted-tail bookkeeping is explicitly corrected for this contract.
The registered 80-tick horizon is a true task endpoint with zero bootstrap,
including timeouts. An individual escape is not a joint episode boundary.

## Validation and budget

Arithmetic, sequence padding, action matching, mode enforcement, old-logp
reconstruction and immutability, optimizer/RNG restoration, frozen evaluation
and interrupted pointer recovery are tested. A sequential-escape synthetic
case pays at ticks 1, 3 and 5, exercising zero-reward inactive gaps.

Before chamber comparisons, three separate learners each trained for 4,096
interactions on a five-color/five-anonymous-slot pixel relationship. All passed
the locked .80 accuracy and .50 absolute improvement validation threshold.
This task validates optimization/plumbing; its fixed action relationship is
easier than the chamber's episode-shuffled controls and is not cooperation
evidence. All validation trajectories and weights are preserved separately.
There was one validation configuration and zero chamber tuning trials.

Both methods have the same interaction ceiling; actual ticks and actions are
reported when early termination reduces usage. Compute is not equalized.
Multiple PPO epochs, parameter counts, optimizer steps, CPU time, wall time,
process RSS and CUDA allocation are charged separately. CPU execution uses
one thread and deterministic operations for these small models.

Actual training and frozen evaluation are viewable through a read-only local
viewer, and every complete archive is playable with verified reconstruction.
The viewer labels method, phase, team seed, episode, tick and resources.
Every episode checkpoint stores all four policies, optimizers, random streams
and pending PPO rollouts. Recovery requires agreement with the episode ledger.
Any example GIF is selected by ordinal, not by success.

## Conditional decision

Only repeatable development improvement permits freezing the fixed final
candidate and writing a separate untouched Room A transfer protocol. Both
methods failing means failure under this pilot contract/budget, with other
methods or budgets unresolved. An implementation check failing stops the study
and is not counted as a learning-method failure. No new mechanism is added
after a negative result without a new hypothesis and protocol.

```bash
python -m pytest tests/test_school_independent_ppo.py -q
python scripts/validate_escape_ppo.py work/independent-ppo-validation-reproduction
python scripts/run_escape_ppo_study.py work/independent-ppo-development-v1 --device cpu --keep-viewer
python scripts/run_escape_ppo_study.py work/independent-ppo-development-v1 --replay-only
```
