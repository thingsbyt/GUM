# Independent PPO / GUM cooperative development pilot

## Decision

**Neither tested method learned reliable three-member cooperation under this
pilot contract and budget. Stop this configuration study. Do not examine Room A
or add another mechanism on the strength of these results.** A larger-budget
study or different treatment needs a new hypothesis and preregistration.
This is not evidence that the room is unlearnable, nor that every GUM or PPO
configuration is incapable. This four-team pilot is exploratory, not powered
for significance. Historical gradient study `26af943` is preserved; its
value-gradient correction was not promoted.

## Preregistration and validation

The [protocol](INDEPENDENT_PPO_PROTOCOL.md) and
[machine-readable plan](../evidence/gum-school/studies/independent-ppo-v1/PROTOCOL.json)
were committed as `ad4d0429c6e19336ea490d90b74bbf797b84ee9b` before
decision-relevant runs. Protocol SHA256:
`f06d5b2a789d4a1e5add53291d5a44e1efd263665cfa973d80347da98c5702af`.
The original GUM team used that commit; the seven remaining independent cases
used `fa088c1`, the registered case-parallel execution amendment. No learning
configuration, seeds, budget, outcome, or stopping rule changed.

Four freshly initialized independent teams per method trained for 256 episodes
of 80 ticks, with no sharing or messaging. Each team was frozen-evaluated on
32 matched development instances at episodes 0, 64, 128, and 256. Neither prior
credit/gradient evaluation seeds nor hard Room A instances were used. Final
checkpoint 256 was fixed, not selected from the curve. There were zero chamber
tuning trials. Independent teams, not their four bodies, are the replicates.

The nine PPO tests passed before training: clipped loss for both advantage
signs; GAE/targets and boundaries; fixed old probabilities; whole-sequence
padding and recurrent reconstruction; action/mode matching; exact pending
rollout, optimizer and RNG restore; frozen weights; and real four-member
checkpoint-pointer recovery for both methods. A sequential-escape synthetic
test at ticks 1, 3, and 5 exercised later team rewards and zero-reward gaps.
All actual frozen evaluations passed learning-state checks; all eight final
four-member restores passed. The full local suite passed **433 tests**.

The separate pixel-to-anonymous-slot validation task passed its locked accuracy
and improvement thresholds for all three learners:

| Validation seed | Before | After 4,096 interactions |
|---|---:|---:|
| 440101 | 23.05% | 95.31% |
| 440211 | 18.75% | 91.41% |
| 440307 | 19.92% | 93.36% |

This easier fixed-control task checks optimization/plumbing, **not cooperation
or episode-shuffled control discovery**. Its weights were not reused. One
validation configuration cost 12,288 training plus 1,536 evaluation
interactions, 3,071 optimizer steps and 37.48 seconds. Historical GUM
development expenditure was not quantified or equalized.

A final presentation-only regression test exposed a standalone-folder replay
label indexing bug. It was fixed without altering training or outcomes; all
10 final PPO/recording/replay tests passed. No failed learning-implementation
check was bypassed to continue chamber training.

## Every team and fixed checkpoint

Every entry below had **0/32 three-member completions at all four evaluation
checkpoints**, and **0/256 training completions**. Escape counts are individual
escapes, not completed teams or independent replications.

| Team seed | Method | Training escapes | Eval escapes: 0 / 64 / 128 / 256 | Final gate crossings | Final mean longest plate dwell, ticks |
|---|---|---:|---|---:|---:|
| 12000100 | GUM | 17 | 4 / 4 / 3 / 3 | 7 | 2.4375 |
| 12000100 | PPO | 28 | 4 / 3 / 5 / 4 | 8 | 2.1875 |
| 12010100 | GUM | 14 | 3 / 3 / 3 / 2 | 7 | 2.65625 |
| 12010100 | PPO | 23 | 3 / 7 / 2 / 2 | 5 | 1.53125 |
| 12020100 | GUM | 17 | 5 / 5 / 3 / 6 | 8 | 1.84375 |
| 12020100 | PPO | 14 | 5 / 3 / 4 / 5 | 6 | 1.15625 |
| 12030100 | GUM | 17 | 1 / 1 / 0 / 2 | 6 | 2.25 |
| 12030100 | PPO | 14 | 1 / 1 / 4 / 0 | 0 | 3.21875 |

Final completion was 0/128 evaluated episodes per method, across four teams.
GUM's total individual escapes changed from 13 before training to 13 afterward;
PPO changed from 13 to 11. Brief plate occupancy and gate crossings occurred,
but these observations do not establish learned roles or causal knowledge.
There was no completed team's time-to-completion to estimate.

Neither method met the preregistered rule: final rate and improvement each at
least 10 percentage points in at least three teams, with the exploratory
team-bootstrap gain lower bound above zero. All four team gains and all paired
PPO-minus-GUM primary differences were zero. The locked bootstrap and t
intervals are [0, 0] because the observed samples are tied; **these do not bound
unseen performance or establish equivalence**. The pre-final-outcome
`d5e9cbb` uncertainty supplement reports a two-sided exact 95% interval
[0, 0.6024] for the probability an independently trained team meets the
threshold, given 0/4. That is not an episode completion-rate interval.

There is no demonstrated cooperation to claim was retained or generalized.
Held-out development curves show no primary improvement beyond initial
behavior. They were repeatedly measured without tuning; they are now revealed
development evidence, not a new sealed examination. No cross-task retention
or Room A transfer claim is made.

## Methods and resource accounting

This compares **complete methods**, not PPO's objective alone. Both actors and
critics use only their own 96x160 RGB raster, previous anonymous action,
scalar reward and recurrent episode history. Diagnostic plate/gate labels never
select rewards, training examples or targets. Four policies, optimizers and
random streams remain separate in each team.

| Difference | GUM reference | Independent PPO |
|---|---|---|
| Representation | 8x8 pooling, visual width 32 | 16x16 pooling, visual width 64 |
| Memory | recurrent width 64, per-action memory 16 | GRU width 64, previous-action one-hot |
| Parameters per member | 33,186 | 76,102 |
| Objective | original Monte Carlo actor-critic | clipped PPO, GAE lambda .95 |
| Discount / learning rate | .97 / .001 | .99 / .0003 |
| Training sampling | declared 20% uniform mixture | categorical, no forced mixture |
| Updates | one episode, one pass | eight episodes, four epochs, four-episode minibatches |
| Evaluation | temperature 1 | temperature 1 |

GUM uses the audited novelty-off/episodic-action-mixer-off configuration and
original shared value gradients. Temperature 1 differs from historical GUM's
.25. Both methods explicitly correct historical inactive reward-tail timing:
each zero or nonzero inactive tick elapses; a later public reward at delay k
is discounted by gamma^k into the last executed action. No fictitious inactive
actor samples are added. Individual escape does not end the joint episode.
The 80-tick finite task horizon has zero bootstrap. These bookkeeping and
evaluation differences are declared, not portrayed as the untouched historical
reference.

PPO uses fixed old log probabilities, advantages and targets, whole-episode
minibatches, hidden-state reconstruction from zero on each pass, masked right
padding and a declared KL stopping threshold. No independent transition
shuffling, centralized critic or privileged state is used. The inspectable
implementation follows [PPO](https://arxiv.org/abs/1707.06347) and
[GAE](https://arxiv.org/abs/1506.02438), with recurrent handling informed by
[SB3's documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html).
[Cooperative PPO literature](https://arxiv.org/abs/2103.01955) motivates a
baseline; it does not guarantee this independent pixel-only variant succeeds.
This is a standard-method reference point, not a novel learning contribution.

| Total across four teams | GUM | PPO |
|---|---:|---:|
| Training joint ticks | 81,920 | 81,920 |
| Frozen evaluation joint ticks | 40,960 | 40,960 |
| Training executed individual actions | 325,140 | 324,432 |
| Evaluation executed individual actions | 162,198 | 162,021 |
| Optimizer steps, all four members | 4,096 | 4,096 |
| Member rollout/episode epochs | 4,096 | 2,048 |
| Valid action-loss presentations | 325,140 | 1,297,728 |
| Padded recurrent update slots | 325,140 | 1,310,720 |
| Worker CPU seconds through collection | 3,399.42 | 2,809.84 |
| Sum of episode wall seconds | 8,577.63 | 7,943.39 |

Interactions were matched; **compute was not**. PPO minibatches have about
four times the action-loss presentations despite equal optimizer-step counts.
Derived presentation counts reconstruct each independent minibatch RNG and
exclude masked inactive/padded actions. CPU time includes environment,
initialization, storage, viewer and recovery work, not just optimization.
Concurrent makespan through aggregate collection was 2,760.71 seconds
(46.01 minutes), not the sum of overlapping episode wall times. Pause time is
included. Peak process RSS was 658.0–675.2 MiB for GUM and 637.2–646.5 MiB for
PPO, not exclusive model memory. CPU, one torch thread per worker, deterministic
operations; no GPU allocation. Exact per-team totals are in the public summary.

## Evidence, recordings and operational deviations

All **3,072 primary episodes** (2,048 training, 1,024 frozen evaluation), their
complete pre/post observations, executed actions, scalar rewards, termination
and truncation flags, failures/timeouts, per-member learning records and
coherent four-member checkpoints remain local. Episode/checkpoint ledger
positions were checked on restore. Actual training was served live; all
episodes are now playable as clearly labeled replay. The four example GIFs
are the first training and first final-evaluation episode for team 1 of each
method, selected by ordinal, not success; every example had zero escapes.

The registered parallel-execution amendment launched the remaining cases
without changing their inner loop. A storage-pressure handoff paused the exact
eight owned worker processes and moved complete artifact roots to D: via C:
junctions, with count/size/hash and recovery checks before resume. No primary
episode was restarted, dropped or selected. The original serial runner was
stopped after GUM team 1 finished, immediately on entering an unwanted duplicate
PPO baseline evaluation. No duplicate training occurred. Its last observed
one-tick/four-action prefix was reconstructed from the preserved frozen birth
checkpoint and matched the authoritative state hash; it is explicitly
reconstructed, incomplete and excluded from all primary outcomes. Any later
inference before suspension is not reconstructed. All handoff records remain
in the full result. This operational extra is not hidden tuning expenditure.

Public concise evidence:
[RESULTS_SUMMARY.json](../evidence/gum-school/studies/independent-ppo-v1/RESULTS_SUMMARY.json).
Raw master root on the original machine:
`<repository>/work/independent-ppo-development-v1`.
Primary team roots resolve to the local evidence volume's
`Codex-GUM-evidence/2026-10-10/independent-ppo-development-v1` directory.
Validation: `work/independent-ppo-validation-v1` in the repository.
The original protocol, dependency snapshots, all source commits, raw
`STUDY_RESULTS.json`, relocation/recovery records and archive SHA256 values
are preserved. Large local brains/recordings are not claimed to be on GitHub.

User deliverables are in the calling task's `outputs` directory:
`independent-ppo-full-results.json`, `independent-ppo-summary.json`,
`learning-curves.csv`, `training-curves.csv`, `resources.csv`, both PNG curves,
this report, and all four GIF/verified sidecar pairs.

## Reproduction

Run from a checkout containing `ad4d042` (serial) or `fa088c1` (independent case
selectors). Locked runtime: Python 3.12.10, NumPy 2.5.2, PyTorch 2.11.0+cu128;
full dependency versions are in each `PROVENANCE.json`. Use new empty output
directories; the runner rejects overwriting a study. The protocol fixes every
training/evaluation seed and configuration.

```bash
python -m pytest tests/test_school_independent_ppo.py -q
python scripts/validate_escape_ppo.py work/ppo-validation-reproduction
python scripts/run_escape_ppo_study.py work/ppo-study-reproduction --device cpu --keep-viewer
# Or eight independent cases, one per method/team, on different loopback ports:
python scripts/run_escape_ppo_study.py work/ppo-case-team1 --only-team 1 --only-method ppo --device cpu --port 8790
# Replay the preserved completed master without learning:
python scripts/run_escape_ppo_study.py work/independent-ppo-development-v1 --replay-only --port 8786
# Report the preserved aggregate; Matplotlib is only a reporting dependency:
python scripts/report_escape_ppo_study.py work/independent-ppo-development-v1 outputs
```

For the actual operational merge, after all seven amended cases finish and
their idle viewers remain alive, use
`python scripts/merge_escape_ppo_cases.py work/independent-ppo-development-v1`.
It requires the preserved first-case and handoff files, rejects missing or
duplicate registered cases, and charges live process CPU at collection. It is
an execution-specific aggregation tool, not a prerequisite for ordinary serial
reproduction. The report figure dependency was isolated in
`work/ppo-report-dependencies` (Matplotlib 3.11.2); its NumPy 2.5.4 did not change
the training runtime. Complete CSVs and full JSON are the numerical authority,
not smoothed plots or example footage.
