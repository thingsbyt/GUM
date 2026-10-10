# Escape chamber learning diagnosis

## What the first rehearsal establishes

The first preserved Room A rehearsal contains six training episodes from one
persistent four-member team. It is one learning history, not six independent
replications. All 1,440 joint transitions replay exactly against their saved
pixels, rewards, terminal flags, and state hashes. Every policy tensor changed.
The run therefore establishes that experience was recorded and optimization
occurred. It does not establish useful learning or cooperation.

The team produced zero escapes. It entered the plate 73 times, opened the gate
72 times for 102 total ticks, and placed another body at the inside gate while
it was open for six ticks. Plate occupancy never lasted more than four ticks.
The team proposed a gate crossing 23 times, but all 23 proposals occurred while
the gate was closed. It never proposed the crossing action while the gate was
open.

## The concrete learning concern

Every episode visited 240 distinct exact joint images in 240 ticks. The generic
exact-frame novelty bonus therefore paid an estimated `+2.333` per learner per
episode while the external task return was `-0.240`. The intrinsic term was
about 9.7 times the magnitude of the task signal. Mean action entropy was
`1.60930`, or 99.9914% of the five-action maximum. These measurements are
consistent with rewarding continual visual change while retaining essentially
uniform behavior. They do not by themselves prove causation.

Frozen linear probes show that gate and plate/gate conditions are often
decodable from the compact visual representation. Matched birth-policy probes
perform similarly: two learners were unchanged on both probes, one regressed
slightly on both, and one improved only on the combined condition. This is
evidence that the random encoder often preserves relevant pixels, not evidence
that Room A training learned the mechanism.

All semantic measurements are diagnostic only. Coordinates, action meanings,
plate labels, and gate state never enter a policy update.

## Smallest justified next experiment

Room A remains frozen. A separately labeled development room retains four
independent bodies, shuffled anonymous controls, sparse external reward,
simultaneous physics, and the requirement that one body hold while another
crosses. Nobody starts on the plate and no roles, messages, routes, or semantic
labels are supplied. The plate-to-gate distance is reduced from five cells to
two, and one body begins at the gate approach. This increases useful encounters
without encoding who should hold.

The matched pilot compares fresh teams under three sequential conditions:

1. the reference exploration configuration;
2. the same configuration with only exact-frame novelty disabled;
3. condition 2 with the exact-observation action mixer also disabled.

Each condition receives identical initial random weights and member, training,
and held-out evaluation seeds within a replication. Evaluation occurs before
and after training. Every transition, failure, checkpoint, and study plan is
retained. The plan is written before the first team is born.

A single lucky crossing or escape is not the milestone. The milestone is
repeated post-training coordination across independent teams that improves over
their matched fresh-policy baselines. Development-room performance cannot be
reported as success in Room A.

This ordering is consistent with prior work rather than copied from it:
goal-proximal curricula are a documented response to sparse-reward encounter
rarity, while modern curiosity methods operate on learned or fixed feature
representations rather than paying every exact raw image as categorically new.
The present pilot is an ablation of GUM's own measured failure mode, not an
implementation of those papers.

- Florensa et al., *Reverse Curriculum Generation for Reinforcement Learning*:
  <https://arxiv.org/abs/1707.05300>
- Pathak et al., *Curiosity-driven Exploration by Self-supervised Prediction*:
  <https://arxiv.org/abs/1705.05363>
- Burda et al., *Exploration by Random Network Distillation*:
  <https://arxiv.org/abs/1810.12894>

## Development-pilot outcome and retention follow-up

The first matched development pilot used two fresh teams per condition, 12
training episodes per team, and eight held-out evaluation episodes before and
after training. All three conditions averaged `0.125` escapes per evaluation
episode both before and after training. Neither replication improved. Removing
exact-frame novelty and then the exact-observation action mixer therefore did
not, by itself, produce retained coordination.

Rewarded training events did occur: the least forced-exploration condition
recorded two escapes in one replication and three in the other. This moves the
immediate bottleneck from “the team never encounters useful experience” to
“ordinary updates do not measurably retain and generalize that experience” in
this small pilot. It does not prove that credit assignment is the only issue,
and two teams per condition are not a definitive sample.

The next predeclared comparison tests ordinary actor-critic learning against
reward-selected consolidation. Consolidation may rehearse only a genuinely
rewarded training trajectory generated by the team itself. Each member receives
its own pixels, executed anonymous actions, and scalar rewards. Evaluation
experience, coordinates, semantic action identities, event labels, roles,
routes, teacher actions, and scripted targets are prohibited. This mechanism is
engineered; any advantage would mean that engineered consolidation helped retain
self-generated experience, not that the team invented replay or language.

The consolidation pilot also used two fresh teams per condition. Each team ran
12 held-out evaluations, 24 training episodes, and the same 12 held-out
evaluations again. Ordinary actor-critic and reward-selected consolidation both
averaged `0.0833` escapes per evaluation episode before and after training. In
each condition one replication improved and one regressed. The consolidation
condition rehearsed two genuinely rewarded episodes, but showed no aggregate
advantage.

The fair conclusion is narrow: neither the tested exploration removals nor
whole-trajectory reward-selected rehearsal solved retention in these small
pilots. GUM has not yet demonstrated learned coordination. The evidence now
supports examining agent-temporal credit assignment and the policy update
itself, using these frozen pilots as regression fixtures, before adding
communication, roles, or a larger world model.

## Reproduce

```bash
python scripts/diagnose_escape_chamber.py work/escape-chamber-rehearsal --output diagnostic.json
python scripts/run_escape_development_study.py work/escape-development-pilot --device cuda
```
