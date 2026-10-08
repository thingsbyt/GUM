# GUM School training lane

## Status

The bounded training lane has completed a learned development-only rehearsal
through the real school engine. The trained swarm scored 32/32 on unseen public
development seeds against 0/32 for a matched fresh swarm. The disposable
candidate was nevertheless quarantined. No official curriculum lesson, sealed
examination, promotion, or transfer-matrix cell has run.

The canonical rehearsal record is
[`evidence/gum-school/rehearsal/foundational-lane-v1/REHEARSAL_REPORT.json`](../evidence/gum-school/rehearsal/foundational-lane-v1/REHEARSAL_REPORT.json).
It hashes the exact learner, trainer, evaluator, engine, protocol, adapters, and
all run artifacts.

## Learner boundary

`CrossSeedSchoolLearner` is a deliberately small persistent learner. It receives
only the declared RGB pixels, anonymous action slots, scalar reward,
termination signal, and public `event` and `success` fields. It never receives
`audit_state`, a private world package, action meanings, goals, answer keys, or
sealed seeds.

Generated world packages are temporary. The saved public evidence retains
public trajectories and replay data but no `genome.private.json` files.

The learner combines a small linear visual value function with episodic object
memory derived from public pixels. In the first lesson it detects a visually
marked object, associates it across visible frames, estimates its motion, and
retains the predicted destination through occlusion. Action values are shared
by adapter and memory state, not keyed by world identity or seed.

The object segmentation and motion-fitting procedure, feature grid, update
rule, discount, exploration schedule, and replica protocol are engineered
biases. The anonymous action values are learned from reward. No action meaning,
correct destination, control map, or task answer is hardcoded or supplied by
the world. All persistent state survives save/reload in strict JSON.

## On-demand swarm

Every run starts with one policy. If action confidence remains below the
predeclared threshold for three consecutive decisions, the learner spawns a
helper by copying the current shared knowledge. It can do this only until four
policies exist.

The trainer rotates later episodes across the available policies, giving each
helper its own experience stream. After each full rotation, the policies share
value estimates for actions they have actually tried. Evaluation is
deterministic: the policies vote, with their mean value breaking ties. They
exchange learned summaries only—never live hidden state, private packages, or
answers.

The reference runner interleaves those learning lanes deterministically rather
than launching four operating-system processes. This keeps evidence replayable;
the lanes are independent and can later be executed concurrently without
changing their information boundary.

## Bounded rehearsal protocol

The rehearsal:

1. creates a fresh persistent learner and initializes the promoted snapshot;
2. clones that snapshot into the first lesson's candidate directory;
3. permits at most 300 interactions over four episodes for each of four public
   training seeds;
4. freezes the resulting candidate snapshot;
5. compares it with an equally sized matched fresh swarm on 32 public
   development trials;
6. replays one candidate trial exactly; and
7. submits the record to the ordinary seven-gate decision engine.

The evaluator explicitly sets sealed-protocol and protocol-hash verification to
false. This is intentional: public development trials cannot impersonate a
sealed examination. Those false fields force quarantine even if development
performance looks strong.

## Recorded result

The learner spawned its second and third policies during the first episode and
its fourth during the second, always because confidence remained low. Each
policy then received four of the 16 training episodes. The group communicated
four times and used 130 interactions, below the 300-interaction ceiling.

The trained swarm succeeded 32/32 times on development trials; an equally sized
fresh swarm succeeded 0/32. The 95% Wilson lower bound for the trained result is
0.893. Deterministic replay and the learner-input boundary passed. A
deterministic same-budget ablation with replication disabled reached 25/32,
below the 0.80 development threshold.

Only the sealed-performance and evidence-integrity gates failed, because the
run intentionally used public development seeds and marked protocol hashes as
unverified. The candidate was quarantined and the promoted snapshot remained
unchanged.

## Readiness decision

The student has learned the first lesson on the public development protocol.
That is a real but narrow development result: its pixel tracker is engineered,
while its anonymous action policy is learned. It says nothing yet about the
later schools.

An official first-lesson run now requires freezing this learner, trainer,
engine, curriculum, adapters, and source hashes before an independent evaluator
selects withheld examination material. No change made after seeing sealed
results may flow back into that run. Later lessons should continue in the
development lane until they meet the same standard.

## Reproduce

Use a new, empty workspace; the command refuses to overwrite an existing run.

```powershell
python scripts/rehearse_school_training.py --workspace .test-temp/school-rehearsal
python -m pytest -q tests/test_school_training.py
```
