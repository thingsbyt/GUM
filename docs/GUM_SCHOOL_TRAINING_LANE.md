# GUM School training lane

## Status

The bounded training lane is implemented and has completed one development-only
rehearsal through the real school engine. The disposable candidate was
quarantined. No official curriculum lesson, sealed examination, promotion, or
transfer-matrix cell has run.

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

Its trainable state is a linear action-value table per admitted adapter. The
visual input is reduced to a 4×4 RGB sample, channel means and standard
deviations, and a bias term. The weights are shared by adapter, not keyed by
world identity or seed, so an update in one training world can affect behavior
on another. The random-generator state, counters, weights, and audit metadata
survive save/reload in a strict JSON format.

The feature grid, update rule, discount, exploration schedule, and adapter
boundary are engineered biases. The learned weights are experience-dependent.
The learner does not store task solutions or semantic action maps.

## Bounded rehearsal protocol

The rehearsal:

1. creates a fresh persistent learner and initializes the promoted snapshot;
2. clones that snapshot into the first lesson's candidate directory;
3. permits at most 256 interactions over two episodes for each of four public
   training seeds;
4. freezes the resulting candidate snapshot;
5. compares it with a matched fresh learner on eight public development trials;
6. replays one candidate trial exactly; and
7. submits the record to the ordinary seven-gate decision engine.

The evaluator explicitly sets sealed-protocol and protocol-hash verification to
false. This is intentional: public development trials cannot impersonate a
sealed examination. Those false fields force quarantine even if development
performance looks strong.

## Recorded result

The learner completed eight early-terminating training episodes and 63 total
training interactions, below the 256-interaction ceiling. On eight development
trials, both the trained candidate and the matched fresh learner succeeded 0/8.
Deterministic replay and the learner-input boundary passed. The
sealed-performance, evidence-integrity, and control-advantage gates failed, so
the candidate was quarantined and the promoted snapshot remained unchanged.

This is a successful systems rehearsal and an unsuccessful learning result. It
shows that the lifecycle, persistence, limits, evidence, replay, and rollback
path work together. It does not show that this simple learner can solve the
first lesson.

## Readiness decision

GUM School is ready for continued development training, but not for an official
curriculum run. The immediate research task is to improve the cross-seed learner
on public training and development partitions—especially temporal memory for
occlusion—while preserving the same input boundary and reporting failed runs.

Only after the learner and trainer are calibrated should their source hashes be
frozen and an independent evaluator select the withheld examination material.
No change made after seeing sealed results may flow back into that evaluated
run.

## Reproduce

Use a new, empty workspace; the command refuses to overwrite an existing run.

```powershell
python scripts/rehearse_school_training.py --workspace .test-temp/school-rehearsal
python -m pytest -q tests/test_school_training.py
```
