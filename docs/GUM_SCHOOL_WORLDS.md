# GUM School foundational worlds

## Status

Phase 2 admits the first three source-reviewed adapters: Object Laboratory,
Causal Workshop, and Changing Maze. This is world-infrastructure evidence, not
a learning result. No curriculum lesson, sealed examination, promotion, or
transfer-matrix cell was run during admission.

The canonical admission record is
[`evidence/gum-school/admissions/foundational-v1.json`](../evidence/gum-school/admissions/foundational-v1.json).
It is referenced by SHA-256 from every corresponding admission record in the
machine curriculum. The record also anchors the exact source hashes for the
public protocol, adapters, and audit runner.

## What the worlds test

### Object Laboratory

Four identities move along crossing paths. One is marked only at the start,
then all pass behind an opaque barrier. Surface colors change before the final
choice. A useful policy must connect motion before and after occlusion rather
than select a fixed color. The six anonymous slots cover advancing, observing,
and four final positions.

### Causal Workshop

Eight anonymous controls contain three ordered progress mechanisms, a reset, a
backward effect, and inert decoys. Their slot assignments change by seed. A
useful policy must intervene, observe visible consequences, preserve discovered
prerequisites, and recover from regressions.

### Changing Maze

The learner receives a local RGB window placed in a fixed global canvas; unseen
cells remain black, so navigation requires accumulated memory. Movement slots
are shuffled by seed. Revision worlds visibly change the control mapping and
close a formerly useful route during the episode, forcing control regrounding
and replanning.

These are distinct mechanisms, not three visual skins of one simulator. Their
state machines, useful temporal abstractions, action counts, horizons, and
failure modes differ.

## Admission boundary

World packages contain exactly two JSON files: a learner-safe public contract
and private generator data. Loaders reject unknown fields, mismatched
identities, unsupported mechanisms, and any attempt to name executable code.
Only explicitly registered Python adapters can run.

For each adapter the audit verified:

- the declared pixel shape, anonymous action count, horizon, and reward range;
- deterministic reset and transition replay from the same seed and actions;
- forced truncation at the hard horizon;
- copy-safe human frames and audit state;
- absence of prohibited fields from the public package and transition info;
- data-only packages and explicit loader registration; and
- random and scripted baselines run through the public observation boundary.

The scripted policies receive the same RGB observations, anonymous slots,
rewards, termination signals, and two public event fields available at the
world contract. They never receive `audit_state`. They are admission controls
showing that a general observation-driven strategy can solve the environment;
they are not GUM and do not count as curriculum performance.

## Admission dry-run results

Each baseline ran 24 deterministic development trials.

| Adapter | Random success | Scripted success | Random mean steps | Scripted mean steps |
|---|---:|---:|---:|---:|
| Object Laboratory | 14/24 | 24/24 | 5.38 | 6.50 |
| Causal Workshop | 22/24 | 24/24 | 72.54 | 16.71 |
| Changing Maze | 1/24 | 24/24 | 234.33 | 29.67 |

The causal random control often succeeds given its long horizon, but requires
more than four times as many interactions as the public-observation controller.
Object random trials end sooner only because early guesses terminate many
failures; their lower raw step count is not an efficiency advantage.
This is why admission records both success and efficiency. These numbers test
world calibration only; they are not evidence that GUM has learned the tasks.

## Curriculum connection

All six authored generator identifiers resolve through a closed registry to one
adapter and one reviewed mechanism. An unknown generator or an
adapter/generator mismatch is rejected. GUM's ordinary harness also registers
all three adapters, so the same public world contract is used by manual tools
and the school lifecycle.

The admission run freezes the existing safety ceilings without changing their
numeric values. It generates no sealed seeds or answers. Sealed evaluation
material still must be selected only after the learner, trainer, curriculum,
adapters, and source hashes are frozen for an evaluated run.

## What remains before official training

The bounded cross-seed learner and trainer now run these adapters through the
Phase 1 engine. Their first development-only rehearsal was safely quarantined:
the trained candidate and a matched fresh learner both scored 0/8. See
[`GUM_SCHOOL_TRAINING_LANE.md`](GUM_SCHOOL_TRAINING_LANE.md).

The remaining work is learner research, not world admission. In particular,
the first lesson needs a temporal representation capable of carrying identity
through occlusion. Official curriculum training must not begin by substituting
the scripted admission controllers or relabeling public development trials as
sealed evidence.
