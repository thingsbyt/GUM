# GUM School engine

## Current status

Phase 1 provides the generic school lifecycle without implementing or running
the Object Laboratory, Causal Workshop, or Changing Maze adapters. Tests use
small synthetic state directories and source-reviewed fake callbacks; they do
not train GUM or create sealed worlds.

The implementation lives in `gum/school/`:

- `validation.py` freezes and validates the curriculum contract;
- `snapshots.py` stores content-addressed learner snapshots and an atomically
  replaced promoted pointer;
- `evaluation.py` validates measurements and computes all seven gates; and
- `engine.py` schedules lessons, isolates candidates, records training and
  evaluation evidence, promotes or quarantines, resumes interrupted decisions,
  and writes the school report.

## Lifecycle

1. Initialize the workspace from a learner-state directory. Its files become a
   verified immutable snapshot; an atomic pointer marks it promoted.
2. Select only the next authored lesson whose prerequisites are satisfied.
3. Clone the promoted snapshot into a run-specific candidate directory.
4. Run a source-reviewed trainer under the lesson's declared limits. Curriculum
   JSON can never name executable code.
5. Snapshot the trained candidate before evaluation. The evaluator receives a
   separate frozen copy; mutation is detected and aborts the run without
   corrupting the candidate or promoted snapshot.
6. Validate evidence paths and SHA-256 hashes, recompute every gate, and prepare
   an explicit decision record.
7. On a pass, atomically update the promoted pointer and progress. On any gate
   failure, leave the pointer unchanged and write a quarantine record.
8. Append the decision once to the anchored hash ledger and regenerate the
   report.

## Interruption safety

Promotion uses a persisted `decision-prepared` state before changing the
promoted pointer. If interruption occurs immediately after pointer replacement,
reopening the workspace observes the prepared decision, recognizes the already
installed content-addressed snapshot, completes progress and evidence records,
and does not increment the generation or append the decision twice.

Initialization and aborts have equivalent recovery paths. Incomplete runs are
never silently treated as promotions. Multiple incomplete runs cause a hard
stop for manual review. The current implementation is deliberately
single-writer; multi-process locking remains a later hardening task.

## Transfer matrix

Sequential lesson evaluations may report useful diagnostic transfer, but they
do not count toward the official matrix. An official cell must prove that its
branch:

- began from the recorded pre-curriculum snapshot;
- trained on exactly one declared source school;
- performed no learning on the target examination;
- used a fresh control with identical perception and limits;
- used the precommitted trial count and confidence level; and
- preserved source, protocol, trajectory, and evidence hashes.

The report remains incomplete until all nine isolated-source cells exist.
Existing cells are append-only and cannot be silently overwritten.

## Gate computation

The engine does not accept a caller-supplied pass/fail result. It recomputes:

1. sealed performance, including point thresholds and confidence lower bounds;
2. retention of every capability from every previously promoted lesson;
3. evidence integrity;
4. measured resource limits;
5. matched-fresh point and confidence-bound advantage;
6. learner-input boundary integrity; and
7. deterministic replay verification.

The evaluator still supplies measurements and verification evidence, so source
review and independent evaluation remain necessary. Phase 1 makes tampering and
missing evidence visible; it does not make the evaluator infallible.
