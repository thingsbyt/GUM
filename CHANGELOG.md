# Changelog

## 2026-10-09 — Reward-gated spatial-memory development pass

- Added a Changing Maze specialist whose fixed visual memory machinery builds
  an episodic map and grounds anonymous controls only through observed pixel
  displacement; it never receives hidden coordinates, control meanings, event
  labels, or authored action sequences.
- Used scalar return to learn whether to activate the reusable map-and-plan
  strategy, exhausting the fixed 15,000-interaction public training budget.
- Passed all public research gates on 128 development mazes at 128/128, versus
  18/128 for the same untrained architecture and 0/128 when only carried
  spatial memory was erased. This is not yet a sealed promotion.

## 2026-10-09 — Official recurrent causal-composition promotion

- Froze the Lesson 4 examiner at commit `e5cd671`, then selected 64 fresh
  operating-system-entropy seeds after excluding all public and 288 previously
  revealed sealed seeds.
- The unchanged candidate scored 64/64, versus 21/64 before Lesson 4 training,
  17/64 matched fresh, and 17/64 random under the 32-action cap.
- Retained all three prior lessons at 72/72, passed all seven engine gates, and
  atomically advanced the official recurrent track to Changing Maze memory.

## 2026-10-09 — Reward-grounded causal-composition development pass

- Added scalar reward-outcome replay: positive consequences reinforce an
  action in context, while negative consequences suppress it in context.
- Shared the fixed 18,000-interaction budget 80/20 between composition and
  prior-lesson rehearsal to prevent catastrophic forgetting.
- Passed the public Lesson 4 gates at 511/512 composition trials with a 0.989
  Wilson lower bound and 254/256 controls retention. No sealed Lesson 4 data
  was used, so this is a development pass rather than an official promotion.

## 2026-10-09 — Causal-composition development attempt

- Continued the officially promoted recurrent policy for the full 18,000
  public-interaction Lesson 4 budget under a 32-action cap.
- Improved composition from 29.9% to 43.4% and retained Lesson 3 at 99.6%, but
  preserved the run as a failure because it missed the 80% criterion.

## 2026-10-09 — Official recurrent-policy promotion

- Added strict v2 engine evaluation records for symmetry-aware causal
  uncertainty and verified curriculum-prefix inheritance for a new track.
- Ran 64 newly selected official sealed trials after freezing commit `5da84f0`.
- Promoted the recurrent candidate at 61/64 versus 8/64 pre-lesson, 7/64
  matched fresh, and 4/64 random, with 48/48 retention.
- Advanced the recurrent track to `causal-workshop.composition.002`.

## 2026-10-09 — Symmetry-aware recurrent confirmation

- Versioned uncertainty measurement for anonymous controls into pre-evidence
  and post-causal-evidence phases without reading event labels or hidden state.
- Publicly validated the frozen recurrent policy on 512 development cases,
  then committed the protocol before selecting fresh sealed seeds.
- Passed a supplemental 64-trial sealed confirmation at 64/64, versus 4/64
  untrained and 4/64 random, with 48/48 prior-lesson retention.
- Preserved the earlier failed confirmation and left the official promoted
  workspace unchanged.

## Unreleased — GUM School sealed promotion

- Added a post-freeze evaluator whose seed manifest is drawn only after public
  training, committed before evaluation, and revealed with the evidence.
- Added structurally shifted Object Laboratory examinations covering palette,
  geometry, speed, barrier, background, and occlusion-duration changes.
- Ran the first official lesson from frozen commit `3237986`: the trained swarm
  scored 32/32, matched fresh scored 0/32, and random scored 14/32.
- Passed all seven gates and atomically promoted the first learned snapshot.

### Training-lane rehearsal

- Added a strict persistent cross-seed learner, bounded trainer, matched-fresh
  development evaluator, deterministic replay, and one-shot rehearsal command.
- Added temporal object memory plus uncertainty-triggered helper policies,
  capped at four, with separate episode assignments, value communication, and
  evaluation voting.
- Ran the first lesson through the actual candidate/evaluation/decision engine
  using public partitions only. The trained swarm scored 32/32 against 0/32
  fresh, while a matching single-replica ablation scored 25/32.
- Preserved the learned development run in quarantine without claiming a sealed
  evaluation, promotion, transfer result, or official curriculum pass.

### Phase 2 foundational worlds

- Added admitted Object Laboratory, Causal Workshop, and Changing Maze
  adapters with occlusion, ordered causal intervention, local navigation,
  topology change, and control-remapping mechanisms.
- Added strict data-only world packages, explicit adapter registration, and a
  closed mapping for all six authored curriculum generators.
- Added deterministic admission checks for contracts, reset, hard horizons,
  inspection copies, replay, hidden-state boundaries, and baseline separation.
- Recorded 24-trial random and public-observation scripted controls for every
  foundational adapter without running GUM training or sealed evaluation.

### Phase 1 foundation

- Added the complete human-readable GUM School curriculum and exact promotion,
  retention, transfer-matrix, sealed-evaluation, and rollback rules.
- Added strict Draft 2020-12 schemas for curricula, lessons, examinations, and
  world admission.
- Added a validated six-lesson authored sequence across Object Laboratory,
  Causal Workshop, and Changing Maze without running training.
- Added structural and semantic validation for unknown fields, unsupported
  identifiers, answer-bearing fields, seed overlap, input leaks, references,
  and infeasible budgets.
- Added adversarial fixtures and tests, while keeping the local language model
  optional, disabled, and outside grading and sealed data.
- Hardened the reviewed curriculum with explicit prerequisites, all-promoted
  retention, confidence-bound gates, provisional-budget labeling, and an
  isolated-source 3×3 transfer design.
- Added the Phase 1 content-addressed snapshot, candidate, evaluation,
  promotion, quarantine, recovery, transfer-matrix, ledger, and report engine.

## 0.2.1 — 2026-10-08

- Made pytest the official collector and added core/full continuous integration.
- Protected local Studio requests with a per-launch token, loopback-only binding,
  strict Host/Origin checks, bounded JSON bodies, and browser security headers.
- Added crash-resistant atomic state replacement with a recoverable prior copy.
- Added ledger head/count anchors so a deleted valid suffix is detectable while
  its checkpoint remains trusted.
- Added explicit concept-acquisition budgets, cancellation, and failure reasons.
- Corrected Concept Genesis wording: model selection uses an internal silhouette
  score; transfer is evaluated in separate worlds.
- Made trusted pickle loading an explicit opt-in at the API boundary.

## 0.2.0 — 2026-10-08

- First sanitized public-preview repository and GUM Studio package.
- Added the Teaching Lab, evidence browser, documentation, and public audits.

## 0.1.0 — 2026-10-08

- Frozen Growing Understanding Machine research snapshot.
- Added autonomous event-concept selection.
- Added persistent and composable concept-level skills.
- Added verified real JSONL repair workflow.
- Added five-run learning-to-learn audit with zero solution retention.
- Added protocols, audits, ledgers, evidence archives, and plain-language documentation.

