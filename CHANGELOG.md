# Changelog

## Unreleased — GUM School Phase 1

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

