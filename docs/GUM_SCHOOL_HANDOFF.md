# GUM School: implementation handoff

## Purpose of this document

This is the durable starting point for the next phase of GUM. It summarizes the
current machine, the evidence already established, the remaining architectural
gap, and the exact plan for a rigorous multi-world curriculum system.

The handoff baseline is public commit `d0065f8` on
<https://github.com/thingsbyt/GUM>. That commit is the sanitized public history;
the private archival branch must never be pushed.

## The objective

Build **GUM School**: an authored, measurable curriculum that exposes a
persistent learner to genuinely different worlds and determines whether
experience produces reusable understanding.

The target is not high training scores. The target is a growing machine that:

1. discovers useful regularities from experience;
2. retains old competence;
3. improves old competence when it returns;
4. transfers knowledge across worlds that differ in appearance and mechanism;
5. composes old knowledge to solve unfamiliar problems;
6. asks for information when uncertainty matters;
7. communicates what it learned in an inspectable form; and
8. becomes measurably faster or more reliable at learning later tasks.

## What exists now

GUM is a compact hybrid system, not one monolithic neural network. Its current
building blocks include:

- pixel-state reinforcement learning for generated grid worlds;
- visual transition encoding and unsupervised event-concept selection;
- persistent, reusable, and composable concept-level skills;
- explicit acquisition-strategy evolution;
- grounded visual word learning and clarification;
- cooperative specialist learners for Asteroids and fictional rescue worlds;
- a bounded real-file JSONL repair environment;
- persistent state, atomic replacement, prior-version backups, hash-linked
  ledgers, and independently stored ledger head/count anchors;
- a loopback-only, token-protected GUM Studio; and
- reproducible protocols, controls, raw trajectories, audits, and replays.

The common world contract lives in `gum/protocol.py`. Adapter registration and
the current learning lifecycle live in `gum/harness.py`. Generated grid worlds
are in `gum/world_creator.py`; first-class cooperative adapters are in
`gum/builtin_worlds.py`; Studio is in `gum/interface.py`.

### Current verified results

- The official pytest collector passed **267/267 tests** locally.
- GitHub verification passed on Python 3.10, Python 3.12, and the complete
  optional neural stack.
- The secured Studio teaching smoke test learned six visual words, asked the
  expected clarification question, and executed the resolved target.
- The v0.2.1 release manifest verifies 279 files byte-for-byte after canonical
  line-ending normalization.
- A fresh, precommitted Concept Genesis rerun selected five concepts without
  being given the count, passed 20/20 transfer checks and 4/4 retention checks,
  and solved the composed fifth task in 23 interactions versus 12,491 for a
  matched fresh learner: **543.09×**.

See `docs/VERIFICATION.md`, `docs/EVIDENCE.md`, and
`RELEASE_MANIFEST_v0.2.1.json` for the exact records.

## The central architectural gap

GUM has a shared lifecycle and evidence boundary, but not yet a single universal
mind that can use every adapter. The grid learner, concept learner, language
bridge, and cooperative teams use different learning machinery and separate
memories.

GUM School must not hide this fact. Its first job is to measure the existing
learners consistently. Its deeper job is to discover which representations and
skills can actually cross those boundaries. A unified dashboard is not evidence
of a unified mind.

## Curriculum philosophy

The curriculum must be authored as a set of meaningful problem families. A
procedural generator may vary an authored lesson, but it may not invent random
tasks and label them education.

Each lesson separates four things:

- **Engineered structure:** sensors, action interface, learning algorithm,
  safety limits, and scoring procedure.
- **Training experience:** examples the learner may interact with and learn
  from.
- **Withheld answer:** controls, mechanisms, successful sequence, concept
  assignment, or other task-specific solution.
- **Sealed evaluation:** worlds and seeds inaccessible during training and
  selected before the evaluated run begins.

Training success alone never promotes a lesson or supports a claim.

## World portfolio

Different skins of the same maze do not count as different worlds. The initial
portfolio should vary embodiment, observations, action spaces, causal rules,
time, feedback, and social structure.

| School | Primary capability | Representative challenge |
|---|---|---|
| Object Laboratory | persistence, tracking, comparison, categorization | follow an object through occlusion and changed appearance |
| Causal Workshop | intervention, affordances, mechanism discovery | identify which unfamiliar controls activate, combine, or block tools |
| Changing Maze | spatial memory, replanning, strategy abandonment | reach goals after paths or control meanings change |
| Physics Playground | prediction, momentum, collision, timing | intervene in moving systems with delayed effects |
| Logic Chamber | prerequisites, long chains, counterfactuals | discover an apparently unrelated event sequence required for success |
| Language Classroom | grounded nouns, verbs, relations, instructions | learn from showing, ask a discriminating question, report the result |
| Cooperation Worlds | partial views, communication, role choice | decide whether cooperation is useful, then share only needed information |
| Asteroids Academy | rapid visual control and coordination | survive, aim, and coordinate under shuffled controls |
| Artifact Workshop | safe real-file and data operations | diagnose and repair unfamiliar structured artifacts without changing sources |
| Sealed Finals | transfer outside the authored training distribution | solve worlds created only after the learner and curriculum are frozen |

Sound remains a future modality. It should receive its own sensory contract and
audits rather than being added as decorative feedback.

## Twelve-grade capability sequence

1. **Attend:** identify stable entities and important changes.
2. **Act:** ground anonymous controls through consequences.
3. **Predict:** anticipate immediate effects before acting.
4. **Conceptualize:** group different-looking events by useful function.
5. **Remember:** retain concepts and skills across shutdown and revisit.
6. **Plan:** solve ordered prerequisites and delayed-reward chains.
7. **Revise:** abandon a once-good strategy when evidence changes.
8. **Communicate:** learn grounded words, follow instructions, and ask questions.
9. **Explain:** produce an evidence-linked account of what changed and why it
   selected an action; templated trace summaries are acceptable initially.
10. **Cooperate:** share partial experience, select roles, and decide whether a
    second agent is necessary.
11. **Work:** use safe tools on real artifacts with source preservation and
    independent validation.
12. **Transfer:** pass sealed, cross-family examinations without task-specific
    hints or post-exam code changes.

Grades are capability dependencies, not a claim that intelligence develops in a
single linear order. A learner may be strong in one grade and weak in another.

## Required evaluation for every lesson family

Every family must define:

- a baseline before training;
- training worlds and exact interaction limits;
- unseen within-family transfer worlds;
- at least one structurally shifted transfer world where appropriate;
- retention tests for previously promoted skills;
- a matched fresh learner with the same perception and resource limits;
- deterministic seeds plus a new-seed evaluation;
- success, efficiency, uncertainty, and unnecessary-action metrics;
- full failure preservation;
- resource measurements;
- source hashes, protocol hashes, trajectories, and replay metadata; and
- explicit abort, rollback, and promotion decisions.

The primary report is a **transfer matrix**. Cell `(A, B)` measures whether
learning in world family A helps, harms, or has no effect on learning or
performance in family B. This is the main test of reusable understanding.

## Promotion rule

Training happens on a candidate snapshot, never directly on the last promoted
mind. A candidate is promoted only when all mandatory gates pass:

1. sealed performance improves by the precommitted minimum;
2. no protected old skill falls beyond its allowed tolerance;
3. evidence and ledger verification pass;
4. time, memory, storage, and interaction budgets remain within limits;
5. the result beats the stated fresh/random/control baseline where required;
6. the candidate did not receive prohibited state or a task solution; and
7. the outcome can be replayed from saved artifacts.

Failed candidates are quarantined with their trajectories and diagnosis. They
must not silently replace the promoted learner. Repeatedly failing lesson
families should be deprioritized, not erased.

## Role of a local language model

The local LLM is optional and untrusted. It may propose curriculum JSON, labels,
or human-readable explanations. It may not:

- execute code;
- modify a promoted mind;
- see sealed evaluation answers;
- choose whether its own proposal succeeded;
- bypass resource limits;
- create adapters dynamically; or
- supply task solutions to the learner.

Proposals enter an inbox and must pass JSON Schema validation, adapter support,
feasibility, duplication, leakage, novelty, and budget checks. Invalid proposals
fall back to the authored curriculum. GUM School must remain fully usable with
the LLM disabled.

## World admission standard

A world adapter is admitted only if it:

1. implements the public world contract;
2. keeps audit-only state outside learner observations and `public_info`;
3. declares observation type, action type/count, horizon, and reward range;
4. provides deterministic reset by seed;
5. terminates within a hard horizon;
6. supports safe inspection and replay;
7. passes an automated hidden-state/leakage check;
8. has nontrivial random and scripted baselines;
9. includes train, development, and sealed seed partitions; and
10. cannot execute arbitrary dropped code.

New executable adapters remain source-reviewed and registered explicitly.
World packages are data, not plugins.

## Machine-readable curriculum contract

Create `curriculum/gum-school-v1.json` and a JSON Schema. Each lesson should
contain, at minimum:

```json
{
  "lesson_id": "causal-workshop.controls.001",
  "schema": "gum-school-lesson-v1",
  "school": "causal-workshop",
  "capabilities": ["control-grounding", "causal-intervention"],
  "adapter": "registered-adapter-name",
  "training": {"generator": "authored-family", "seeds": [], "interaction_budget": 0},
  "evaluation": {"sealed_seed_commitment": "sha256:...", "trials": 0},
  "retention": {"protected_skills": [], "maximum_regression": 0.05},
  "promotion": {"minimum_success": 0.8, "minimum_fresh_advantage": 1.0},
  "learner_inputs": [],
  "prohibited_inputs": [],
  "resource_limits": {},
  "evidence": {"preserve_failures": true, "hash_algorithm": "sha256"}
}
```

The real schema must use positive values and enumerated adapter/capability
identifiers. Empty values above are placeholders showing structure, not a valid
lesson.

## Implementation sequence

### Phase 0 — specification before training

1. Add curriculum, lesson, examination, and world-admission schemas.
2. Write the complete human-readable GUM School curriculum.
3. Create a validator that rejects unknown fields, unsupported adapters,
   solution-bearing fields, overlapping seed partitions, and invalid budgets.
4. Add schema and adversarial-validator tests.

### Phase 1 — the school engine

1. Add a `gum/school/` package for curriculum loading, scheduling, execution,
   evaluation, promotion, rollback, and reporting.
2. Snapshot the promoted mind before every lesson.
3. Run candidate learning in a bounded workspace.
4. Evaluate transfer and retention before promotion.
5. Store an append-only school ledger and machine-readable decision record.
6. Resume safely after interruption without double-promoting a candidate.

### Phase 2 — first three foundational schools

Implement Object Laboratory, Causal Workshop, and Changing Maze first. Each
must change mechanisms, not merely colors. Do not add all ten schools before the
promotion and retention machinery is trustworthy.

### Phase 3 — School view in Studio

Add a nontechnical School interface showing:

- current grade and active lesson;
- promoted versus candidate brain;
- mastered, fragile, forgotten, and untested skills;
- transfer matrix;
- before/after/fresh comparisons;
- failures and rollback reasons;
- brain and artifact growth;
- resource use; and
- replay links.

The interface must never turn an internal pass into an AGI or consciousness
claim.

### Phase 4 — language-model proposal lane

Benchmark the local model on at least 100 proposals. Record valid-JSON rate,
schema pass rate, feasibility, duplicates, prohibited leakage, novelty, repair
success, and whether accepted proposals produce useful learning. Begin with a
small proposal share and automatically disable it if reliability falls below the
precommitted threshold.

### Phase 5 — sealed cross-world examination

Freeze the learner, school engine, curriculum, adapters, and source hashes.
Only then generate or obtain the final worlds. Prefer task selection by someone
who did not implement the learner. No code changes are allowed after examination
worlds are revealed; failures remain part of the result.

## First implementation milestone

The first milestone is intentionally smaller than the full vision:

> One persistent candidate learns an authored sequence across Object
> Laboratory, Causal Workshop, and Changing Maze; it passes unseen tests,
> retains protected skills, beats matched fresh learners on at least one
> cross-world transfer, survives interruption, and produces a complete transfer
> matrix and promotion ledger.

This milestone should be attempted before Asteroids, cooperation, language, or
real-file work are folded into the automated school loop.

## Definition of success for the next chat

The next working session should not start by training. It should finish Phase 0:

- a reviewed `GUM_SCHOOL_CURRICULUM.md`;
- strict JSON Schemas;
- a valid `gum-school-v1.json`;
- negative fixtures proving malformed, leaky, overlapping, and impossible
  curricula are rejected;
- documentation of the exact promotion and retention rules; and
- passing tests and updated public documentation.

Only after that foundation is committed should implementation of the school
runner begin.

## Non-negotiable research rules

- Problems may be authored; solutions must remain withheld.
- Evaluation rules and thresholds are written before evaluated runs.
- The learner never receives audit-only state.
- Training and sealed evaluation data never overlap.
- A language model cannot grade itself or the learner.
- Regression and failure evidence is preserved.
- Claims stay scoped to the tested world families.
- A visually impressive replay never substitutes for a controlled comparison.
- New evidence never overwrites a historical frozen release.

## Recommended opening prompt for the next chat

> Continue GUM from the sanitized public repository attached to this task. Read
> `docs/GUM_SCHOOL_HANDOFF.md` completely before acting. Implement Phase 0 of
> GUM School: the reviewed curriculum document, strict schemas,
> machine-readable curriculum, adversarial fixtures, validators, tests, and
> documentation. Do not train GUM yet. Preserve the public/private Git-history
> boundary, do not expose sealed solutions, and run the existing verification
> suite before publishing changes.
