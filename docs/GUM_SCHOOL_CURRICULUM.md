# GUM School curriculum

## Status and scope

This document is the human-readable specification for GUM School version 1.
It accompanies [`curriculum/gum-school-v1.json`](../curriculum/gum-school-v1.json)
and the strict schemas beginning with
[`gum-school-curriculum-v1.schema.json`](../curriculum/schemas/gum-school-curriculum-v1.schema.json).

**Phase 0 is specification only.** No lesson in this document has trained GUM,
no new world adapter is admitted, and no result is claimed. The machine file is
marked `specified-not-trained`; the three foundational adapter records are
marked `specified-not-admitted`. Their checks must produce evidence before a
school runner may load them.

The curriculum asks whether a persistent learner can gain reusable competence
from an authored sequence of meaningfully different worlds. A high score on a
training world is not enough. Promotion requires unseen performance, retention,
matched controls, intact evidence, resource compliance, input-boundary
compliance, and replay.

## Non-negotiable boundaries

Every lesson separates four things:

1. **Engineered structure:** observations, anonymous actions, reward interface,
   learning machinery, hard limits, metrics, and validation.
2. **Training experience:** authored examples with public training seeds.
3. **Withheld answer:** action meanings, causal rules, goal coordinates,
   successful sequences, concept assignments, and task-specific answers.
4. **Sealed evaluation:** post-freeze worlds unavailable to the learner,
   trainer, optional language model, and promotion code until evaluation.

The learner receives only the inputs named by a lesson. Audit state remains in
the harness. World packages are data, never executable plugins. New adapter
code is source-reviewed and registered explicitly. Failed candidates and their
trajectories are preserved; they never replace the last promoted learner.

The local language model is optional, disabled by default, and untrusted. It
may eventually propose JSON, labels, or prose through a validated inbox. It may
not execute code, modify a promoted learner, view sealed data, grade itself or
the learner, relax a budget, register an adapter, or supply a task answer. The
entire authored curriculum remains usable when no language model is present.

## Twelve grades

Grades are dependency labels, not ages and not a claim that intelligence grows
along one universal line. A learner can be strong in one grade and weak in
another.

| Grade | Capability | Passing evidence |
|---:|---|---|
| 1 | Attend | Tracks stable entities and important changes through clutter or temporary disappearance. |
| 2 | Act | Grounds anonymous controls from consequences without receiving an action map. |
| 3 | Predict | Predicts immediate effects before acting and improves intervention efficiency. |
| 4 | Conceptualize | Groups different-looking events by a useful shared function. |
| 5 | Remember | Retains promoted skills after shutdown, reload, and intervening lessons. |
| 6 | Plan | Completes ordered prerequisites and delayed-reward chains. |
| 7 | Revise | Detects changed evidence and abandons a formerly successful strategy. |
| 8 | Communicate | Learns grounded words, follows instructions, and asks a discriminating question. |
| 9 | Explain | Produces an evidence-linked trace summary of what changed and why an action was selected. |
| 10 | Cooperate | Uses partial views, selects roles, shares only needed information, and tests whether a partner helps. |
| 11 | Work | Operates on real artifacts with source preservation and independent validation. |
| 12 | Transfer | Passes sealed cross-family examinations without task hints or post-reveal code changes. |

## School portfolio

Each school is an authored family with a teaching purpose. Procedural variation
may instantiate a reviewed family; it may not invent arbitrary tasks and call
them a curriculum.

### Object Laboratory

Primary grades: 1, 3, 4, and 5.

- **Identity through occlusion:** one object disappears behind a barrier and
  returns with changed texture, illumination, or orientation. Identity depends
  on trajectory and continuity, not a color lookup.
- **Comparison under transformation:** pairs differ in one controlled property
  while distractors vary independently.
- **Function over appearance:** differently shaped objects share an observable
  interaction consequence; similar-looking decoys behave differently.
- **Revisit after interference:** object skills are re-evaluated after a lesson
  in another school and after save/reload.

### Causal Workshop

Primary grades: 2, 3, 4, and 6.

- **Anonymous controls and effects:** interventions reveal which controls move,
  toggle, connect, or block unfamiliar objects.
- **Affordance contrasts:** a tool works only on compatible targets, with
  matched decoys preventing a simple proximity rule.
- **Activator–transformer–blocker composition:** delayed success depends on an
  ordered set of tool roles; local visible effects alone do not reveal the full
  chain.
- **Counterfactual pair:** matched worlds differ by one causal dependency so a
  memorized sequence fails and a mechanism-sensitive policy succeeds.

### Changing Maze

Primary grades: 2, 5, 6, and 7.

- **Partial-view mapping:** local pixels require landmark memory; global maps
  and coordinates remain audit-only.
- **Blocked-route revision:** a once-short route closes after a visible event.
- **Control remapping:** action meanings change at a signaled boundary, testing
  fresh grounding without erasing prior control knowledge.
- **Prerequisite detour:** an apparently unrelated event opens the true route,
  making planning more useful than wall following.

### Physics Playground

Primary grades: 3, 6, and 7.

- **Delayed push:** forces have delayed effects, separating action from outcome.
- **Momentum and braking:** successful interception requires predicting motion
  rather than chasing the current position.
- **Collision chains:** one moving body transfers motion to another under
  varied mass and friction.
- **Rule shift:** gravity, friction, or restitution changes after a visible cue;
  stale dynamics must be revised.

### Logic Chamber

Primary grades: 4, 6, 7, and 9.

- **Hidden prerequisites:** visible switches affect distant gates only after a
  discoverable ordering constraint.
- **Long-chain credit:** intermediate events are individually unrewarded but
  needed for terminal success.
- **Counterfactual chamber:** a minimally changed world invalidates one inferred
  dependency while preserving the rest.
- **Trace explanation:** the learner reports observations and transitions that
  support its current dependency graph; unsupported causal prose does not pass.

### Language Classroom

Primary grades: 8 and 9.

- **Showing and pointing:** words are grounded through varied rendered scenes,
  natural phrases, and pointer demonstrations.
- **Nouns, verbs, and relations:** withheld combinations test composition rather
  than memorized sentences.
- **Discriminating questions:** ambiguous instructions require a question whose
  answer separates the live candidates.
- **Grounded report:** the learner names what it observed and did, linked to the
  recorded trace; fluency without grounding is not success.

### Cooperation Worlds

Primary grade: 10.

- **Complementary views:** each agent observes evidence the other lacks.
- **Optional partner:** some worlds are cheaper alone, so always communicating
  or always adding an agent is penalized.
- **Role exchange:** capability or control asymmetry changes across episodes;
  fixed roles do not generalize.
- **Information bottleneck:** communication has an explicit cost and only
  task-relevant messages improve efficiency.

### Asteroids Academy

Primary grades: 3, 7, and 10.

- **Anonymous flight controls:** control mappings and spawn patterns vary within
  the reviewed game rules.
- **Aim and survival:** delayed motion, collision risk, and limited lives require
  prediction and restraint.
- **Two-rocket coordination:** agents divide targets while avoiding friendly
  fire under partial local views.
- **Shuffled-control revisit:** retained visual and coordination concepts must
  help while obsolete motor mappings are relearned.

### Artifact Workshop

Primary grades: 6, 9, and 11.

- **Diagnosis before mutation:** unfamiliar structured artifacts are inspected
  and a proposed repair is validated before replacement.
- **Source-preserving repair:** work occurs in a bounded copy; atomic replacement
  and prior-version backup are mandatory.
- **Schema and semantic validation:** syntactic validity alone is insufficient;
  invariant checks must pass independently.
- **Cross-format transfer:** previously learned normalization and verification
  ideas are tested on a structurally different, explicitly supported format.

### Sealed Finals

Primary grade: 12.

- The learner, school engine, curriculum, adapters, and source hashes freeze
  before final task material is selected.
- An independent evaluator is preferred. A precommitted generator is acceptable
  only when independence is unavailable and its selection procedure is frozen.
- Final worlds combine mechanisms from at least two schools and include a
  within-family control so failure can be localized.
- No code, threshold, lesson, or learner-state change is allowed after the
  examination material is revealed. Failures remain in the published matrix.

Sound is deferred. It will need a separate sensory contract and leakage audit;
decorative sound does not count as a new modality.

## Foundational machine sequence

The v1 JSON intentionally specifies only the first milestone. Later schools
remain human-authored here but do not enter the executable curriculum until the
candidate/promotion boundary works reliably.

| Sequence | Lesson | Grade | Structural teaching change | Training interactions |
|---:|---|---:|---|---:|
| 1 | `object-laboratory.occlusion.001` | 1 | Occlusion plus appearance change breaks color matching. | 12,000 |
| 2 | `object-laboratory.functional-category.002` | 4 | Function is preserved while shape and texture change. | 16,000 |
| 3 | `causal-workshop.controls.001` | 2 | Anonymous controls produce distinguishable consequences. | 12,000 |
| 4 | `causal-workshop.composition.002` | 6 | Tool roles must be ordered around prerequisites and decoys. | 18,000 |
| 5 | `changing-maze.memory.001` | 5 | Local views require persistent landmark memory. | 15,000 |
| 6 | `changing-maze.revision.002` | 7 | Paths and control meanings change, invalidating the old strategy. | 20,000 |

Generators have names such as `authored-object-occlusion-v1`. The name denotes
a reviewed family template with bounded variation. It is not permission to
sample arbitrary mechanics. Each implementation must document which properties
vary, which remain invariant, and why the resulting tasks teach the named
capability.

## Seed and examination policy

Training, development, and retention seeds are public, deterministic, and
pairwise disjoint. The validator checks overlaps within every lesson and world
admission. Public seeds are not evidence of transfer; they exist for debugging
and reproducibility.

The public Phase 0 file contains no sealed seed list. Its three SHA-256 values
commit to the named post-freeze selection protocols, and each examination says
`commitment_scope: selection-protocol`. They are not represented as commitments
to seed manifests that do not yet exist. At examination time, the independent
evaluator creates the inaccessible seed manifest after source freeze and before
the evaluated run, records its own `seed-manifest` commitment, and releases the
preimage only with the completed evidence package. A validator must reject a
raw `sealed_seed` or `sealed_seeds` field anywhere in a public curriculum.

Every evaluated lesson includes:

- a before-training candidate baseline;
- a matched fresh learner with identical perception and resource limits;
- a random baseline and a scripted non-solution baseline when the examination
  protocol requires one;
- public development seeds and separately generated new-seed trials;
- at least one structural shift where the family supports one;
- retention checks on protected promoted skills;
- complete successes and failures, resource samples, hashes, trajectories, and
  replay metadata; and
- an explicit promote, rollback, abort, or quarantine decision.

## Metrics

All metrics are computed per trial and summarized with counts, means, medians,
and bootstrap confidence intervals. Raw trial rows remain available.

- **Success:** the adapter's precommitted binary terminal criterion. Audit state
  may score it but may never enter learner observations or `public_info` before
  the criterion becomes externally visible.
- **Efficiency:** interactions used on successful trials, reported both directly
  and as `horizon / interactions`. Failures consume the full horizon for
  aggregate cost comparisons; early aborts are labeled, not imputed as success.
- **Uncertainty:** explicit clarification, probe, or abstention decisions divided
  by opportunities defined before the run. When an adapter lacks such an
  action, the learner must expose a bounded confidence report; the evaluator
  cannot infer confidence after seeing the answer.
- **Unnecessary actions:** repeated no-ops, known collisions, redundant tool use,
  or messages identified by a source-reviewed predicate committed before the
  run. The predicate may use audit state for scoring but may not guide the
  learner.
- **Fresh advantage:** candidate sealed success rate divided by
  `max(matched_fresh_success_rate, 1 / trials)`. Efficiency is reported
  separately and cannot rescue a failed success gate.
- **Retention regression:** the protected skill's last promoted success rate
  minus its current success rate on the same retained protocol. Positive values
  are regressions; a missing or unverifiable baseline fails the gate.

## Transfer matrix

The primary report is a matrix, not a highlight reel. Cell `(A, B)` compares a
candidate that has completed promoted family `A` with a matched fresh learner
on family `B`. Each cell records candidate and fresh success, success difference,
fresh-advantage ratio, interaction cost, uncertainty, unnecessary actions,
trial count, confidence interval, and evidence references. Negative and null
transfer remain visible.

The first milestone requires the complete 3×3 matrix for Object Laboratory,
Causal Workshop, and Changing Maze. At least one off-diagonal cell must beat
the matched fresh threshold in its lesson specification. No claim extends to an
unmeasured school.

## Exact promotion and retention decision

Training always starts from a snapshot of the last promoted learner and writes
to a candidate workspace. The promoted snapshot is read-only for the duration
of a lesson. A candidate is promoted only when **all seven gates** pass:

1. **Sealed performance:** success is at least the lesson's
   `minimum_success` on the sealed examination.
2. **Retention:** every protected skill's absolute success regression is no
   greater than `maximum_regression` (0.05 in v1).
3. **Evidence integrity:** source and protocol hashes, append-only ledger,
   trajectory chain, and required artifacts verify.
4. **Resource limits:** wall time, peak memory, stored artifacts, and interactions
   stay at or below every declared limit. A measurement failure is a gate
   failure, not zero usage.
5. **Control advantage:** sealed fresh advantage meets
   `minimum_fresh_advantage`; the same perception and resource limits are
   verified for both learners.
6. **Input boundary:** no prohibited input, audit-only state, sealed data, or
   task answer reached the learner, proposal system, or training process.
7. **Replay:** the decision-relevant run reproduces from frozen source, protocol,
   initial snapshot, world package, and recorded seeds within the documented
   deterministic tolerance.

The lesson also has maximum uncertainty and unnecessary-action rates. Exceeding
either is a sealed-performance failure. Passing training scores, attractive
replays, or an explanation generated after the fact cannot override a gate.

On any failure, the runner must stop before replacement, retain the promoted
snapshot, quarantine the candidate, preserve all evidence, record the failed
gate and diagnosis, and append a rollback or abort decision. Repeated failure
may deprioritize a lesson family but never erase its history. Promotion must be
atomic and idempotent so interruption cannot double-promote.

## World admission

An adapter changes from `specified-not-admitted` to `admitted` only after all of
these are evidenced:

1. it implements the public world contract;
2. audit-only state is absent from observations and learner-visible information;
3. observation kind, action kind/count, horizon, and reward range are declared;
4. reset is deterministic by seed;
5. every episode terminates or truncates at a hard horizon;
6. safe human inspection and replay work;
7. an automated hidden-state/leakage test passes;
8. random and scripted non-solution baselines are nontrivial;
9. train, development, retention, and sealed partitions cannot overlap; and
10. world data cannot execute arbitrary code.

The admission schema conditionally requires passed checks and at least one
evidence hash for `admitted`. Phase 0 supplies no such hash because those
adapters do not yet exist.

## Validation and adversarial fixtures

Validate the official file with:

```powershell
python -m gum.school curriculum/gum-school-v1.json
python -m pytest -q tests/test_school_curriculum.py
```

The validator first applies strict Draft 2020-12 schemas with unknown fields
disabled. It then rejects duplicate JSON keys, recursive answer-bearing field
names, unknown catalog references, duplicate identifiers, mismatched lesson and
admission references, missing primary exams, seed overlaps, input-boundary
collisions, impossible resource limits, invalid reward ranges, and curriculum
totals that exceed their budgets.

The negative fixtures in `tests/fixtures/gum_school/` are mutation recipes over
the valid curriculum. They prove rejection of an unknown field, a synthetic
answer field, an unsupported adapter, an overlapping seed, a zero interaction
budget, and a resource budget smaller than the declared training budget. The
synthetic leak fixture contains no real task answer.

## Phase 0 exit criteria

Phase 0 is complete only when:

- this document and the machine curriculum agree;
- all four schemas pass their metaschema checks;
- the official curriculum passes structural and semantic validation;
- every adversarial fixture fails for its intended reason;
- existing GUM tests still pass;
- publication-integrity checks find no private path, credential, or sealed
  material; and
- the repository remains on the sanitized public history with no publication
  performed by the implementation session.

The next phase may build the school engine. It may not claim any school result
until adapters are admitted and a precommitted run produces verified evidence.
