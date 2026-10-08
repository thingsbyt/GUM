# Architecture, without the fog

## GUM School specification boundary

Phase 0 added specification-time validation, not another learner. The
human-readable curriculum is in `docs/GUM_SCHOOL_CURRICULUM.md`; the canonical
machine curriculum and its four strict schemas are in `curriculum/`; and
`gum.school` performs structural, leakage, seed-partition, reference, and budget
checks. It contains no sealed answers.

Phase 1 implements the generic lifecycle in `gum.school`: content-addressed
snapshots keep the last promoted mind isolated, training touches only a run
candidate, evaluation uses a separate frozen copy, and all seven precommitted
gates must pass before atomic pointer replacement. Prepared decisions resume
without double promotion after interruption. Phase 2 adds three explicitly
registered, source-reviewed adapters behind strict data-only world packages.
Their admission audit verifies protocol, hidden-state boundary, baseline,
horizon, replay, and evidence checks before training.

The training lane adds a strict JSON learner whose action-value parameters are
shared by adapter rather than exact world identity. Candidate training is
bounded, evaluation reads a frozen snapshot, and public development rehearsals
are explicitly unable to satisfy the sealed-protocol gates. The first such
rehearsal was quarantined without moving the promoted pointer.

## The shortest explanation

GUM has five jobs:

1. **Notice:** compare what the world looked like before and after an action.
2. **Compress:** group recurring kinds of change into compact concepts.
3. **Remember:** save successful concept sequences as skills.
4. **Reuse:** re-identify those concepts when appearances or controls change.
5. **Improve learning:** preserve and evolve the strategy used to investigate new tasks.

The environment and the learner are deliberately separated. The environment may know the answer for scoring. The learner receives only its declared public inputs.

```text
PRIVATE AUDIT SIDE                     LEARNER SIDE

hidden mechanism ──┐                  RGB observation
true control map ──┼── scorer         anonymous actions
target solution ───┘                  scalar reward
                                            │
                                            ▼
                                  concept → skill → strategy
```

Private audit facts are written only to the final report. Leakage scans check that their field names never appear in learner traces.

## 1. Perception and concept genesis

Early GUM used a handwritten signature: count brightened and darkened connected regions. That was useful but it meant a human had already decided the vocabulary.

Concept Genesis removes the five predefined event classes. It:

1. receives raw before/after RGB frames;
2. extracts changed pixel values without assigning meaning;
3. clusters those values into a learned pixel-change vocabulary;
4. represents a whole event as a distribution of learned tokens;
5. tries several possible event partitions;
6. selects the simplest partition whose separation score is effectively best;
7. gives each resulting cluster an opaque identifier such as `genesis-05c45482b326`.

The learner is not told that five event kinds exist. In the frozen run, candidate partitions of two, three, four, and five concepts were compared; five won with a silhouette score effectively equal to 1.0.

This is learned categorization, but not unrestricted vision. The changed-pixel assumption and distribution statistics are still architectural biases.

## 2. Grounding anonymous controls

A stored skill refers to concepts, not button numbers.

Suppose an old skill says:

```text
concept-A → concept-D → concept-B
```

In a new world, button 2 may now produce concept-A and button 0 may produce concept-D. GUM briefly tests the controls, builds the local mapping, and translates the old concept-level skill back into current actions.

That is why a shuffled controller does not automatically destroy the skill.

## 3. Skill compilation

When an action sequence succeeds, GUM replaces the local button numbers with the concepts those actions produced. The resulting concept sequence is assigned a stable hash-based skill ID and saved.

Skills record support, successes, failures, confidence, and source contexts. Exact context memories permit rapid revisits; portable skills omit local control maps so another agent must ground them independently.

## 4. Composition

The composition test creates a new six-step problem only after the four three-step skills are learned. The new target is derived from the hash of the post-curriculum brain, preventing a hand-picked favorable target.

GUM tries pairs of existing skills. In Concept Genesis it solved the fifth world in 83 interactions; a perception-matched blank learner required 41,339.

## 5. Learning-to-learn

The meta-learning experiment does not preserve solutions. It preserves only three weights controlling generic best-first investigation:

- preference for states visually closer to the goal;
- preference concerning path depth;
- preference concerning how exhausted a state's actions are.

A population of strategies is evaluated across tasks. Better strategies survive; mutated descendants are tried later. Puzzle answers, paths, action maps, and task identities are discarded.

This is strategy meta-learning: experience changes how later problems are investigated. It is not unrestricted self-rewriting because the three features and mutation machinery were supplied by the program.

## 6. Persistence and evidence

Brains are JSON, not mysterious binary weights. Major evaluations include:

- a protocol written before the run;
- SHA-256 hashes of the evaluated source;
- before/after hash equality checks;
- append-only, hash-chained transition or event ledgers;
- explicit learner-input and learner-withheld lists;
- baseline results;
- readable results plus the complete audit.

This does not make an experiment automatically correct. It makes silent alteration easier to detect.

