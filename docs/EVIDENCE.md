# Evidence map

`EVIDENCE_INDEX.json` is the machine-readable catalog used by GUM Studio. This page explains the hierarchy. A screen recording makes behavior understandable; the frozen protocol and JSON audit carry the claim.

For the public-preview tree, absolute release-machine path prefixes in text audits were replaced with `RELEASE_MACHINE_OUTPUTS\` or `RELEASE_MACHINE_HOME\`. The same path-only sanitation was applied to text members inside the public evidence ZIPs. Relative suffixes, measurements, seeds, source hashes, decisions, and outcomes were not changed. The original v0.1 frozen Git tag and full-history bundle preserve the byte-exact earlier snapshot.

## Tier 1: learning-to-learn

Primary files:

- [`evidence/learning-to-learn/LEARNING_TO_LEARN_PROTOCOL.json`](../evidence/learning-to-learn/LEARNING_TO_LEARN_PROTOCOL.json)
- [`evidence/learning-to-learn/LEARNING_TO_LEARN_AUDIT.json`](../evidence/learning-to-learn/LEARNING_TO_LEARN_AUDIT.json)
- [`evidence/learning-to-learn/LEARNING_TO_LEARN_LEDGER.jsonl`](../evidence/learning-to-learn/LEARNING_TO_LEARN_LEDGER.jsonl)

Five independent meta-seeds produced 160 training tasks and 120 held-out tasks. Four runs beat their original strategy. Median speedup was 2.30×; mean speedup was 4.48×. Four runs improved from early to late life. One run regressed to 0.90×. No solution paths were retained.

This supports: **GUM learned a reusable acquisition strategy.**

It does not support: **GUM can invent arbitrary learning algorithms.**

## Tier 2: concept formation

Primary files:

- [`evidence/concept-genesis/CONCEPT_GENESIS_PROTOCOL.json`](../evidence/concept-genesis/CONCEPT_GENESIS_PROTOCOL.json)
- [`evidence/concept-genesis/CONCEPT_GENESIS_AUDIT.json`](../evidence/concept-genesis/CONCEPT_GENESIS_AUDIT.json)

The concept count was withheld. GUM selected five clusters from unlabeled transitions. An audit-only comparison found a perfect one-to-one mapping to the five hidden mechanisms. The previous handwritten signature distinguished only three of them. Transfer was 20/20, retention 4/4, and distinct-skill composition succeeded.

This supports: **GUM formed useful event categories rather than receiving the old five-category vocabulary.**

It does not support: **GUM understands arbitrary natural images.**

## Tier 3: actual files

Primary files:

- [`evidence/data-rescue/DATA_RESCUE_PROTOCOL.json`](../evidence/data-rescue/DATA_RESCUE_PROTOCOL.json)
- [`evidence/data-rescue/DATA_RESCUE_AUDIT.json`](../evidence/data-rescue/DATA_RESCUE_AUDIT.json)

Anonymous actions changed real JSONL working files. GUM discovered a verified five-operation workflow and transferred it to five unfamiliar files. Clean JSONL and CSV outputs were produced. Source files remained byte-for-byte unchanged. Experienced learning took 10 interactions versus 3,480 for a perception-matched blank copy.

This supports: **the learner can control a useful real-file workflow.**

It does not support: **the learner can synthesize arbitrary data-repair code.** The environment supplied five safe tools and a visual quality view.

## Tier 4: reusable skill growth

Primary files:

- [`evidence/growth/GROWTH_CHALLENGE_PROTOCOL.json`](../evidence/growth/GROWTH_CHALLENGE_PROTOCOL.json)
- [`evidence/growth/GROWTH_CHALLENGE_AUDIT.json`](../evidence/growth/GROWTH_CHALLENGE_AUDIT.json)

GUM created five opaque concepts and five compiled skills, transferred over 12/12 surface variants, retained 4/4 skills, shared portable knowledge with a blank recipient, and composed two distinct prior skills. Experienced search took 29 interactions versus 35,687 for a blank copy.

## Tier 5: grounded clarification

Primary files:

- [`evidence/clarification-test/CLARIFICATION_AUDIT.json`](../evidence/clarification-test/CLARIFICATION_AUDIT.json)
- [`evidence/clarification-test/TRANSCRIPT.txt`](../evidence/clarification-test/TRANSCRIPT.txt)
- [`evidence/clarification-test/CLARIFICATION_LANGUAGE_MEMORY.json`](../evidence/clarification-test/CLARIFICATION_LANGUAGE_MEMORY.json)

A fresh learner induced six color/shape words from pixels, utterances, and pointing demonstrations. It resolved 30/30 ambiguous references by asking a targeted question, executed 15/15 clear references without a question, and refused an ungrounded word. A blind random choice succeeded on 15/30 ambiguous trials.

This supports: **GUM can learn and use a tiny grounded visual vocabulary, including interactive clarification.**

It does not support: **GUM understands unrestricted natural language.** The action family, rendering, and question mechanism are engineered.

## Showcase evidence

The `evidence/showcase` folder preserves earlier or complementary experiments used in the public gallery:

- `two-rocket` tests two-agent Asteroids coordination and includes solo, random, and no-message comparisons.
- `dual-reveal-gate` tests information sharing where each agent observes only part of a hidden answer.
- `cooperative-rescue` is a fictional two-agent environment, not biomedical evidence.
- `useful-workflows` covers ten bounded file/data workflows.
- `grounded-dialogue` is an earlier, broader command-grounding experiment.
- `concept-bridge` is a deliberately preserved mixed result: 96% apple accuracy but 53.22% overall across ten categories, below the precommitted target.

Showcase experiments have different protocols and generations. Their scores must not be averaged into a universal GUM score.

## Evidence archives

The `evidence/archives` directory contains the larger evidence packages, including source snapshots and raw transition traces. Public-preview copies have release-machine home paths sanitized as documented above; the release manifest records their current SHA-256 values.

