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

## GUM School: general recurrent maze development

Primary file:

- [`evidence/gum-school/research/general-recurrent-maze-quatro-forti-v2/QUATRO_FORTI_REPORT.json`](../evidence/gum-school/research/general-recurrent-maze-quatro-forti-v2/QUATRO_FORTI_REPORT.json)

This study excludes the engineered maze specialist. One recurrent policy,
continued from the officially promoted causal-composition policy, receives
pixels, previous anonymous action, scalar reward, termination, recurrent
memory, first-visit novelty, and generic online action-effect statistics. It
never receives coordinates, a map, action meanings, a route, event labels, or
audit state.

Four learner seeds trained for 15,000 interactions each on disjoint generated
world partitions, then faced 128 untouched mazes each. The candidates solved
125/512 (24.41%), the inherited policy with the identical exploration biology
solved 116/512 (22.66%), and uniform random solved 52/512 (10.16%). Two
replicas improved on the inherited baseline and two regressed. The earlier v1
run is retained, but its policy-gradient likelihood did not match the sampled
exploration mixture; v2 corrects that mathematical error.

This supports: **generic episodic action-effect exploration transfers to unseen
mazes, and the corrected aggregate candidate result was modestly above its
inherited baseline.**

It does not support: **maze mastery, a robust benefit from recurrent weight
learning, or official Lesson 5 promotion.** The predeclared four-of-four
replication criterion failed.

## GUM School: cooperative mission swarm

Primary file:

- [`evidence/gum-school/research/cooperative-mission-swarm-v2/MISSION_SWARM_SMOKE_REPORT.json`](../evidence/gum-school/research/cooperative-mission-swarm-v2/MISSION_SWARM_SMOKE_REPORT.json)

Four durable recurrent learners watched five distinct development missions,
with captaincy distributed across all four. The team completed one causal
mission and one of four mazes. Each success was shared to the other three
members; three failures triggered recorded strategy revisions. All five public
experiences, every post-mission brain, identities, and both hash-chained ledgers
survived verification after reload.

This supports: **the cooperative identity, observation, communication,
strategy-change, and persistence machinery works end to end.**

It does not support: **a multi-agent performance advantage or maze mastery.**
This was an architecture smoke run without a baseline, not a promotion trial.

## Showcase evidence

The `evidence/showcase` folder preserves earlier or complementary experiments used in the public gallery:

- `two-rocket` tests an engineered two-agent Asteroids controller and includes
  solo, random, and no-message comparisons. Anonymous control meanings and
  pixel-derived motion tracks are learned online; tracking, target assignment,
  pursuit, avoidance, and firing logic are engineered rather than learned end
  to end.
- `dual-reveal-gate` tests information sharing where each agent observes only part of a hidden answer.
- `cooperative-rescue` is a fictional two-agent environment, not biomedical evidence.
- `useful-workflows` covers ten bounded file/data workflows.
- `grounded-dialogue` is an earlier, broader command-grounding experiment.
- `concept-bridge` is a deliberately preserved mixed result: 96% apple accuracy but 53.22% overall across ten categories, below the precommitted target.

Showcase experiments have different protocols and generations. Their scores must not be averaged into a universal GUM score.

## v0.2.1 fresh hardening reproduction

After the persistence, Studio, and bounded-search changes, Concept Genesis was
run again from a newly written protocol against the changed source hashes. It
again selected five event concepts without receiving a requested count, passed
4/4 curriculum tasks, 20/20 separate transfer worlds, and 4/4 retention checks.
On the composed fifth world, the experienced mind succeeded in 23 interactions;
the perception-matched fresh mind needed 12,491, a **543.09×** advantage.

This is a fresh internal reproduction, not an independent replication. The
readable [protocol](../evidence/v0.2.1-fresh-concept/CONCEPT_GENESIS_PROTOCOL.json),
[audit](../evidence/v0.2.1-fresh-concept/CONCEPT_GENESIS_AUDIT.json), and
[summary](../evidence/v0.2.1-fresh-concept/RESULTS.md) are included. The complete
archive in `evidence/archives/GUM_V021_FRESH_CONCEPT_GENESIS_EVIDENCE.zip`
contains the mind snapshots and raw hash-anchored traces.

## Evidence archives

The `evidence/archives` directory contains the larger evidence packages, including source snapshots and raw transition traces. Public-preview copies have release-machine home paths sanitized as documented above; the release manifest records their current SHA-256 values.

