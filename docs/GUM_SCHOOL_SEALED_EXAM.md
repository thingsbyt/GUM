# GUM School sealed examinations

## Third lesson result

The third official lesson, `causal-workshop.controls.001`, passed on 2026-10-09.
The trained swarm scored **32/32** on post-freeze color- and position-shifted
control problems. Its pre-lesson self and a matched fresh learner each scored
**6/32**; random scored **2/32**. The 95% Wilson lower bound was 0.893. The two
earlier lessons jointly retained **48/48 before and after**, and all seven gates
passed.

The source freeze names commit
`6043e3b231df34cd92d4a7aa8104ff0fb5529358`. The engine promoted snapshot
`sha256-d2ff17304c7286ccb1c18c7a33a8c769c9dfeca998566cb14a557261aaa2dd31`.
Training used 854 interactions; the complete training, baseline, control,
retention, and replay protocol used 2,566. The canonical report is
[`evidence/gum-school/sealed/object-laboratory-occlusion-v1/SEALED_EXAM_REPORT_003.json`](../evidence/gum-school/sealed/object-laboratory-occlusion-v1/SEALED_EXAM_REPORT_003.json),
with SHA-256 digest
`ba8a624ea60b348911a8ca98e3f213e029f4d1c91a7fd2af8387222268778f7e`.

The intervention schedule is engineered. Reward teaches the learner to retain
and activate a probe-then-repeat strategy; within each new world, it grounds
the shuffled anonymous control from observed progress. This is a bounded
control-grounding result, not a claim of general causal reasoning.

## Second lesson result

The second official lesson, `object-laboratory.functional-category.002`, passed
on 2026-10-08. The trained swarm scored **32/32** on a post-freeze RGB-shifted
sealed set. The same promoted learner before this lesson scored **14/32**, a
matched fresh swarm scored **14/32**, and random scored **15/32**. The 95%
Wilson lower bound was 0.893. Lesson 1 retention was **24/24 before and after**
training, and all seven gates passed.

The source freeze names commit
`b5888dc63f95cabdd997174fe1aef85e0cb21f0b`. The engine promoted snapshot
`sha256-9f4dfc739f7b96199269464b92630e1728d639456ebd6fce2e86ad404a6af463`.
Training used 1,067 interactions; the entire training, baseline, control,
retention, and replay protocol used 1,904. The canonical report is
[`evidence/gum-school/sealed/object-laboratory-occlusion-v1/SEALED_EXAM_REPORT_002.json`](../evidence/gum-school/sealed/object-laboratory-occlusion-v1/SEALED_EXAM_REPORT_002.json),
with SHA-256 digest
`4198237756d9b46d60d5d7e7c8c6c09ef5a44c47f53ffcdd3e8c1edce64726eb`.

The learner inferred stable object identity across the appearance change and
learned the anonymous final action mapping from scalar reward. The visual glyph
comparison and the policy for when to probe are engineered. This is evidence
for a bounded functional-categorization lesson, not unrestricted concept
learning.

## First lesson result

The first official GUM School lesson, `object-laboratory.occlusion.001`, passed
its post-freeze sealed examination on 2026-10-08. The frozen four-policy swarm
scored **32/32**, the matched fresh swarm scored **0/32**, and a random policy
scored **14/32**. The 95% Wilson lower bound was 0.893. All seven promotion
gates passed, so the school engine atomically promoted candidate snapshot
`sha256-2ce4ead31ceb61f73b70af8ed19cfd60ec70e293031d926c9e2ea208b1cacee5`.

The canonical report is
[`evidence/gum-school/sealed/object-laboratory-occlusion-v1/SEALED_EXAM_REPORT.json`](../evidence/gum-school/sealed/object-laboratory-occlusion-v1/SEALED_EXAM_REPORT.json).
Its SHA-256 digest is
`d79eae166176147be86e743a27797737dfb6485e1555120f7eacc99ab012ffa5`.

## Freeze and selection order

The decision-relevant source was committed at
`3237986684115e23c522d3f25f7599872ec22f45`. The examiner recorded hashes for
the learner, engine, adapters, curriculum, schemas, and transitive school
support before training. Training then completed using only the four public
training seeds. Only after the candidate snapshot was frozen did the evaluator
draw 32 unique seeds from operating-system entropy and record a commitment to
the seed manifest. The completed evidence package releases the manifest and
nonce so its commitment can be recomputed.

Neither the trainer nor learner received the seed manifest, private world
packages, audit state, action meanings, final positions, or task answers. The
temporary world packages were removed before evidence was saved. Source and
protocol hashes were rechecked after evaluation.

## Structural shift and controls

The sealed world kept the reviewed public contract but varied the object
palette, post-occlusion palette, horizontal slot geometry, vertical speed,
occlusion barrier position and height, animation duration, background, and
barrier appearance. These variations were implemented and tested before the
source freeze; none was changed after the sealed seeds were selected.

The matched fresh swarm used the same pixel inputs, four-policy size, horizons,
and evaluation resources but had no training. A random policy supplied the
second precommitted control. Candidate evaluation was frozen and deterministic;
one decision-relevant trial replayed exactly.

## Resource and gate result

Official training used 239 interactions over 32 public-seed episodes. Training,
candidate evaluation, both controls, and replay used 5,869 interactions in
total, below the lesson's 12,000-interaction ceiling. The success, retention,
evidence-integrity, resource-limit, control-advantage, input-boundary, and
replay gates all passed. Fresh advantage was 32.0.

This is an official pass for one narrow lesson, not a general-intelligence
claim. Pixel segmentation and motion fitting remain engineered. The anonymous
destination-to-action mapping is learned from reward. No result is claimed for
the later schools or for cross-family transfer.

## Reproduce the protocol

The official command refuses to reuse a non-empty workspace:

```powershell
python scripts/run_school_sealed_exam.py --workspace .test-temp/school-sealed
python -m pytest -q tests/test_school_sealed.py
```

A reproduction draws a new seed manifest and is therefore a replication, not
the original decision record.
