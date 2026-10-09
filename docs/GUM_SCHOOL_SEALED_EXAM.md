# GUM School sealed examinations

## Official recurrent-policy promotion

The School engine now accepts a strict v2 evaluation record that separates
uncertainty before causal evidence from uncertainty afterward. The recurrent
track inherited only the two verified Object Laboratory promotions, imported
the unchanged public-trained recurrent policy, froze source and protocol at
commit `5da84f02f1e4c9a1f283d9ba664cd65152d8f4b7`, and excluded all 224 seeds
revealed by five earlier sealed manifests before drawing 64 new seeds.

The candidate scored **61/64 (95.3%)**, versus **8/64** for its pre-Lesson-3
base, **7/64** for the matched untrained recurrent network, and **4/64** random.
The 95% Wilson lower bound was 0.871. Initial uncertainty was 100%,
post-causal-evidence uncertainty was 0%, and unnecessary actions were 4.9%.
Lessons 1–2 retained **48/48**. All seven gates passed, so the engine atomically
promoted snapshot
`sha256-59b238c909e0dc560f175cab364c57a5ffcc121e2a703e34497963ba3c80306c`.
The next lesson is now `causal-workshop.composition.002`.

This is an official pass for the recurrent policy on the bounded anonymous
controls lesson, not a claim of unrestricted causal reasoning. The canonical
report is
[`evidence/gum-school/sealed-recurrent-official-v2/OFFICIAL_RECURRENT_PROMOTION.json`](../evidence/gum-school/sealed-recurrent-official-v2/OFFICIAL_RECURRENT_PROMOTION.json).

## Symmetry-aware recurrent confirmation

The uncertainty rule was versioned after the first recurrent confirmation
exposed a measurement error: it treated uncertainty before any evidence as a
failure even though the anonymous controls were indistinguishable. The new
rule requires both honest uncertainty on the first decision and low uncertainty
after an earlier action has produced positive scalar reward. It reads only the
policy's decision confidence and prior scalar rewards, never event labels or
hidden state. On 512 public development cases, the frozen policy had 100%
initial uncertainty, 0% post-causal-evidence uncertainty, and a mean confidence
gain of 0.720. The policy weights were not changed.

Source, candidate, public validation, and thresholds were then committed at
`7f5effed1ea5e662c3f341dfaee6c72f3bb3cc11` before a new seed manifest was
drawn. The examiner excluded all 160 seeds revealed by four earlier sealed
manifests. On 64 fresh sealed cases the candidate scored **64/64**, versus
**4/64** for the same untrained architecture and **4/64** random. Initial
uncertainty was **100%**, post-causal-evidence uncertainty was **0%**, mean
confidence rose by 0.670 after causal evidence, and unnecessary actions were
3.1%. The 95% Wilson lower bound was 0.943. Lessons 1–2 retained **48/48**,
replay was exact, and both the seven standard gates and the added
symmetry-calibration gate passed.

This is a supplemental symmetry-aware qualification, not an official
curriculum promotion. The promoted workspace was not mutated, and the original
failed confirmation below remains intact. The canonical report is
[`evidence/gum-school/sealed-recheck/causal-recurrent-meta-v2/SEALED_SYMMETRY_CONFIRMATION.json`](../evidence/gum-school/sealed-recheck/causal-recurrent-meta-v2/SEALED_SYMMETRY_CONFIRMATION.json).

## Recurrent causal-policy supplemental confirmation

The cumulative candidate combined the promoted Object Laboratory learner with
the frozen recurrent causal policy. After source commit `ed0735e`, the
supplemental protocol froze source and candidate hashes and selected 64 new
seeds from operating-system entropy. The candidate scored **64/64**; the same
untrained architecture and random each scored **4/64**. The 95% Wilson lower
bound was 0.943, unnecessary actions were 3.4%, replay was exact, and Lessons
1–2 retained **48/48**.

Formal qualification failed solely on the uncertainty gate. The policy
correctly assigns equal probability to initially indistinguishable controls,
so its first intervention is entropy-uncertain. Fast completion made those
first interventions 34.5% of all actions, above the pre-existing 20% ceiling.
The result is preserved as a failure: no confidence value was falsified, no
revealed seed was reused, and the promoted workspace was not changed. The
canonical report is
[`evidence/gum-school/sealed-recheck/causal-recurrent-meta-v1/SEALED_RECURRENT_CONFIRMATION.json`](../evidence/gum-school/sealed-recheck/causal-recurrent-meta-v1/SEALED_RECURRENT_CONFIRMATION.json).

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

### Stricter scaffold-ablation result

A subsequent public-seed test removed the complete probe-each-control and
repeat-on-progress scheduler. The promoted learner then scored **4/64**. A
Lesson 2 learner retrained without that scheduler used the full **12,000**
interaction budget and scored **10/64** (15.6%; 95% Wilson interval
8.7%–26.4%), far below the 80% criterion. The scaffolded control remained
64/64.

This stricter test failed. It shows that the current system did not invent a
general intervention policy. The sealed promotion remains valid under its
frozen protocol, but its interpretation is narrower: it demonstrates learned
use of a supplied experimental routine and within-world control grounding.
The negative-result report is
[`evidence/gum-school/strict/causal-controls-unscaffolded-v1/STRICT_CAUSAL_ABLATION.json`](../evidence/gum-school/strict/causal-controls-unscaffolded-v1/STRICT_CAUSAL_ABLATION.json).

### Recurrent policy follow-up

A research follow-up then trained a symmetry-aware recurrent policy from
pixels, previous anonymous actions, scalar rewards, termination, and learned
memory. It used no event labels, hidden world state, tried-action mask,
prescribed probe order, or repeat-on-progress rule. Under a stricter 11-action
development cap, it scored **439/512 (85.7%)**, compared with **38/512 (7.4%)**
random and **32/512 (6.3%)** for the same untrained architecture. Its 95% Wilson
lower bound was 82.4%, passing the 65% gate. Iterative development informed the
final generic sampling temperature, so the result remains development evidence,
not sealed confirmation or an official promotion; no sealed cases were opened.
The report is
[`evidence/gum-school/research/causal-recurrent-meta-v1/RECURRENT_META_REPORT.json`](../evidence/gum-school/research/causal-recurrent-meta-v1/RECURRENT_META_REPORT.json).

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
