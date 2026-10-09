# Frozen release verification

## Recurrent causal meta-policy research result

A development-only follow-up replaced the causal scheduler with a
permutation-equivariant recurrent policy. The learner received pixels, its
previous anonymous action, scalar reward, termination, and learned memory. It
did not receive event labels, hidden state, a tried-action mask, a probe order,
or a repeat-on-progress rule. Training used the curriculum's full 12,000 public
interactions and reward-ranked self-imitation; evaluation allowed only 11
actions per held-out public-seed trial.

The frozen policy scored **439/512 (85.7%)**, versus **38/512 (7.4%)** random
and **32/512 (6.3%)** for the same untrained network. Its 95% Wilson interval
was 82.4%–88.5%; mean completion was 7.21 actions and the repeated-useless-action
rate was 11.2%. It passed the pre-existing 80% success and 65%
Wilson-lower-bound development gates. Iterative development informed the final
generic sampling temperature, so this is not sealed confirmation or an
official promotion. No sealed cases were used.

```powershell
python scripts/run_school_recurrent_meta_experiment.py --budget 12000 --trials 512
python -m pytest -q tests/test_school_recurrent_meta.py
python scripts/verify_freeze.py --manifest RELEASE_MANIFEST_GUM_SCHOOL_RECURRENT_META.json
```

## Strict causal-scaffold ablation

The development-only ablation removed the handcrafted probe-then-repeat
intervention schedule and used no sealed cases. The promoted learner fell from
64/64 with the scaffold to 4/64 without it. Starting from the Lesson 2 snapshot
and retraining without the scaffold for the full 12,000-interaction budget
reached 10/64, with a 95% Wilson upper bound of 0.264. The strict 0.80 criterion
failed.

```powershell
python scripts/run_school_strict_causal_ablation.py --trials 64
python scripts/verify_freeze.py --manifest RELEASE_MANIFEST_GUM_SCHOOL_STRICT_CAUSAL_ABLATION.json
```

## GUM School causal-control promotion verification

The third official lesson ran from source freeze commit
`6043e3b231df34cd92d4a7aa8104ff0fb5529358`. The trained learner scored 32/32;
the candidate before this lesson and matched fresh learner each scored 6/32;
random scored 2/32. The earlier lessons retained 48/48 both before and after
training. Exact replay, the input boundary, cumulative retention, the evidence
ledger, and all seven promotion gates passed. The promoted snapshot is
`sha256-d2ff17304c7286ccb1c18c7a33a8c769c9dfeca998566cb14a557261aaa2dd31`.

The focused school suite passed **67/67 tests** and the complete suite passed
**334/334 tests**, with only the existing PyTorch scalar-conversion warning.
Verify this tree with:

```powershell
python scripts/verify_freeze.py --manifest RELEASE_MANIFEST_GUM_SCHOOL_CAUSAL_PROMOTION.json
```

## GUM School second sealed-promotion verification

The second official lesson ran from source freeze commit
`b5888dc63f95cabdd997174fe1aef85e0cb21f0b` after the first lesson's promoted
snapshot. Training completed before 32 unique sealed seeds were selected.

The trained learner scored 32/32; the candidate before this lesson and the
matched fresh learner each scored 14/32; random scored 15/32. Lesson 1 scored
24/24 both before and after the new training. Exact replay, the input boundary,
the evidence ledger, and all seven promotion gates passed. The promoted
snapshot is
`sha256-9f4dfc739f7b96199269464b92630e1728d639456ebd6fce2e86ad404a6af463`.

The complete suite passed **334/334 tests**, with only the existing PyTorch
scalar-conversion warning. Verify this tree with:

```powershell
python scripts/verify_freeze.py --manifest RELEASE_MANIFEST_GUM_SCHOOL_FUNCTIONAL_PROMOTION.json
```

## GUM School sealed-promotion verification

The first official lesson was run on 2026-10-08 from frozen source commit
`3237986684115e23c522d3f25f7599872ec22f45`. Training completed before the
evaluator selected and committed its 32 withheld seeds:

```powershell
python scripts/run_school_sealed_exam.py --workspace evidence/gum-school/sealed/object-laboratory-occlusion-v1
python -m pytest -q tests/test_school_sealed.py tests/test_school_training.py tests/test_school_engine.py tests/test_school_worlds.py tests/test_school_curriculum.py
```

The trained swarm scored 32/32, matched fresh scored 0/32, and random scored
14/32. The success lower bound was 0.893, fresh advantage was 32.0, exact replay
passed, and no private world package remained in the saved evidence. All seven
gates passed and the engine promoted snapshot
`sha256-2ce4ead31ceb61f73b70af8ed19cfd60ec70e293031d926c9e2ea208b1cacee5`.
Training used 239 interactions; training plus candidate, controls, and replay
used 5,869 interactions.

The focused school suite passed **66/66 tests**. The complete repository suite
passed **333/333 tests** in 160.96 seconds, with only the existing PyTorch
scalar-conversion warning. Package checks, dependency checks, curriculum
validation, and the publication-integrity scan passed. The canonical sealed
report digest is
`sha256:d79eae166176147be86e743a27797737dfb6485e1555120f7eacc99ab012ffa5`.

Verify the final tree with:

```powershell
python scripts/verify_freeze.py --manifest RELEASE_MANIFEST_GUM_SCHOOL_SEALED_PROMOTION.json
```

## GUM School training-lane verification

The development-only training lane was verified on 2026-10-08. A fresh
cross-seed learner ran through the real candidate, frozen evaluation, gate, and
quarantine lifecycle using only public training and development seeds:

```powershell
python scripts/rehearse_school_training.py --workspace .test-temp/school-rehearsal
python -m pytest -q tests/test_school_curriculum.py tests/test_school_engine.py tests/test_school_worlds.py tests/test_school_training.py
```

The then-current focused school suite passed **63/63 tests**. The then-current
complete repository suite passed **330/330 tests** in 158.67 seconds and emitted only the existing PyTorch
scalar-conversion warning in `tests/test_method.py`.
The secured Studio smoke test, documentation/package links, installed
dependencies, and public-package integrity scan also passed.

The saved run used 130 training interactions over 16 episodes. Sustained
uncertainty spawned three helpers, reaching the maximum of four; each handled
four episodes and the group communicated four times. The trained swarm scored
32/32 on development trials versus 0/32 fresh. A deterministic same-budget
single-replica test scored 25/32 and missed the 0.80 development threshold.

Replay and input-boundary verification passed. Sealed-performance and
evidence-integrity failed by design, because no sealed protocol was used. The
engine quarantined the candidate, preserved the initial promoted snapshot, and
recorded no promoted lesson. The canonical report digest is
The rehearsal was regenerated after the sealed-world source addition; its
current digest is
`sha256:78652decca994b623eb984722425b5d157b0258812222e0b36bdd37e79a12fa4`.

This establishes narrow first-lesson development learning, not an official
curriculum pass. Pixel tracking is engineered; anonymous action values are
learned from experience. No sealed data was generated or used, and no promotion
or transfer claim follows. Verify the exact tree with:

```powershell
python scripts/verify_freeze.py --manifest RELEASE_MANIFEST_GUM_SCHOOL_SWARM.json
```

## GUM School Phase 2 verification

Phase 2 was verified on 2026-10-08 without running a curriculum lesson,
generating sealed examination worlds, promoting a learner, or filling a
transfer-matrix cell. The focused curriculum, lifecycle, and world suite passed
**54/54 tests**:

```powershell
python -m gum.school curriculum/gum-school-v1.json
python -m pytest -q tests/test_school_curriculum.py tests/test_school_engine.py tests/test_school_worlds.py
python scripts/admit_school_worlds.py --trials 24
```

The admission audit passed all three adapters and produced the evidence digest
`sha256:8fa1317698b9e8689b4a0169cfe6a939074369089fddc967ae83ebadea9c61f7`.
Every adapter passed contract, deterministic-reset, hard-horizon,
copy-safe-inspection, replay, hidden-state-boundary, and data-only-package
checks. Public-observation scripted controls achieved 24/24 in each world; the
corresponding random controls achieved 14/24, 22/24, and 1/24. The causal result
is admitted on a large efficiency separation as well as a smaller success
difference: 16.71 versus 72.54 mean interactions.

The complete repository suite passed **321/321 tests** in 146.58 seconds on
Python 3.12. The first sandboxed attempt produced eight loopback permission
errors in existing web-interface tests; the complete suite was rerun with local
loopback permission and passed. It emitted the existing PyTorch
scalar-conversion warning in `tests/test_method.py`. The secured Studio smoke
test, package/link check, dependency check, and public-package integrity scan
also passed, with zero publication-leak hits.

This proves that the worlds satisfy their implemented admission contract. It
does not prove that GUM can learn them. The scripted controls are independent
admission probes, not the GUM learner, and no sealed or promotion claim follows.

Verify the exact Phase 2 tree with:

```powershell
python scripts/verify_freeze.py --manifest RELEASE_MANIFEST_GUM_SCHOOL_PHASE_2.json
```

## GUM School Phase 1 verification

The reviewed curriculum and generic school lifecycle were verified on
2026-10-08 without implementing a school adapter, training GUM, or generating a
sealed world. The focused curriculum and lifecycle suite passed **39/39 tests**:

```powershell
python -m gum.school curriculum/gum-school-v1.json
python -m pytest -q tests/test_school_curriculum.py tests/test_school_engine.py
```

The tests cover strict schemas and adversarial fixtures, explicit prerequisites,
global budgets, confidence-bound gates, content-addressed snapshots, candidate
isolation, evaluation-copy mutation detection, all-promoted retention,
quarantine, atomic promotion, initialization/abort/promotion recovery,
idempotent ledger decisions, progress-tamper detection, frozen curriculum
checks, evidence path confinement, and all nine isolated-source transfer cells.

The complete repository suite passed **306/306 tests** on Python 3.12 in 147.75
seconds. It emitted the existing test-only PyTorch scalar-conversion warning in
`tests/test_method.py`; there were no failures. The secured Studio smoke test,
package/link check, dependency check, and public-package integrity scan also
passed. The integrity scan reported zero publication-leak hits.

Verify the exact Phase 1 tree with:

```powershell
python scripts/verify_freeze.py --manifest RELEASE_MANIFEST_GUM_SCHOOL_PHASE_1.json
```

The Phase 0 and v0.2.1 manifests remain unchanged historical snapshots. Phase 1
is infrastructure evidence only: the three foundational adapters remain
`specified-not-admitted`, the 3×3 transfer matrix has no experimental cells,
and no learning result is claimed.

## GUM School Phase 0 verification

The Phase 0 specification was verified on 2026-10-08 without running any school
training or generating sealed worlds. The standalone validator accepted the
canonical curriculum, and all 18 focused schema/validator tests passed:

```powershell
python -m gum.school curriculum/gum-school-v1.json
python -m pytest -q tests/test_school_curriculum.py
```

The complete repository suite then passed **285/285 tests** on Python 3.12 in
305.88 seconds. The run emitted the already documented test-only PyTorch warning
in `tests/test_method.py` about scalar conversion; there were no failures. The
public-package integrity scan also passed with no path, credential, private-key,
or other publication-leak hit, and `python -m pip check` reported no broken
requirements.

The exact Phase 0 working tree is recorded separately in
`RELEASE_MANIFEST_GUM_SCHOOL_PHASE_0.json` and can be checked with:

```powershell
python scripts/verify_freeze.py --manifest RELEASE_MANIFEST_GUM_SCHOOL_PHASE_0.json
```

`RELEASE_MANIFEST_v0.2.1.json` remains unchanged as a historical snapshot. It
correctly reports that files edited for Phase 0 no longer match v0.2.1; that is
not repaired by rewriting old evidence. No Phase 0 training result, adapter
admission, or general-capability claim follows from these verification checks.

This page records what was actually checked before the first local Git release was sealed on 2026-10-08.

## Result

**Historical v0.2.0 result: 234 tests ran; 234 passed.**

The test suite covers the neural JEPA foundation, pixel-only worlds, concept formation, skill memory, continual learning, cooperation, language grounding, file workflows, safety checks, dashboard behavior, and the later GUM challenges.

Historical command:

```powershell
python -m unittest discover -s tests -p 'test_*.py'
```

That command only discovers class-based `unittest` cases. The repository also
contains module-level pytest tests, so v0.2.1 makes pytest the official test
collector:

```powershell
python -m pip install -e ".[dev,neural]"
python -m pytest -q
```

The 234-test figure remains a truthful record of the frozen v0.2.0 run; it is
not presented as the count produced by the broader collector.

Runtime: 133.435 seconds on the release machine.

The separate end-to-end Studio smoke test also passed. It launched the local server, trained the starter visual vocabulary, elicited a clarification question, answered it, and verified execution of the resolved target:

```powershell
python scripts/smoke_studio.py --workspace .test-temp/studio-smoke
```

The publication-integrity audit also passed. It found no local home path, common credential form, or private-key header in ordinary files or text members inside evidence ZIPs, and all selected learning-boundary gates were true:

```powershell
python scripts/audit_release_integrity.py
```

This is a strong automated check, not a mathematical guarantee that no undiscovered bug or indirect cue exists. See `docs/LEARNING_INTEGRITY.md`.

## Current v0.2.1 hardening run

The broader official collector ran on 2026-10-08 after the Studio, persistence,
ledger, and bounded-search hardening:

```powershell
python -m pytest -q
```

**267 tests passed** in 141.25 seconds on Python 3.12. That run emitted one
test-only warning while converting an attached PyTorch tensor to a scalar. The
test was then changed to detach explicitly, and its five-test module passed
without warnings. The secured end-to-end Studio smoke test and publication scan
are separate commands and are reported separately so stored-evidence validation
is not confused with experiment reruns.

A fresh, precommitted Concept Genesis reproduction also passed against the
hardened source: 5 concepts selected, 20/20 transfer, 4/4 retention, and a
543.09× experienced-versus-fresh advantage on the composed fifth world. Its
protocol, audit, summary, and full raw-trace archive are linked from
`docs/EVIDENCE.md`. This remains an internal reproduction.

Verify the v0.2.1 source and evidence package with:

```powershell
python scripts/verify_freeze.py --manifest RELEASE_MANIFEST_v0.2.1.json
```

## Tested software

- Windows
- Python 3.10.16
- NumPy 2.1.2
- Pillow 11.0.0
- SciPy 1.15.3
- scikit-learn 1.7.0
- PyTorch 2.5.1+cu121
- Git 2.54.0.windows.1

## An important testing detail

The first complete run used Windows' default temporary directory. The Codex app sandbox denied writes there, causing 45 permission errors. Those were infrastructure errors, not failed assertions. The suite was rerun with `TEMP` and `TMP` pointed at a writable folder inside the repository. That exposed one genuine packaging omission: the dashboard's `web` directory had not been copied. The directory was restored from the working project, its 17 dashboard and pipeline tests passed, and then the complete suite passed in one run. The public-preview verification added seven clarification/Teaching Lab checks, bringing the total to 234.

This history is documented because a serious evidence package should include mistakes found during verification, not only the polished ending.

## Integrity checks

`FROZEN_RELEASE.json` maps every released file to its SHA-256 digest. Binary files are hashed byte-for-byte. Text is hashed in the repository's declared LF form so the same checkout verifies on Windows, macOS, and Linux. Run:

```powershell
python scripts/verify_freeze.py
```

A valid result means every binary byte and every canonical text byte still matches the v0.2.0 frozen release. Later source hardening intentionally changes the working tree and must be verified against its own release manifest. The Git commit and annotated tag provide a second, independent snapshot mechanism.

## What this does not prove

Passing tests shows that the implementation behaves as specified on this machine. It does not establish artificial general intelligence, consciousness, or independent scientific replication. Those require stronger tests—especially third-party tasks chosen only after the code is frozen.
