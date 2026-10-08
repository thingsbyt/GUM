# Frozen release verification

## GUM School training-lane verification

The development-only training lane was verified on 2026-10-08. A fresh
cross-seed learner ran through the real candidate, frozen evaluation, gate, and
quarantine lifecycle using only public training and development seeds:

```powershell
python scripts/rehearse_school_training.py --workspace .test-temp/school-rehearsal
python -m pytest -q tests/test_school_curriculum.py tests/test_school_engine.py tests/test_school_worlds.py tests/test_school_training.py
```

The focused school suite passed **60/60 tests**. The complete repository suite
passed **327/327 tests** in 149.28 seconds and emitted only the existing PyTorch
scalar-conversion warning in `tests/test_method.py`.
The secured Studio smoke test, documentation/package links, installed
dependencies, and public-package integrity scan also passed.

The saved run used 63 training interactions over eight episodes. The trained
candidate and matched fresh learner each scored 0/8 on development trials.
Replay and input-boundary verification passed; sealed-performance,
evidence-integrity, and control-advantage failed. The engine quarantined the
candidate, preserved the initial promoted snapshot, and recorded no promoted
lesson. The canonical report digest is
`sha256:99ddb90a0361e335ce4e8c210daf7e0055df87eff83974b3e25e0454e5bcfde5`.

This verifies the training system's safety path, not first-lesson competence.
No sealed data was generated or used, and no promotion or transfer claim
follows. Verify the exact tree with:

```powershell
python scripts/verify_freeze.py --manifest RELEASE_MANIFEST_GUM_SCHOOL_TRAINING_LANE.json
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
