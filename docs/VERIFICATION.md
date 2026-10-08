# Frozen release verification

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
