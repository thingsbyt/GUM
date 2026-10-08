# Frozen release verification

This page records what was actually checked before the first local Git release was sealed on 2026-10-08.

## Result

**234 tests ran; 234 passed.**

The test suite covers the neural JEPA foundation, pixel-only worlds, concept formation, skill memory, continual learning, cooperation, language grounding, file workflows, safety checks, dashboard behavior, and the later GUM challenges.

Command:

```powershell
python -m unittest discover -s tests -p 'test_*.py'
```

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

A valid result means every binary byte and every canonical text byte still matches the frozen release. The Git commit and annotated tag provide a second, independent snapshot mechanism.

## What this does not prove

Passing tests shows that the implementation behaves as specified on this machine. It does not establish artificial general intelligence, consciousness, or independent scientific replication. Those require stronger tests—especially third-party tasks chosen only after the code is frozen.
