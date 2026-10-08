# Reproducing the evidence

## Requirements

- Windows, Linux, or macOS
- Python 3.10+
- NumPy, Pillow, SciPy, and scikit-learn
- PyTorch only for older neural experiments

Create an isolated environment if possible:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Core tests

```powershell
python -m unittest `
  tests.test_concept_skill_factory `
  tests.test_concept_genesis `
  tests.test_data_rescue_world `
  tests.test_meta_learning -v
```

Run the complete suite:

```powershell
$env:TEMP = Join-Path $PWD '.test-temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
python -m unittest discover -s tests -p 'test_*.py'
```

The explicit temporary folder avoids a Windows app-sandbox restriction that can otherwise prevent tests from writing to the operating system's default temporary directory. It does not change the learner or its results.

## Correct freeze procedure

Never run an evaluation first and write its rules afterward. Use a new empty output directory.

```powershell
python -m gum.meta_learning_challenge --source . --output .\local-results --protocol-only
python -m gum.meta_learning_challenge --source . --output .\local-results
```

The protocol-only command records thresholds, seeds, withheld information, and source hashes. Do not modify the hashed source between the two commands.

Other entry points:

```powershell
python -m gum.growth_challenge_v2 --source . --output .\growth-results --protocol-only
python -m gum.growth_challenge_v2 --source . --output .\growth-results

python -m gum.concept_genesis_challenge --source . --output .\concept-results --protocol-only
python -m gum.concept_genesis_challenge --source . --output .\concept-results

python -m gum.data_rescue_challenge --source . --output .\data-results --protocol-only
python -m gum.data_rescue_challenge --source . --output .\data-results
```

## What successful reproduction means

A reproduction should report every run, including regressions. It should preserve the generated protocol, audit, ledger, source hashes, software versions, and machine details. Matching the exact random seeds checks determinism; new seeds test generalization.

## Stronger independent test

The preferred next test is not another project-authored world. A third party should choose or generate tasks after receiving a frozen commit hash. The team should not edit the learner after seeing those tasks.

