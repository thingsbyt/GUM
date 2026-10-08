# Contributing

## GUM School specifications

Changes to the curriculum, lessons, examinations, or world-admission records
must keep the schemas strict and pass both validation layers:

```powershell
python -m gum.school curriculum/gum-school-v1.json
python -m pytest -q tests/test_school_curriculum.py tests/test_school_engine.py tests/test_school_worlds.py
python scripts/admit_school_worlds.py --trials 24
```

Do not add sealed seed lists, action maps, answer keys, successful sequences, or
other task solutions to the public tree. A new adapter identifier is not an
admitted adapter: source review, deterministic reset, hard horizon, leakage
checks, nontrivial controls, replay, disjoint seed partitions, and hashed
evidence are required before its admission status can change.

GUM values reproducible failures as much as successful demonstrations.

Before proposing a change, open an issue describing the behavior, world contract, withheld information, baseline, success criterion, and expected compute. A new capability claim should include a frozen protocol, machine-readable audit, random seeds, control or ablation, and an explicit limitation statement.

Keep the learner/evaluator boundary visible. Do not move simulator-only state, solutions, semantic action names, or validators into learner inputs. Tests should fail if hidden information crosses that boundary.

For code changes:

1. Use a focused branch.
2. Add or update tests.
3. Run the verification commands in `docs/REPRODUCING.md`.
4. Update the evidence catalog if a claim changes.
5. Never rewrite a frozen evidence artifact without creating a new protocol and version.

Unless explicitly marked otherwise, contributions intentionally submitted for inclusion are accepted under the repository's Apache License 2.0, as described in Section 5 of that license.

