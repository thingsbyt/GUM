# Contributing

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

