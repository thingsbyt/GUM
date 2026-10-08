# Learning integrity: leaks, hardcoding, and what “learns” means

## Short answer

GUM genuinely learns **within the tested task families**: experience changes persistent internal state, and that state improves later behavior on held-out or remapped tasks relative to appropriate controls.

This is not the same as saying that nothing is engineered. GUM is software. Its learning rules, representations, action interfaces, thresholds, safety boundaries, and rewards are designed by people. The experimental claim is that the particular answers being measured were withheld and acquired through interaction.

## Engineered versus learned

| Engineered | Learned in the relevant experiments |
|---|---|
| Sensors, anonymous action count, reward and termination channels | Which anonymous control causes which observed effect |
| Clustering, search, memory, and strategy-evolution algorithms | Opaque event groupings and selected concept count |
| Confidence thresholds and exploration budgets | Successful concept/action sequences compiled as skills |
| Safe file operations available to Data Rescue | Which operations matter and their effective order |
| The `approach` action family in the clarification test | Visual meanings of six words and which clarification resolves a request |
| Three features available to the meta-learner | Their evolved acquisition weights |

## Why this is more than replay

1. **Matched blank controls:** experienced and blank learners share perception and algorithms; retained experience is the controlled difference.
2. **Held-out variation:** appearance, controls, goals, dimensions, and dynamics change across evaluation tasks.
3. **Remapped controls:** successful behavior cannot rely only on a fixed button sequence.
4. **Persistence and reload:** acquired state is saved, reopened, and retested.
5. **Source freezing:** major audits hash the evaluated implementation before and after execution.
6. **Learner/auditor separation:** hidden mechanisms, private coordinates, validators, and solutions stay on the audit side; primary traces are scanned for forbidden private fields.
7. **Ablations and failures:** blank, random, shuffled, solo, no-message, and earlier failed results are preserved where applicable.
8. **No solution retention in meta-learning:** the strategy study records zero retained task paths while its three acquisition weights change.

## Publication leak audit

Run:

```bash
python scripts/audit_release_integrity.py
```

It scans ordinary repository files and text members inside evidence ZIPs for local home paths, common credential forms, and private-key headers. It also rechecks the strongest machine-readable evidence gates: frozen source, clean learner traces, unchanged source data, zero retained meta-learning solutions, and the absence of a preinstalled dictionary or language model in the clarification test.

The public-preview evidence archives have release-machine path prefixes replaced by neutral placeholders. The byte-exact pre-sanitation history remains in the private/local v0.1 Git history bundle; it is not required for using the public-preview audits.

## What cannot honestly be guaranteed

No finite audit proves the absence of every possible implementation bug, indirect cue, or task-author bias. The environments were created during this project, and external researchers have not yet independently replicated the results. Therefore the defensible statement is:

> The included source, controls, traces, tests, and audits support genuine experience-dependent learning in GUM's documented bounded tasks; no known answer leakage or credential/private-path leak remains in the public-preview tree.

The next confidence jump must come from outside task authors choosing sealed tasks after the learner and protocol are frozen.
