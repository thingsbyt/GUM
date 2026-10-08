# GUM untouched dual-gate results

## Verdict

Both frozen tests passed on 20 new worlds. The seeds were not used in development and do not overlap the first failed reveal gate.

## Repaired single mind

- First encounters: 20/20 successful, zero protected-object damage
- Mean first-encounter cost: 293.40 interactions
- Fresh isolated minds: 486.75 interactions
- Cross-world experience advantage: 39.72% fewer interactions
- Revisit cost: 38.15 interactions, a 7.69x speedup
- Reloaded memory: 8/8 successful
- Automatic recovery branches: 4

This validates the negative-transfer repair on untouched worlds. The mind retained shared knowledge, detected four stalled contexts, isolated them, and completed every task safely.

## Two-mind collective

- First encounters: 20/20 successful, zero protected-object damage
- Mean first-encounter cost: 261.30 interactions
- Each world was initially experienced by only one member
- The opposite member later solved all 20 worlds using exchanged learned memory
- Cross-owner revisit cost: 38.40 interactions
- Cross-owner advantage over fresh learning: 12.68x
- Reloaded collective: 8/8 successful
- Twenty explicit experience exchanges
- Parallel curriculum advantage over the single mind: 1.79x
- Interaction cost was 10.94% lower than the single mind

## Integrity

- Frozen implementation hashes were unchanged before and after both tests.
- The single-mind trace contains 6,631 valid hash-linked transitions.
- The collective trace contains 5,994 valid hash-linked transitions.
- Hidden-state scan found no control maps, recipes, simulator coordinates, or audit state in either learner trace.
- Random controls solved 0/20 worlds.
- Thirteen focused implementation and integrity tests pass.

## Interpretation

This is strong internal evidence for persistent learning, recovery from negative transfer, knowledge exchange between two embodied learners, and faster learning from accumulated experience. It is now suitable for a controlled public technical demonstration.

It is not yet sufficient for a scientific-breakthrough claim. The two world families share an underlying visual-composition structure, and the evaluation was performed by the project authors. A breakthrough claim still requires more unrelated external environments, stronger published baselines, and independent replication.
