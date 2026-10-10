# Cooperative mission swarm

The mission swarm is persistent biological scaffolding around GUM's general
recurrent learner. It is not a maze specialist and it does not contain a
route planner, action dictionary, world model supplied by the environment, or
authored solution.

## Four durable members

The team has exactly four independently parameterized learners:

- **Fortis-1 / pathfinder** favors lower-temperature exploitation.
- **Fortis-2 / challenger** explores more broadly.
- **Fortis-3 / modeler** places still more weight on unfamiliar actions.
- **Fortis-4 / keeper** uses the broadest action exploration.

Each member receives a UUID, creation time, parent, generation, seed, role,
and SHA-256 fingerprint at birth. The fingerprint detects accidental or local
record changes; it is explicitly not a cryptographic identity signature.

## Everything is a mission

All four members see the same pixels, selected anonymous action, scalar reward,
and termination signal. All four propose an action. During learning, a rotating
captain acts; during evaluation, a confidence-weighted vote acts. No learner
receives coordinates, action meanings, event labels, routes, audit state, or
private generator data.

When a mission succeeds, its reward-grounded trajectory is rehearsed by the
three non-captains. When it fails, the failure remains in the knowledge ledger
and the next exact attempt must rotate to a different captain and exploration
profile. Failure is evidence, not a cue to rerun an identical strategy.

## Nothing learned is discarded

After every mission, including failure, the system durably preserves:

1. the complete public transition sequence as compressed data: pre-action and
   next observations, actions, rewards, termination/truncation flags, and all
   four proposals;
2. all four post-mission neural checkpoints;
3. a compact experience capsule;
4. append-only hash-chained mission and knowledge records; and
5. hashes that are verified when the team is reloaded.

The archive is local and auditable. Its hash chain detects modification while
its anchor remains trusted, but it is not a remote notarization or an identity
signature.

Reload reconciles the root checkpoint pointer with both anchored ledgers. If a
process stops after committing a mission but before replacing the pointer, the
single complete checkpoint matching the durable ledger heads is recovered and
the pointer is repaired. If no unique coherent checkpoint exists, reload stops
with an explicit inconsistency error rather than combining state from different
moments. Experience files created before the v2 transition format are retained
as historical pre-action traces; new archives are required to include terminal
next observations and per-step termination/truncation flags.

## First architecture smoke run

The corrected v2 smoke run used one new causal mission and four distinct new
maze missions. Captaincy reached every member. The team completed the causal
mission and one maze, shared both successful trajectories to the other three
members, preserved all three failures, and passed a save/reload integrity
check over all five experience files and both ledgers.

That demonstrates the structure works. It does **not** establish that four
members outperform one, that communication caused either success, or that GUM
has mastered mazes. Those claims require a frozen controlled evaluation later.

Primary evidence:

- [`cooperative-mission-swarm-v2/MISSION_SWARM_SMOKE_REPORT.json`](../evidence/gum-school/research/cooperative-mission-swarm-v2/MISSION_SWARM_SMOKE_REPORT.json)
- [`gum/school/mission_swarm.py`](../gum/school/mission_swarm.py)
- [`tests/test_school_mission_swarm.py`](../tests/test_school_mission_swarm.py)

The scaffolding is consistent with recurrent meta-learning, episodic
exploration, and pixel-grounded curiosity, but the current smoke result is a
project engineering result rather than a new scientific claim.
