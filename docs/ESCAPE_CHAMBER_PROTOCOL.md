# Cooperative escape chamber protocol

## Status and starting state

This is the implementation contract for the four-body GUM cooperation study.
Development began from clean commit
`6489dc1f7f5b2f2f3c261f3a9b95c3cae17d32b3` on `public-main`, synchronized
with `origin/main`. No earlier evidence package is overwritten. Room A is a new
experiment path, not a reinterpretation of the one-body mission-swarm smoke run.

The presentation question and scientific question are deliberately the same:
**will four separately acting learners discover behavior that gets others out,
who remains inside, and how do incentives and experience sharing change that
observable result?** No cooperation result is promised.

## First accepted scope

Room A is the only room in the first implementation gate. Four persistent
identities control four separate bodies through five anonymous action slots.
One semantic mapping—wait plus four cardinal moves—is shared by the team within
an episode and reproducibly shuffled between episodes. Every body selects and
executes its own action. There is no captain, vote, role scheduler, route planner,
or supplied holder.

The room is fully observable through raster images. Each learner sees the room,
all bodies, the visibly depressed plate, and the visibly open or closed gate.
Its own body has a consistent white outline. Coordinates, tile matrices, action
meanings, mechanics, routes, other policies, audit state, and spectator labels
are absent. Full observability is an initial treatment, not a claim of partial
observability.

## Authoritative Room A mechanics

The interior plate is five Manhattan cells from the inside gate approach. The
gate separates the room from a single-file exterior vestibule, followed by an
exit boundary. An escaped body remains visible outside but becomes mechanically
inactive and cannot return.

Each joint tick is resolved in this fixed order:

1. Decode one anonymous action for each active body.
2. Hold gate and exit crossing proposals at their source. Resolve all ordinary
   moves simultaneously. Bodies that request the same destination all wait;
   two-body swaps both wait; moves into a body that is not leaving are cancelled
   repeatedly until stable. There is no identity-priority tie break.
3. Determine plate occupancy after ordinary movement. The gate is open only if
   an active, non-escaped body occupies the plate at that point.
4. Resolve the single-file gate crossing and then the exit crossing. Therefore,
   leaving the plate on the same tick as another body proposes a gate crossing
   closes the gate before that crossing. A different body may replace the holder
   during ordinary movement.
5. Apply rewards, produce consequence pixels, and end at three escapes or the
   declared horizon.

Interpolation and browser timing are spectator concerns and cannot alter these
steps.

### Why four escapes are impossible

A lone body cannot cross the gate because its own gate-cross proposal leaves no
body on the distant plate after ordinary movement. This is exhaustively checked
over every reachable lone-body cell. Three bodies can escape while a fourth
holds the plate; an isolated scripted positive control tests that fact without
entering learner code or training data. After three have crossed, the fourth is
the only non-escaped body and cannot simultaneously occupy the distant plate and
propose the gate crossing. The episode terminates at the legal maximum of three.

## Frozen v1 reward table

Every body active at the start of a tick pays `0.001`. In team mode, every
escape pays `1.0` to every member, including members already outside; reaching
three escapes pays an additional `1.0` to every member, including the holder.
In individual mode, only the newly escaping member receives `1.0`, and there is
no completion bonus. Mixed mode is 50% of the team component plus 50% of the
individual component; its completion bonus is therefore `0.5` per member.

These rewards specify the objective but contain no route, action, holder
identity, intermediate plate bonus, or imitation target. Raw returns are not
compared across treatments as though their meaning were identical.

## Learner input schema

Permitted policy inputs are the member-view `uint8` RGB raster, previous
executed anonymous action, permitted scalar reward, public episode boundary,
the member's own recurrent state, and—only in a later sharing-enabled
treatment—declared bounded experience payloads. The public contract exposes
image shape, action count, horizon, agent count, and protocol version.

The environment audit object contains coordinates and the shuffled action map.
It is private to environment verification. Experimental code must receive only
the restricted observation/transition façade. Trace scanning supplements this
boundary; it is not treated as proof against every indirect implementation bug.

## Engineered versus learned ledger

| Mechanism | Status | Evidence and scope |
|---|---|---|
| Four bodies, collision rules, plate, gate, vestibule, max-three termination | Engineered | Room mechanics and adversarial tests |
| Shared anonymous episode action map | Engineered and randomized | Seeded audit/replay record; never policy input |
| Member colors, shapes, self outline, spectator overlays | Engineered sensor/presentation | Pixel boundary tests |
| Reward treatment and tick cost | Engineered objective | Frozen table above |
| Recurrent pixel policy and generic action memory | Reused engineered architecture | `RecurrentCausalLearner`; no Room A solution |
| Initial weights | Declared per team | Either independent random weights or explicitly identical inherited policy weights; optimizer state is never inherited |
| Action consequences, plate/gate relationship, routes, holder behavior | Learning targets | Must arise from permitted experience |
| Experience sharing | Strictly off in protocol v1 | A later on-treatment must add provenance, fixed budgets, and matched controls |
| Cooperation, burden allocation, transfer, sharing advantage | Unsupported | Require controlled observed results |

## Reuse and gaps

The bounded recurrent learner, atomic storage, hash-ledger, and checkpoint
reconciliation primitives are reusable. The existing `MissionSwarm` controller
is not: it gathers four proposals for one acted body and trains a captain. It
must not drive or be cited as the four-body controller.

The Room A development apparatus now has four independent learner lifecycles,
complete joint transitions and provenance, coherent all-member checkpoints,
strict sharing-off enforcement, live authoritative monitoring, deterministic
replay, and playable GIF recording. Recordings are reconstructed from saved
actions and must match archived pixels, rewards, boundaries, and state hashes at
every joint tick before they are written.

The sharing-on treatment, precommitted evaluation matrix, matched baselines,
multi-seed learning curves, and transfer rooms remain future gates. Until those
are frozen and run, Room A rehearsals are development history rather than a
cooperation or learning breakthrough.

## First learning diagnosis

The first six-episode Room A rehearsal has now been replayed and measured. It
contains one persistent team's history, zero escapes, 72 gate-opening events,
and zero gate-crossing proposals made while the gate was open. Its exact-frame
novelty return is about 9.7 times the magnitude of the external task return,
and its action entropy is 99.9914% of the five-action maximum. Frozen visual
probes do not show consistent improvement over matched birth encoders.

These are measurements, not learner inputs and not proof of a single cause.
They justify a declared easier development room and matched exploration
ablations before adding roles, messages, or a world model. Room A remains the
frozen hard target. See `docs/ESCAPE_CHAMBER_DIAGNOSIS.md` for the full claim
boundary and study design.

## Run and inspect locally

The live viewer creates or reloads one durable four-member team. The page begins
in an explicit non-training preview; learning starts only after the button is
pressed. Use `--device cuda` when CUDA is available.

```bash
python scripts/run_escape_chamber_watch.py --workspace work/escape-chamber-live --device cuda --open-browser
```

Every completed episode is retained. A verified replay of the latest episode
can be made without consulting the policies again:

```bash
python scripts/record_escape_chamber_episode.py work/escape-chamber-live work/room-a.gif
```

The adjacent `room-a.gif.json` identifies the episode, seed, archive hash,
source commit, outcome, and verification status. No failed episode is removed
from the ledger or outcome history.
