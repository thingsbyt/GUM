# Cooperative gradient-attribution study

## Decision

Stop isolated patches and prepare a matched established multi-agent baseline.
The tested correction solved its narrow recorded-context mechanism but produced
no held-out advantage over the original update in any independent team. GUM has
not demonstrated learned cooperation.

Room A was not modified or trained. No role, message channel, world model, new
critic, reward redistribution, semantic target, scripted demonstration, or
human-selected action was added.

## 1. What reproduced

The audited successful episode and its immediately preceding checkpoint were
replayed from preserved pixels, executed anonymous actions, scalar rewards, and
episode boundaries. The original complete actor-critic update reproduced all
four historical post-episode policy tensors exactly before attribution began.
The three additional outcome-selected successes also reproduced their original
post-episode tensors exactly before comparison.

The study plan was committed as `16b8c6f`; attribution code as `10ddc6b`; the
correction protocol as `a4281ea`; and the evaluated implementation as
`1bf9d20`. The study used an RTX 3060 with PyTorch 2.11.0+cu128.

## 2. What attribution establishes

The scaled value gradient was 112–358 times larger than the actor gradient in
the shared visual/recurrent groups. Value-only and actor-only branches moved
recorded-action probability in opposite directions at two of three audited
holder/crosser contexts. Fortis-4's actor/value gradient cosine was negative in
both shared groups. Gradient clipping did not activate.

One additional issue was measured but not selected: Fortis-2's gate action had
a positive raw advantage (`+0.4251`) that became negative after per-episode
normalization (`-0.6906`). This sign reversal occurred at only one of the three
predeclared contexts, so it did not meet the correction rule.

Stateful Adam steps were poorly aligned with the negative instantaneous full
gradient (all-parameter cosine `0.215–0.285`). Because the value-interference
criterion appeared first in the committed hierarchy and passed, update-size or
optimizer changes were not bundled into this study.

These measurements establish objective magnitude imbalance and a real
value-loss effect on specific recorded probabilities. They do not establish
that value interference is the sole cause of failed cooperation, that the
representation encodes the causal relation usefully, or that a probability
change will generalize.

## 3. Single correction

The existing critic was retained, continued to supply the detached advantage
baseline, and continued to train its own head. The only change was setting the
value-loss gradient entering the shared recurrent policy representation to
zero. Actor loss, entropy loss, rewards, returns, optimizer state, learning
rate, architecture size, and selected experience remained unchanged.

This is a narrow form of established actor/value optimization decoupling, not
a novel learning mechanism. Related primary work includes [Phasic Policy
Gradient](https://arxiv.org/abs/2009.04416), [Decoupling Value and Policy for
Generalization in Reinforcement
Learning](https://proceedings.mlr.press/v139/raileanu21a.html), and the broader
multi-objective analysis in [Gradient Surgery for Multi-Task
Learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).

The mechanism check passed: at all three audited contexts, the corrected update
was within `3e-8` of the actor-only action probability, versus original-update
distances of `1.7e-5` to `3.9e-5`. The correction therefore removed the measured
value-loss effect where expected.

## 4. All replications

Each independent four-member team contributed its earliest successful training
episode under an outcome-only selection rule. The 29 preceding training
failures remained preserved. Every branch received 64 matched, previously
unused development-room seeds.

| Team | No update | Original | Corrected | Corrected − original |
|---|---:|---:|---:|---:|
| 1 | 4 escapes | 6 | 6 | 0 |
| 2 | 5 escapes | 5 | 5 | 0 |
| 3 | 1 escape | 1 | 1 | 0 |
| 4 | 2 escapes | 2 | 2 | 0 |
| **Total / 256 episodes** | **12** | **14** | **14** | **0** |

Neither original nor corrected policies reached the legal maximum of three
escapes. Original and corrected updates tied in every team. The exploratory
team-level bootstrap interval for the corrected-minus-original escape-rate
difference was exactly `[0, 0]` because every team difference was zero.

Secondary totals also do not support a cooperative advantage: original versus
corrected produced 38 versus 37 gate crossings, 1,943 versus 1,972 gate-open
ticks, and 1,943 versus 1,972 plate-occupancy ticks. More plate time did not
translate into additional crossings or escapes.

## 5. Retention versus generalization

The correction retained the actor-only tendency at the audited episode's
recorded contexts. That is a successful mechanism/retention diagnostic.

It did not improve held-out escapes over the original update in any team. The
aggregate improvement from no update to either trained branch came entirely
from Team 1; Teams 2–4 were unchanged. This is not repeatable team-level
evidence that one success taught a reusable cooperative strategy.

## 6. Continue/change/stop outcome

The committed rule classifies the result as
`recorded-context-only-retention-generalization-gap`. Do not promote the
correction, claim cooperation, run a sealed examination, or add another
isolated patch. The next protocol should compare GUM with an established
multi-agent learning baseline under the same local pixels, anonymous actions,
team reward, seed blocks, interaction budget, and compute budget. Independent
PPO is the clean first candidate because it does not require privileged global
state or a centralized critic.

## 7. Reproduction and preserved artifacts

```bash
python scripts/attribute_escape_gradients.py work/escape-development-pilot-20261010/replication-1/novelty-off-action-mixer-off 7e8df33d-3876-4908-b2c3-242815b1f210 work/cooperative-gradient-attribution-v1/audited --device cuda
python scripts/run_escape_gradient_correction_study.py work/cooperative-gradient-correction-study-v1 --device cuda
python -m pytest tests/test_school_escape_gradient_attribution.py tests/test_school_escape_gradient_study.py tests/test_school_escape_credit_audit.py tests/test_school_recurrent_meta.py tests/test_school_escape_team.py -q
```

- Preregistered plans and concise evidence:
  `evidence/gum-school/studies/cooperative-gradient-attribution-v1/`
- Full attribution, component checkpoints, and measurements:
  `work/cooperative-gradient-attribution-v1/audited/`
- Full study results, coherent branch checkpoints, recovery record, and all 768
  complete frozen-evaluation transition archives:
  `work/cooperative-gradient-correction-study-v1/`

The 768 archive hashes and shapes were independently rechecked after the run,
covering all 61,440 evaluation ticks. The artifact set is approximately 316 MB.
There was no new environment training in this phase: historical actual training
remains in its original replayable team archives; new policy updates are labeled
offline replay, and all new environment runs are labeled frozen evaluation.
