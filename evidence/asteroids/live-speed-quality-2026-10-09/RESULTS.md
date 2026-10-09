# Live Asteroids speed and quality development result

Date: 2026-10-09

This development pass made the local GPU learner materially faster and gave
the dashboard a genuinely continuous, accurately labeled live view.

## Speed

The same brain, replay, batch size, objective, GPU, and exploration schedule
were used before and after the implementation change.

| Measurement | Before | After |
| --- | ---: | ---: |
| Seconds per update | 0.44136 | 0.10279 |
| Updates per second | 2.27 | 9.73 |
| Relative throughput | 1.00x | 4.29x |

The main changes were a bounded decoded-episode replay cache and a vectorized
GPU random-shift augmentation that avoids per-sample synchronization.

## Frozen quality check

After 20,629 environment decisions and 2,944 gradient updates, an anonymous
action prior was learned by optimizing episode return. The search received no
action names, engine state, object coordinates, or evaluation episodes. The
frozen policy mixed 90% of that learned prior with 10% of the visual DQN.

All policies below were evaluated without replay writes or weight updates on
the same 64 previously unseen seeds, beginning at seed offset 300,000.

| Frozen policy | Mean hits | Mean return | Termination rate |
| --- | ---: | ---: | ---: |
| Visual + learned prior | 7.328 | 7.106 | 9.38% |
| Learned prior alone | 7.000 | 6.301 | 18.75% |
| Constant-fire baseline | 4.094 | 3.393 | 18.75% |
| Uniform random | 4.078 | 2.093 | 43.75% |
| Corrupted control mapping | 1.109 | -2.815 | 81.25% |

The corrupted-map collapse is evidence that the learned anonymous control
mapping matters. The small gain from 7.000 to 7.328 suggests the visual model
helps, but most current performance comes from the state-independent learned
action prior. This is meaningful progress, not evidence of rich visual tactics
or general Asteroids mastery.

## Verification

- Focused learner/dashboard tests: 22 passed.
- Complete repository suite: 370 passed, with one pre-existing test warning.
