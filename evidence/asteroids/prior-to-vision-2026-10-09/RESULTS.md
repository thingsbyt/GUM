# Prior-to-vision curriculum: development attempt

Date: 2026-10-09

Status: **failed; not promoted**

## Question

Can the stable pixel controller use the outcome-trained action prior as
temporary exploration scaffolding, then outperform that prior after its
influence is linearly reduced to zero?

## Protocol

- Preserve the pre-curriculum checkpoint at update 2,944.
- Run 192 training episodes with prior weight decreasing from 0.90 to 0.00.
- Continue Double DQN and auxiliary JEPA training only on real transitions.
- Calibrate the visual-only action temperature on 24 disclosed development
  seeds without replay writes or weight updates.
- Freeze the selected rule and evaluate all policies on the same 64 untouched
  seeds, including occluded pixels, shuffled frame order, and corrupted control
  mapping.
- Promote only if paired-bootstrap 95% intervals show that visual-only beats
  the learned prior, beats complete occlusion, and depends on the control map.

## Training

| Measurement | Before | After |
| --- | ---: | ---: |
| Environment decisions | 20,629 | 43,611 |
| Gradient updates | 2,944 | 6,016 |
| Curriculum episodes | 0 | 192 |
| Prior weight | 0.90 | 0.00 |

## Frozen 64-seed result

| Policy or ablation | Mean hits | Mean return | Termination rate |
| --- | ---: | ---: | ---: |
| Learned prior | 6.422 | 5.640 | 20.31% |
| Visual-only selected | 4.297 | 2.717 | 35.94% |
| Visual greedy | 2.344 | 0.513 | 40.63% |
| Fully occluded visual input | 4.953 | 3.613 | 31.25% |
| Shuffled frame order | 4.125 | 2.942 | 28.13% |
| Corrupted visual control map | 2.578 | -0.375 | 62.50% |
| Uniform random | 4.781 | 3.358 | 32.81% |
| Constant fire | 4.281 | 3.584 | 18.75% |

Visual-only trailed the learned prior by 2.125 hits; the paired-bootstrap 95%
interval was [-3.063, -1.156]. It also failed to beat complete visual
occlusion: the difference was -0.656 hits with interval [-1.531, 0.219]. The
control mapping still mattered, but that gate alone is insufficient.

## Decision

The candidate failed the competence and pixel-use gates. It was retained as a
diagnostic 6,016-update milestone and not promoted. The active learner was
rolled back to the preserved 2,944-update checkpoint. This result argues
against further plain off-policy Q-learning on prior-generated trajectories;
the next attempt should change the learning objective rather than merely add
more episodes.
