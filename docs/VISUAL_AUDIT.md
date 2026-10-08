# Visual audit

Audit date: 2026-10-08

Method: the start, middle, and final frame of every GIF in `assets/demos` was extracted and inspected at native resolution. Each asset was also assessed at its intended README or Studio placement. This is a visual-quality audit, not a scientific re-analysis; quantitative claims still come from the linked JSON evidence.

## Results

| Asset | Native size | Status | Inspection notes |
|---|---:|---|---|
| `two-rocket-asteroids.gif` | 640×226, 121 frames | Pass | Both local-camera halves, rockets, projectiles, asteroids, and HUD remain distinct. Start/mid/end show meaningful progression. Best used full-width. |
| `maze-escape.gif` | 474×224, 96 frames | Pass | Maze walls, agent, goal, step count, memory growth, and ESCAPED state are legible. No overlap or clipping. |
| `logic-chain.gif` | 494×224, 16 frames | Pass | Trial, inferred prefix, event diagram, and SUCCESS state remain readable. Light reveal effect is intentional. |
| `json-normalizer.gif` | 900×470, 8 frames | Pass | Workflow title, anonymous control, observed result, progress states, and VERIFIED COMPLETE ending are crisp. |
| `cooperative-rescue.gif` | 448×300, 59 frames | Pass | Two panels, agents, resources, mode, recipe, and reward progression are visible. It is captioned as fictional and non-medical. |
| `lifelong-showcase.gif` | 480×184, 72 frames | Pass | CATCH/AVOID/NAVIGATE lanes and moving objects are clean. This is a conceptual montage, so it is not used as standalone quantitative evidence. |
| `apple-concept.gif` | 768×430, 24 frames | Pass with editorial condition | Photos, candidate words, selection highlight, reward, and decision-chain note are legible. It must remain paired with the 53.22% overall mixed result; the README and gallery do so. |

## Issue log

- No corrupted frames, unintended transparency, cutoff text, or obvious rendering artifacts were found.
- Several dark-background clips use small status text. They remain readable at native or full-width display, but should not be used as tiny thumbnails without captions.
- The cooperative rescue recording could be mistaken for a biological simulation without context. Every public placement therefore labels it fictional.
- The apple recording could overstate the broader experiment if isolated. Every public placement therefore links or states the failed overall threshold.

Final verdict: **all seven assets are approved for the documented placements, subject to the two editorial conditions above.**

