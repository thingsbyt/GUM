# Concept Bridge: apple recognition result

## Outcome

The learned system identified **960 of 1,000 unseen apple trials (96.0%)**. Chance was 10%. Each trial used one of 100 held-out apple photographs, an independently restyled set of ten word representations, and shuffled candidate locations.

This meets the precommitted apple threshold of 90%. It does **not** justify saying “every possible apple, every time.” The scientifically supported claim is 96% on this defined held-out test.

## What it learned

The neural networks began with random weights. Training supplied two kinds of trial-and-reward experience:

1. choose which anonymous symbol matches a photograph;
2. choose which written word matches that successful symbol.

The system was never trained on photograph-to-word pairs. At use time it composes the two learned skills: photograph → internal symbol → written word.

## Evidence

- Training photographs: 5,000 from the official CIFAR-100 training split.
- Test photographs: 1,000 from the official CIFAR-100 test split, 100 per concept.
- Test trials: 10,000, because every photograph was tested against ten unseen representation styles.
- Apple: 960/1,000 (96.0%).
- Photograph → anonymous symbol: 88.71% overall.
- Composed photograph → word: 53.22% overall.
- Fresh untrained network: 10.0%.
- Intentionally wrong mapping: 5.98%.
- Recorded apple replay: 22/24 correct.

The overall ten-concept word result failed its 80% threshold because the held-out typography bridge generalized poorly for several non-apple words. That failure is retained in the audit; it is not hidden. The earlier v21 run is also preserved and failed at 56% on apple, demonstrating that the final 96% result came from a substantive architecture and reasoning-chain repair.

## Boundaries

This is a learned, closed-set visual concept system—not general visual understanding. It must choose among apple, bicycle, butterfly, clock, dolphin, lamp, pickup truck, pine tree, telephone, and wardrobe. It does not yet know when an image contains none of them.

## Reproduce or use it

Training and evaluation implementation: `jepa_asteroids/concept_bridge.py`

Ordinary file inference implementation: `jepa_asteroids/concept_bridge_identify.py`

Checkpoint: `CONCEPT_BRIDGE_BRAIN.pt`

Machine-readable audit: `CONCEPT_BRIDGE_AUDIT.json`

Visual replay: `APPLE_CONCEPT_REPLAY.gif`
