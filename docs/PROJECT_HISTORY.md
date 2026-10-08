# Project history: how GUM got here

This is a conceptual history, not a claim that every experiment belongs to one unchanged implementation.

## 1. Start with embodied learning

The project began from an Asteroids-style JEPA reference. The useful idea was to give an agent eyes and consequences instead of a table of correct actions. Early work established frame-based control, learned representations, recordings, and evaluation discipline.

## 2. Test more than one game

Maze, changing-control, delayed-chain, deceptive, noisy, and partially observable worlds exposed a central problem: success in one environment could be memorization. The tests increasingly remapped controls and withheld solutions so transfer mattered more than one score.

## 3. Add persistent skills

Successful action chains became stored procedures. Retention and revisiting tests asked whether old abilities survived and whether encountering an old task could improve it instead of spawning unnecessary duplicates.

## 4. Separate concepts from names

The learner moved from hand-described events toward opaque event identifiers formed from visual changes. Concept Genesis then withheld even the number of event types and selected the partition from evidence.

## 5. Make old knowledge useful in new tasks

Composition tests required distinct older skills to solve a fifth problem faster than a matched blank learner. Shared-memory tests asked whether another agent could benefit without inheriting local control mappings.

## 6. Learn how to investigate

The learning-to-learn study discarded task solutions but retained a tiny acquisition strategy. Across five independent lives, four improved their held-out efficiency and one regressed. Keeping the regression made the result more believable and more useful.

## 7. Leave games

Data Rescue applied the same experience → concept → skill loop to actual JSONL files through five bounded, safe operations. It produced independently verified outputs while leaving originals unchanged.

## 8. Build a communication bridge

Grounded language tests paired pixels, utterances, and pointing. The current clarification study learns six visual words, asks a useful question when several objects match, and requests teaching for an unknown word.

## 9. Make the work inspectable

GUM Studio now provides live world learning, a teaching lab, a claim ledger, replays, machine state, and documentation in one local interface. Frozen evidence remains separate from new interactive runs.

## 10. The next scientific step

The next threshold is not another handcrafted success demo. It is a frozen agent and interface evaluated by outsiders on unrelated environments chosen after freezing, with matched fresh controls, preregistered outcomes, and all failures preserved.

