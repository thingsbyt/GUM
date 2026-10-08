# Grounded clarification test

This is a small post-`v0.1.0` experiment. The frozen release tag remains unchanged.

## Question

Can GUM notice that a human instruction refers to more than one visible object, ask a question that separates the candidates, understand the answer, and then act—without guessing?

## What the learner receives

- rendered pixels;
- phrases paired with a person pointing to the intended object;
- a later instruction and the human's clarification answer.

It receives no color dictionary, shape dictionary, pretrained language model, object labels, or test answers. The action family—approach a selected object—is fixed so the experiment isolates reference learning and dialogue.

## How it works in plain language

Across demonstrations, the word `red` occurs with differently shaped red objects. The word `circle` occurs with differently colored circles. GUM looks for the visual feature that stays dependable whenever each word is used. Common words such as `the` occur with everything and therefore acquire no visual meaning.

During a test, `approach the red object` may identify both a red circle and a red square. Selecting either would be an unsupported guess. GUM looks for a learned feature that divides the candidates and asks, `Which one do you mean: the circle or the square?` The answer becomes an additional constraint. Action occurs only when one candidate remains.

## Success criteria

- resolve every ambiguous reference after clarification;
- ask a genuinely discriminating question in every ambiguous trial;
- execute unambiguous requests without unnecessary questions;
- refuse an ungrounded word instead of guessing;
- retain the learned visual lexicon after saving and reloading.

Run it with:

```powershell
python -m gum.clarification_challenge --output .\clarification-results
```

The output contains a readable transcript, the learned opaque visual mappings, a machine-readable audit, and file hashes.

## What it would mean

Passing demonstrates bounded pragmatic communication: the system connects learned words to perception, detects referential uncertainty, requests useful information, incorporates a reply, and acts. It does not demonstrate open-ended English understanding or human-level conversation.
