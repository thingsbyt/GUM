# Playing with, teaching, and testing GUM

GUM Studio is a local browser interface. It does not send experiments to a cloud service. Its working memory lives in `.gum-workspace`, which is deliberately excluded from Git.

## One-click launch

### Windows

Double-click `START_GUM_STUDIO.bat`. The first launch creates a private Python environment and installs the declared dependencies. Later launches reuse it. Close the terminal window to stop GUM.

### macOS or Linux

Run:

```bash
chmod +x START_GUM_STUDIO.sh
./START_GUM_STUDIO.sh
```

For a manual start:

```bash
python -m pip install -r requirements.txt
python -m gum --workspace .gum-workspace serve --release . --open-browser
```

## Teaching lab

There are two ways to teach.

**Guided lesson.** Press **Load starter lessons**. Behind the button are 18 pointing demonstrations covering nine color/shape combinations. The learner is not passed a word dictionary. It measures anonymous visual features and uses cross-situational statistics to find the words that consistently travel with those features.

**Your own examples.** Type a sentence, click its object, move to another lesson scene, and repeat. Then press **Learn meanings**. A word is accepted only after at least three examples, at least 80% consistency, and at least 2× lift over the feature's background frequency. These thresholds are inspectable in `gum/clarification_challenge.py`.

After teaching, open the ambiguity test. The scene has two red objects. Ask for `the red object`. If both match, GUM finds a learned feature that divides the candidates and asks which one you meant. Your answer is evaluated only against the candidates from the unresolved request.

The **fresh 45-trial audit** is stronger than a manual demo: it trains a new learner, evaluates 30 ambiguous scenes and 15 clear scenes, checks an unknown-word refusal, writes an audit file, and reports the baseline comparison.

## Live lab

The Live lab is for action learning rather than language learning.

1. Ask GUM to create a world.
2. Set a bounded episode budget.
3. Run learning explicitly.
4. Compare the before and after evaluation.
5. Inspect the machine state and hash-linked lineage.

Worlds expose pixels, anonymous actions, scalar reward, and termination. Private coordinates and action meanings remain on the audit side. A dropped world cannot execute arbitrary code; adapters must be registered in the harness.

## What counts as teaching?

Showing a labeled example is teaching. Writing the answer directly into a lookup table is not learning. GUM's teaching interfaces preserve the distinction by logging the observation, utterance, pointer, learned mapping, test decision, and outcome separately.

## Resetting

**Reset teaching memory** clears only the interactive visual-word learner. Deleting `.gum-workspace` clears all local Studio experiments; it does not change the frozen evidence stored in the repository.

