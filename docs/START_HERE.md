# Start here

GUM is a research prototype for a specific question: can a compact machine turn experience into reusable concepts and skills, retain them, and improve the way it approaches later tasks?

The quickest honest tour takes about ten minutes.

1. On Windows, double-click `START_GUM_STUDIO.bat`. On macOS or Linux, run `./START_GUM_STUDIO.sh`.
2. Open **Teaching lab** and press **Load starter lessons**. GUM learns six visual words from rendered pixels, sentences, and pointing demonstrations.
3. Press **Open ambiguity test**, send `approach the red object`, and answer GUM's question with `the square`.
4. Press **Run fresh 45-trial audit**. This discards the interactive learner, trains a fresh one using the documented procedure, and reports the score.
5. Open **Live lab**, create a world, record the starting result, run bounded learning, and compare the result afterward.
6. Open **Evidence** to see real recordings and the claim ledger. Gold means a mixed result, not a hidden failure.

## The important distinction

The Studio contains three different things:

- **Live experiments** create new evidence in your local `.gum-workspace`.
- **Teaching lab** lets you supply demonstrations and inspect the learned vocabulary.
- **Frozen evidence** records earlier experiments whose exact protocols and outputs are included in the repository.

A replay is an illustration. An audit file is evidence. Neither is independent replication.

## What to read next

- [Playground guide](PLAYGROUND_GUIDE.md) for the interface
- [Capabilities](CAPABILITIES.md) for the complete ability map
- [Architecture](ARCHITECTURE.md) for how the machine grows
- [Evidence](EVIDENCE.md) for claim-to-file traceability
- [Learning integrity](LEARNING_INTEGRITY.md) for leaks, hardcoding, and what “learns” means
- [GUM School curriculum](GUM_SCHOOL_CURRICULUM.md) for the pre-training lesson, transfer, retention, and promotion specification
- [Preliminary paper](PRELIMINARY_PAPER.md) for the research argument
- [Claims and limitations](CLAIMS_AND_LIMITATIONS.md) for the boundaries

