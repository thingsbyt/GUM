# Human–AI collaboration disclosure

GUM was created through unusually intensive human–AI collaboration.

**Tolga Uz** originated and directed the project: setting the goal, choosing what mattered, challenging weak results, requesting new environments and controls, deciding which directions to pursue, and taking responsibility for the public release.

**OpenAI's ChatGPT and Codex** served as continuous research-and-engineering collaborators. Across many iterative sessions, the systems helped explore architectures, implement and revise code, design adversarial tests and controls, run experiments, analyze failures, produce visual demonstrations, organize evidence, and draft documentation.

This disclosure is intentionally prominent because the development process is part of the project's story and should not be mistaken for unaided human authorship.

## Important scientific distinction

ChatGPT/Codex helped **build and evaluate** GUM. They are not silently supplying answers inside the core experiments. Unless an individual protocol explicitly says otherwise, the evaluated GUM learner does not call an LLM, retrieve ChatGPT output, or receive hidden solutions during its learning loop. Learner inputs and withheld information are listed in each protocol.

AI assistance does not make OpenAI a project author, copyright owner, sponsor, endorser, or independent validator. Generated suggestions and code were subject to human direction, executable tests, frozen protocols, evidence review, and the limitations described in this repository.

For citation purposes, the public project author is **Tolga Uz**. The collaboration is acknowledged here, in the README, citation metadata, and Apache NOTICE file.
