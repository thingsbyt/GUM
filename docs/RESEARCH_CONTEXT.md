# Research context and citations

GUM combines ideas from several established fields. The combination and audit style may be distinctive; the ingredients do not come from nowhere.

## Meta-learning

Meta-learning asks whether experience across tasks can make a learner adapt faster to a new task. Model-Agnostic Meta-Learning optimizes neural parameters so a few later gradient updates work well. GUM instead evolves a tiny explicit acquisition policy.

- Finn, Abbeel, and Levine, [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://proceedings.mlr.press/v70/finn17a.html), ICML 2017.

## Temporal abstraction and skills

The options framework formalized reusable, temporally extended actions. GUM's compiled concept sequences play a related engineering role, although they are much simpler than the full options framework.

- Sutton, Precup, and Singh, [Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning](https://incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/Options.pdf), Artificial Intelligence 1999.

## Learned symbolic libraries

DreamCoder grows a library of symbolic abstractions while learning programs. It is an important precedent for the idea that learned concepts can become a language for solving later problems.

- Ellis et al., [DreamCoder: Growing Generalizable, Interpretable Knowledge with Wake-Sleep Bayesian Program Learning](https://arxiv.org/abs/2006.08381), 2020.

## Unsupervised object and concept representations

Slot Attention shows that structured object-like representations can emerge without object labels. GUM's present event concepts are transition categories, not full objects.

- Locatello et al., [Object-Centric Learning with Slot Attention](https://arxiv.org/abs/2006.15055), NeurIPS 2020.

## Predictive visual representations

I-JEPA learns semantic image representations by predicting target representations from context. GUM's current Concept Genesis path is not an I-JEPA model, but predictive representation learning remains relevant to the planned natural-scene upgrade.

- Assran et al., [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243), 2023.

## World-model reinforcement learning

DreamerV3 demonstrated one configuration across more than 150 tasks and learned difficult Minecraft behavior from pixels and sparse rewards. It is an essential performance baseline for any future claim of broad pixel-based learning.

- Hafner et al., [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104), 2023.

## Open-ended skill libraries

Voyager uses an LLM, automatic curriculum, environment feedback, and a persistent executable skill library in Minecraft. GUM differs by being small, non-LLM in its core loop, and explicitly auditable; Voyager is far more capable in a rich open world.

- Wang et al., [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291), 2023.

## Generalist interactive agents

SIMA learns across several commercial games using large pretrained models and human-generated data. GUM is not yet comparable in breadth, visual complexity, or language following.

- Google DeepMind, [A Generalist AI Agent for 3D Virtual Environments](https://deepmind.google/blog/sima-generalist-ai-agent-for-3d-virtual-environments/), 2024.

## Positioning

No source above should be described as “the same as GUM.” Likewise, GUM should not be described as replacing these systems. The research question is whether a compact, explicit, persistent, and easily audited learner can obtain useful continual and meta-learning behavior without relying on a massive pretrained model.

