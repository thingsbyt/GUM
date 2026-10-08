# Growing Understanding Machine: Auditable Concept, Skill, and Strategy Growth from Interaction

## Abstract

We present GUM, a compact continual-learning system that converts unlabeled visual transitions into opaque event concepts, compiles successful concept sequences into persistent skills, and evolves a small task-acquisition policy across problems. Unlike foundation-model agents, the evaluated core uses no language model or pretrained semantic knowledge. Frozen internal audits test concept selection, transfer under appearance and control changes, distinct-skill composition, verified JSONL repair, learning-to-learn without solution retention, and grounded clarification from pixels and pointing. Concept Genesis selected five event concepts without receiving the concept count and achieved 20/20 held-out transfer. On a post-training composed world, experienced learning used 83 interactions versus 41,339 for a perception-matched blank learner. The unchanged concept-learning implementation discovered a real-file repair workflow and transferred it to 5/5 unfamiliar inputs, requiring 10 interactions versus 3,480 for a matched blank learner. Five independent strategy-evolution runs covered 160 training and 120 held-out puzzles from four dynamics families. Four runs beat their initial acquisition policy; median held-out speedup was 2.30×, mean speedup was 4.48×, and zero task solutions were retained. A separate grounded-language audit learned six visual words from demonstrations and resolved 30/30 ambiguous references through questions. These results demonstrate narrow, auditable cumulative and strategy learning in controlled environments. They do not establish unrestricted understanding or state-of-the-art performance.

## 1. Motivation

Most successful modern agents begin with large pretrained models. GUM explores a complementary question: how much adaptive behavior can emerge from a small, explicit memory that begins without semantic world knowledge?

The design goal is not biological imitation. It is cumulative competence with inspectable evidence. A useful result must distinguish learning from replay, prevent hidden-state leakage, survive reload, and compare experienced learning with a matched blank learner.

## 2. System

GUM observes RGB frames, anonymous actions, scalar outcomes, and termination. Concept Genesis learns pixel-change tokens, constructs event-level feature distributions, compares candidate clusterings, and assigns opaque concept identifiers. Successful action sequences are rewritten as concept sequences and stored as skills. New control mappings are grounded by intervention. Skills can then be executed, shared without local mappings, or composed.

The meta-learning extension retains no task solutions. A population evolves three weights used by a generic best-first acquisition process: visual goal distance, path depth, and action-exhaustion preference. Fitness is measured by interactions required to solve prior tasks.

## 3. Experimental controls

Each primary audit writes its protocol before execution, including seeds, thresholds, learner inputs, withheld fields, and source hashes. The evaluated source must match before and after the run. Learner traces contain observation hashes, anonymous actions, rewards, and termination; private mechanism identities and solutions remain on the audit side. Hash-chained ledgers detect record alteration or reordering.

Baselines include a prior handwritten visual signature, a perception-matched learner without skills, original unevolved acquisition weights, and shuffled learned weights.

## 4. Results

The Growth Challenge demonstrated persistent opaque concepts, skill transfer, sharing, and distinct-skill composition. Concept Genesis then removed the old five-way event signature. The learned partition distinguished five mechanisms where the old signature distinguished three.

The data-rescue audit used actual JSONL working files and independent verification. Outputs required canonical fields, normalized values, integer types, deduplication, sorting, and stable record identifiers. Original inputs remained byte-identical.

The meta-learning audit produced heterogeneous results rather than universal improvement: four seeds improved and one regressed. This supports adaptation while warning against overgeneralization from a single successful run.

The grounded-clarification audit supplied rendered pixels, utterances, and pointer locations but no word-to-feature dictionary. Cross-situational statistics induced six visual words. The learner resolved 30/30 ambiguous requests after asking a discriminating question, handled 15/15 clear requests directly, and asked for teaching rather than guessing an unknown word. This is a small communication bridge, not broad language understanding.

A browser-based local Studio exposes these mechanisms to non-programmers. Its teaching surface separates demonstrations, induced mappings, dialogue turns, and fresh-learner audits. This interface is not additional evidence by itself; it is an instrument for inspecting and reproducing evidence.

## 5. Relationship to prior work

GUM relates to meta-learning such as [MAML](https://proceedings.mlr.press/v70/finn17a.html), temporal abstraction through [options](https://incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/Options.pdf), learned symbolic libraries in [DreamCoder](https://arxiv.org/abs/2006.08381), unsupervised representations in [Slot Attention](https://arxiv.org/abs/2006.15055) and [I-JEPA](https://arxiv.org/abs/2301.08243), world-model learning in [DreamerV3](https://arxiv.org/abs/2301.04104), and persistent agent skills in [Voyager](https://arxiv.org/abs/2305.16291). GUM's proposed contribution is not any single ingredient, but a compact explicit combination with frozen, inspectable evidence.

## 6. Limitations

The worlds and interfaces were authored by the project. Changed pixels are assumed to be informative. The data task provides five safe operations. The visual-language study uses three colors, three shapes, and one action family. The meta-strategy has only three engineered features. Comparisons against established external systems are not yet available. Five meta-seeds are insufficient for a definitive statistical claim, and one seed regressed. No independent replication has occurred.

## 7. Conclusion

GUM provides evidence that a small explicit learner can grow useful concepts, procedural memory, and an improved acquisition policy from interaction. The strongest current statement is narrow: GUM demonstrated strategy meta-learning without retaining task solutions in controlled pixel-goal puzzles. External task ownership, stronger baselines, more seeds, and less engineered perception are required before claiming a field-level breakthrough.

