# Did GUM learn to learn?

## Two meanings

“Learning to learn” can mean two different things.

**Weak meaning:** old knowledge makes a related task faster. A multiplication table helps with division. This is transfer.

**Stronger meaning:** experience changes the method used to learn new tasks, even when their answers are not remembered. A student learns to draw diagrams because diagrams repeatedly made unfamiliar problems easier. This is meta-learning.

GUM has evidence for the second meaning in a narrow setting.

## The decisive control

The meta-learning experiment stored no task solution, path, state graph, action meaning, or family label. Every new puzzle changed its seed, dimensions, presentation, goal, controls, and dynamics. Only three strategy weights persisted.

The evolved strategy was compared with:

- its original all-zero strategy;
- a shuffled-weight control;
- fresh task learning under the same interaction budget.

Across five independent lives, four evolved learners beat their originals on unseen puzzles. This could not come from replaying an old answer because no answer was retained.

## What actually changed

The three weights change which unexplored state GUM investigates next. Through selection and mutation, the population learned that some mixtures of goal closeness, path depth, and action exhaustion lead to answers with fewer interactions.

That is analogous to changing study habits, not memorizing another fact.

## What did not change

GUM did not invent the ideas of goal closeness, depth, or action exhaustion. It did not rewrite arbitrary source code. It did not grow a new neural architecture. Those are later research targets.

The correct present-tense claim is:

> GUM demonstrated narrow strategy meta-learning: it evolved a persistent acquisition policy that improved learning on unseen tasks without retaining their solutions.

