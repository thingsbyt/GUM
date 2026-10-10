# Execution-only amendment

Registered after the initial GUM team began, before any remaining independent
case was launched. The initial GUM case continues to its fixed final checkpoint.
The remaining seven pristine cases execute in separate CPU processes, each
with one PyTorch thread, to use available cores. This amendment is motivated by
measured wall-time throughput, not comparative outcomes. It changes no learning
algorithm, model configuration, seed, schedule, budget, checkpoint, outcome,
success threshold or tuning allowance. Compute was not equalized in the plan.

The runner now accepts a team/method selector and postpones aggregate analysis
until all eight cases finish. Each selected case executes the identical inner
loop. Source commits and dependency provenance are retained per case. Wall time
includes concurrent scheduling overhead and is not an equal-compute comparison.

After the initial process finishes GUM team 1, it is stopped upon first entering
its duplicate PPO pre-training evaluation, before any duplicate PPO training.
The actual PPO team 1 is a fresh case in its separate committed process. Any
completed duplicate evaluation prefix is retained, labeled excluded, and its
extra evaluation interactions/compute reported. It does not enter outcome
comparisons or tuning. No interrupted training episode is discarded or rerun.

All complete primary cases are merged by team/method identity. An assertion
requires exactly eight distinct cases and all fixed checkpoints. The original
GUM code, PPO code and room remain frozen. A separate viewer maintenance change
handles ordinary disconnected image requests and suppresses overlapping polling;
it does not enter the environment or optimizer loop.
