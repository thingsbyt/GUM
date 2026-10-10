# Storage relocation record

The parallel workers increased Windows paging-file allocation on C:, leaving
insufficient headroom for the complete future archive. All eight workers were
suspended inside live joint ticks, away from episode archive/checkpoint writes.
Their in-memory policies, optimizer history, recurrent states and random streams
remained in the same processes. No episode was restarted and no experience was
discarded. Eight team artifact directories were moved intact to the dedicated
`D:/Codex-GUM-evidence/2026-10-10/independent-ppo-development-v1/` directory.
Directory junctions preserve the original study paths. File counts and byte
totals agreed before and after moving; checkpoint and archive hashes were
verified before resuming the same workers.

This is an operational storage change, made while the locked experiment was
running, not a change to hypotheses, information, learning, budgets, seeds,
reward, outcomes or selection. Paused wall time is included in the recorded
episode wall times; those values must not be interpreted as exclusive model
compute. The machine-readable worker states and verification records remain in
`work/ppo-storage-handoff.json`. The plan already disclosed unequal compute.
