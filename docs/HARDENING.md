# Hardening notes

This page records the engineering safeguards added after the first public
preview. They improve reliability and auditability; they do not turn a research
prototype into a security boundary for hostile code or files.

## GUM Studio

- Binds only to a loopback address.
- Generates a fresh, unpredictable session token at launch.
- Requires that token for every request, including media.
- Rejects foreign `Host` and `Origin` values.
- Accepts mutation bodies only as bounded JSON.
- Adds browser security headers and serializes access to mutable learner state.

These checks protect a local Studio from ordinary cross-site request attacks.
GUM Studio is still a local research tool, not an internet-facing service.

## Persistent state

- Important JSON, text, and binary snapshots are written to a temporary file,
  flushed, and atomically replaced.
- The previous complete snapshot is retained as an adjacent `.bak` file.
- Hash-linked ledgers now maintain an independent adjacent anchor containing the
  expected entry count and head hash. Removing a valid suffix is therefore
  detected rather than silently accepted.
- Cooperative-mind snapshots use Python pickle and are **trusted-local-only**.
  A checksum detects accidental corruption; it does not make an untrusted pickle
  safe to load.

## Learning bounds

Concept acquisition has explicit candidate and interaction budgets and supports
cancellation. A stopped or exhausted search reports the reason instead of
appearing to hang indefinitely.

## Verification

Pytest is the official collector because it covers both class-based unittest
cases and module-level test functions. Continuous integration runs a
dependency-light core matrix and a complete job with the optional neural stack.

The original v0.2.0 evidence remains frozen and historical. New hardening is
verified separately and must receive its own release manifest before a new
release is declared frozen.
