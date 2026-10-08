# Security policy

GUM is a local research prototype, not a sandbox for untrusted code.

World folders contain data contracts; executable adapters must be registered in source. Do not add a feature that automatically imports or executes code from a dropped world. GUM Studio is deliberately loopback-only and must not be exposed through a proxy or tunnel without a separate security review.

Studio uses a fresh bearer token, Host/Origin validation, bounded JSON request
bodies, and browser security headers. These controls reduce ordinary local web
attack risk; they do not make Studio suitable for hostile users or the public
internet.

Cooperative specialist snapshots contain Python pickle data. Loading them is an
explicit trusted-local operation. Never opt in to loading a snapshot obtained
from another person or an untrusted download: its checksum checks integrity, not
safety.

Do not place secrets, private datasets, credentials, or sensitive personal files in `.gum-workspace` or evidence bundles. Report a vulnerability privately to the repository owner after a public contact address is added. Until then, do not publish exploit details in an issue.

