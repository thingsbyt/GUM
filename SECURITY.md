# Security policy

GUM is a local research prototype, not a sandbox for untrusted code.

World folders contain data contracts; executable adapters must be registered in source. Do not add a feature that automatically imports or executes code from a dropped world. Do not expose the Studio server beyond `127.0.0.1` without a separate security review.

Do not place secrets, private datasets, credentials, or sensitive personal files in `.gum-workspace` or evidence bundles. Report a vulnerability privately to the repository owner after a public contact address is added. Until then, do not publish exploit details in an issue.

