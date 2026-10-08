# Public release checklist

This repository is technically packageable. The author and license decisions are complete; the remaining work is GitHub connection and final review.

- [x] Public author recorded as Tolga Uz in `CITATION.cff` and package metadata.
- [x] Apache License 2.0 selected, with `LICENSE` and `NOTICE` included.
- [x] Human–AI development provenance documented in `AI_COLLABORATION.md`.
- [x] Initial release named `v0.2.0-public-preview`.
- [x] GitHub account created under `thingsbyt`.
- [x] Local Git authenticated as `thingsbyt` through Git Credential Manager.
- [x] Local remote prepared as `https://github.com/thingsbyt/GUM.git`.
- [ ] Create an empty `thingsbyt/GUM` repository; do not initialize it with another README.
- [ ] Confirm public or private visibility.
- [ ] Set a public commit email or the exact GitHub-provided no-reply email.

Then:

- [ ] Read the claims and limitations pages once more.
- [ ] Run the documented verification suite.
- [ ] Verify the release manifest and archive hashes.
- [ ] Search for credentials, private paths, and unintended large files.
- [ ] Create and push a fresh sanitized root commit from the audited v0.2 tree. Do not push the archival branches, old tags, or full-history bundles.
- [ ] Create a GitHub release from the verified source archive.
- [ ] Invite an independent person to reproduce at least one protocol without help.

The first public description should say “research prototype” or “research preview.” It should not say AGI, consciousness, medical intelligence, or proven general understanding.

