# Write the guidance a framework maintainer needs

Stable delivery requires `releases/<VERSION>.md` in the exact source revision used to build the wheels. The version excludes the wheel's ROCm/local suffix; a release wheel for `1.2.3+rocm7.2.manylinux.2.28` uses `releases/1.2.3.md`. There are no accepted release notes yet in this local redesign.

[The unreleased technical draft](unreleased.md) explains this implementation's behavior, compatibility changes and upgrade considerations. It does not name a release or claim maintainer approval.

Copy [the intentionally incomplete template](../ci/release/templates/release-notes.md), replace every marker, and have the content reviewed before choosing the release revision. Keep the six section headings. Explain compatibility and exclusions, exact upgrade steps, known issues, rollback and qualification in concrete terms. Review attribution names the people who accepted the text and links their pull request or review. Do not claim a review that has not happened.

Run `python -m ci.release.notes --version VERSION --file releases/VERSION.md`. This checks the document's structure, attribution, version and unfilled markers; it cannot determine whether a human's technical judgment was sound or enforce GitHub approvals. Repository review rules and the named reviewers provide that part of the process.

The stable gate reads the file from the wheel's Git revision, verifies that all wheels name the same version, and records a digest of the curated content. A changed local file or a generated PR list cannot replace it. Image publication waits for this gate. The release body preserves the curated text and appends generated changes and asset links afterward. Nightly records use shorter source and qualification-scope notes and do not claim stable support.
