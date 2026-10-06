# Versioning

Use `MAJOR.MINOR.PATCH` for release versions in `VERSION`:

- Bug fixes increment PATCH: `0.0.X`.
- Backwards-compatible features increment MINOR: `0.X.0`, resetting PATCH.
- Breaking new features or major releases increment MAJOR: `X.0.0`, resetting
  MINOR and PATCH.

Feature-branch test archives append an incrementing build number:
`MAJOR.MINOR.PATCH.BUILD`. For example, the first test build for the 3.0.1
bug-fix release is **3.0.1.1**. The fourth number is a test-build counter, not
another release-version component. Keep `VERSION` at the three-part release
version; `build.sh release` selects the test-build suffix before compiling.

The archive filename, DLL version (including the SKSE exported plugin version),
and packaged readme must agree on the complete selected build version. Do not
only rename an archive to change its version.

# Compatibility tests

Run `./build.sh compat`; see [tests/README.md](tests/README.md) for fixtures.
Offline passes do not establish in-game compatibility. Missing fixtures are
blocked, not passing. Do not download third-party fixtures automatically.

# Branches

Reuse a dedicated branch for each feature or fix; preserve unrelated work.
Commit scoped changes before making release archives, and record the source
commit and artifact hashes in the main checkout's untracked research journal.

# Documentation

Keep README/AGENTS/docs limited to commands and facts needed to build, run or
test. Update existing text instead of appending a section for every prompt.
Do not add task recaps, policy essays, proposals, repeated inventories or agent
rules unless requested. Put experiment history in the main checkout's untracked
RESEARCH_JOURNAL.md. Link to code or the case inventory instead of restating it.
