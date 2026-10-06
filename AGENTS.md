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

`./build.sh compat` builds and runs the offline compatibility contracts;
[tests/README.md](tests/README.md) has the commands, the fixture format and the
case inventory ([tests/compatibility.json](tests/compatibility.json)). A static
pass is never an in-game compatibility claim: name the exact mod versions and
what was actually observed. Missing fixtures are blocked, not passing.

# Writing documentation

Documentation is for a person who wants to do one thing: build, run, test, add
a case. Write the commands and the facts they need, in plain sentences, and
stop. Concretely:

- No policy essays, principle lists, citation keys (`[S1]`) or "proposed"
  designs in README/AGENTS/docs files. Experiment history, triage and proposals
  go in the untracked research journal, not the repository.
- Do not restate what the code, the JSON case inventory or `build.sh --help`
  already say. Link to the file instead.
- A change to a README or this file should be about the size of the change it
  documents; a 50-line feature does not need 400 lines of prose.
- Do not add process rules for agents (branching etiquette, journal rituals) to
  this file unless the user asks for them.
