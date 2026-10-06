# Compatibility tests

Offline contracts against the production code; nothing here launches Skyrim,
installs anything or downloads anything. Needs the normal build environment
(MinGW-w64, Wine, Python 3.10+; `CXX`, `WINE` and `FFMPEG` overrides apply).

```sh
./build.sh compat                                  # build, then every contract case
python3 tools/compatibility.py --case scene-ownership   # one case, current build objects
python3 tools/compatibility.py --tier assets --fixtures fixtures.json
python3 tools/compatibility.py --build --tier all --fixtures fixtures.json  # release evidence
python3 tools/compatibility.py --list
./build.sh test                                    # the full suite; includes scene_compat_test
```

Each run writes a new directory under `build/compatibility/` (or `--output`)
with `result.json`, logs, commands and hashes of the sources, objects, test
executables and fixtures. Exit codes: 0 passed, 1 failed, 2 blocked (a missing
fixture is blocked, never a pass).

[compatibility.json](compatibility.json) lists the cases. `contracts` compile a
test from `sources` (plus the plugin objects when `plugin_objects` is set, and
any `link` flags), or run a `python` test; `build.sh test` uses the same entries.
`assets` check that a locally extracted plugin owns the FACT records BFA looks
up, by local id and editor id. `coverage_gaps` names what is not tested.

## Fixtures

`--fixtures` maps asset case ids to extracted plugins. A path alone is accepted;
pin the SHA-256 for repeatable release gates. Do not commit third-party files.

```json
{
  "sexlab-ae-166d": {"path": "/fixtures/SexLab-1.66d/SexLab.esm", "sha256": "<64 hex>"},
  "ostim-standalone-751": {"path": "/fixtures/OStim-7.5.1/OStim.esp"}
}
```

A record pass does not say anything about the mod's DLL. For that,
`python3 tools/skse_preflight.py <plugin.dll> --runtime 1.6.1170` reads the
DLL's declared SKSE runtimes without loading it (0 declared match, 1 declared
mismatch, 2 undetermined). An actual SKSE load log is still required.

## Adding a case

Add an entry to `compatibility.json`: an `assets` entry for a plugin record,
a `contracts` entry for code behavior. Contract tests include the production
source and mock only engine calls (see `scene_compat_test.cc`). Include a
negative control (absent mod, wrong version, actor leaving the scene) so the
case fails when the integration breaks.
