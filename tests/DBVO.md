# DBVO 2.0.2.2 regression tests

These tests target the native speech calling convention observed with DBVO
**2.0.2.2** on Skyrim **1.6.1170**, with automatic DBVO detection, PlayerFacialAnimations enabled and default
DBVO pitch. Use a local copy of that version; there are no downloads, Nexus
credentials or scheduled upstream checks.

## Running the tests

Verify that your local DLL is the exact build used for the prototype, then run
the native regression test:

```
python3 tools/dbvo_compatibility.py --dll /path/to/DBVO.dll --report work/dbvo-local.json
x86_64-w64-mingw32-g++ -std=c++17 -O2 -Wall -static src/native_speech_hook.cc tests/native_speech_hook_test.cc -o /tmp/dbvo-hook.exe
WINEDEBUG=-all timeout 90 wine64 /tmp/dbvo-hook.exe
```

The DLL checker compares the local binary against the reviewed SHA-256 in
`tests/dbvo2_baseline.json` and checks its x64 PE headers. It does not execute DBVO.
Exit 0 means the fixture matches; 1 means a different or malformed binary;
2 means the fixture could not be checked. Reports record the hash and version.
DBVO.dll is not redistributed in this repository.

The native test executes BFA's production detour and trampoline with DBVO's
observed 14-argument speech ABI and a float return. It checks register and stack
arguments, every modified prologue byte, and refusal of an existing entry hook.
This uses a synthetic speech target, so it runs without Skyrim or the DBVO DLL.
`build.sh test` additionally runs the production observer's load/reset and nested
speech controls.

The checker's own positive and negative controls run with:

```
python3 -m unittest discover -s tests -p test_dbvo_compatibility.py -v
```

GitLab CI runs those controls and the native regression test on normal pipelines.
It requires no DBVO download or API credentials. The local DLL identity check
and the synthetic ABI test serve different purposes: neither alone proves
in-game animation behavior.

## Runtime coverage

The entry observer is enabled experimentally on all BFA-supported desktop
runtimes: 1.5.97, 1.6.353, 1.6.640, GOG 1.6.659, 1.6.1170, GOG 1.6.1179,
1.7.99 and 1.7.104.
The native test checks this exact allowlist, including the GOG store field,
and refusal of VR, other stores and unknown runtimes. The exact entry-byte
check and refusal of existing detours remain mandatory. Reviewed local
executables have the same relocatable seven-byte entry prefix and observed
speech argument layout; Address Library resolves the different locations:

| Runtime | Address Library ID | Speech entry RVA |
| --- | --- | --- |
| 1.5.97 | 36541 | 0x5E8700 |
| 1.6.1170 | 37542 | 0x67A9D0 |
| 1.7.104 | 37542 | 0x68D490 |

The pinned DBVO DLL explicitly declares the three inspected runtimes plus
1.6.640 and GOG 1.6.1179, but excludes 1.7.99. Enabling BFA's observer cannot
make an incompatible DBVO DLL load. The newly enabled 1.6.640, GOG 1.6.1179
and 1.7.99 have no BFA/DBVO playback validation.
Binary inspection, synthetic trampoline execution and live SKSE/plugin startup
checks pass on 1.5.97 and 1.7.104, including DBVO detection and entry-hook
installation. These do not establish player playback; both runtimes still
require that in-game validation. Provenance and
acceptance trials are recorded in the main checkout's research journal.

## In-game coverage and baseline changes

The pinned DLL was tested with a player line, NPC control and two save/reload
trials. The contextual project journal records the experiment and raw evidence
under `work/dbvo-prototype`. Other runtimes, altered pitch and rapid skipping
remain outside that evidence.

When deliberately adopting a different DBVO version, review its native speech
call and repeat the comparable in-game trials before updating the baseline.
Check duplicate sessions, cleanup during playback, audio/frame timing and visible
facial motion. Record the version, hash, observations and limitations in the same
project journal; do not change the baseline solely to clear a failing check.
