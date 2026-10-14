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

Tests compile in parallel (`--jobs`, default `$JOBS` or the CPU count) and run one at
a time, as they share the working directory.
Each run writes a new directory under `build/compatibility/` (or `--output`)
with `result.json`, logs, commands and hashes of the sources, objects, test
executables and fixtures. Exit codes: 0 passed, 1 failed, 2 blocked (a missing
fixture is blocked, never a pass).

[compatibility.json](compatibility.json) lists the cases. `contracts` compile a
test from `sources` (plus the plugin objects when `plugin_objects` is set, and
any `link` flags), or run a `python` test; `build.sh test` uses the same entries.
`source-basenames` keeps `src/**/*.cc` names unique, as `tools/build.mk` finds
sources by `vpath`.
`runtime-layout` checks supported AE declarations, AE ID selection, and actor
race/voice reads before and after the 1.6.629 layout change. `build.sh test`
also checks the built DLL and every required ID against local Address Libraries,
including Steam 1.6.353 and GOG 1.6.659.

`conversation-gaze` checks v2/v3 eye-track alignment, additive look controls,
late-start sync, player-target/scene/runtime gating, restoration and hook coexistence.
`independent-device` and `independent-routing` check strict factory isolation, retained GPU context ownership across model/CPU transitions, latched device failures and all model routes without GPU hardware. They do not establish driver or graphics-hook compatibility.
`gpu-startup` checks persistent startup protection and file errors; `gpu-fallback` checks one GPU attempt, CPU recovery and persistence across settings changes. `ort-output` checks tensor validation and the shared facial/sentiment GPU failure gate. `installer` checks the FOMOD presets and GPU-first default.
`sentiment-lifecycle` checks optional-model gating, GPU-to-CPU fallback, CPU failure backoff and unloading.
`npc-menu` checks disabled sentiment controls; `build.sh test` checks additive caps and the local sentiment model when present.
`worker-start` checks that CPU/GPU workers wait for successful game entry, with either startup listener order, failed loads, repeated events and thread creation failure.
Settings persistence is covered by `sex-settings`; eye-mesh calibration still needs
in-game review. `provider-api` runs the provider interface (`include/a2f/speech/BFA_API.h`)
over the production plugin with mocked engine lookups and a synthetic generation;
covers deferred admission, animation-start events, playback clocks, per-line emotion,
reply-scoped queue/cache cleanup, late callback cancellation, capacity, reset and expiry.
The simplified event ABI needs in-game validation.

`assets` check local plugin record identities and pinned geometry/filter corpora.
`coverage_gaps` names what is not tested.

`voice-timing` and `audio-fallback` cover accelerated native lip clocks,
repeated samples, pause gaps, 1.25x playback, and an exhausted lip resource
while its voice continues. The fallback estimates speech time; it does not
measure the audio device cursor or certify a particular animation replacer.

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

The `efa-calibration` contract uses synthetic controls. To also check positive
recognition, supply locally extracted EFA Male Edition 1.21 and Female Edition
1.7 head assets to its executable after `./build.sh compat`:

```sh
wine64 build/compatibility/<run>/efa_test.exe /path/MaleHead.tri /path/FemaleHead.tri
```

This checks both editions, each alone, modified/swapped files and failed reads.
Missing supplied files return 2. Do not commit the TRI files.

For changes to head-part classification, hair exclusion or face-shape traversal,
run `./build.sh compat` and the `hair-assets` regression gate before merging or
releasing those changes.
`head-parts` runs synthetic oral-extra inheritance, hair-extra graph, shader, alias and facial-hair controls
on SE/AE/VR layouts in the normal suite and CI. `hair-assets` replays the reviewed
installed-source corpus through the current production traversal: all 18,582
scalp-hair occurrences must remain excluded and all 12,314 beard/brow/lash
controls admitted. It includes the KS Scarlet alias and Erin SMP support cases.

Add `"hair-assets": {"path": "/path/to/hair-corpus"}` to the fixture map and run:

```sh
python3 tools/compatibility.py --tier assets --case hair-assets --fixtures fixtures.json
```

The local corpus contains `queries.bin` (HDPT graphs and mesh names/shader flags)
and independently reviewed `expectations.json`; hashes and counts are pinned in
[hair_assets.json](hair_assets.json). Missing files are blocked; changed hashes
or admission results fail. Keep third-party derived fixtures outside Git. To
refresh the corpus after asset changes, repeat the installed-source audit, review
hair versus facial-hair labels and parser exclusions, then update the pins; never
regenerate expected admissions from the filter being tested. Source occurrences
include inactive/overridden variants and valid synthetic FOD data. This gate does
not establish runtime asset precedence, SMP behavior or in-game shadow rendering.

## Adding a case

Add an entry to `compatibility.json`: an `assets` entry for a plugin record,
a `contracts` entry for code behavior. Contract tests include the production
source and mock only engine calls (see `scene_compat_test.cc`). Include a
negative control (absent mod, wrong version, actor leaving the scene) so the
case fails when the integration breaks.

The `follower-teeth` asset case requires the ten locally prepared captures
and the independent anatomical-label file pinned in
[follower_teeth.json](follower_teeth.json), plus a native C++17 compiler
(`HOST_CXX`, default `g++`). Add `"follower-teeth": {"path": "/path/to/captures"}`
to the fixture map, then run:

```sh
python3 tools/compatibility.py --tier assets --case follower-teeth --fixtures fixtures.json
```

It checks both model bins, base/SMP heads, both tongue modes, tooth/jaw
separation, preserved embedded-tongue articulation, skin/lip independence and
finite basis responses. The meshes contain teeth, gums and a tongue; labels
come from inspected TRI connectivity, not from BigAah or BFA tongue weights. Missing captures
are blocked; changed hashes fail. Captures contain original local NIF positions
and all 36 named TRI morphs, with source hashes in the manifest. Regenerate and
review the pins when the morph inventory changes; do not pad missing captured
channels to pass. Captures are not distributed. This is an offline geometry
contract, not an in-game or SMP-physics test.

`model-switch` exercises live V3/Performance/Old selection, session release,
retained playback mappings, cancellation, CPU fallback and failed mapping loads.
It does not establish in-game switching quality.

`performance_model_test` covers native performance-model chunking and cancellation.
With arguments `<plugin-dir> <performance.onnx> <pcm-f32> <Claire-reference-f32>
<Mark-reference-f32>` it also checks the actual shipped CPU model against Python
for both identities and reports frontend-inclusive timing; missing fixtures do not
establish parity. `performance-frontend` checks the native frontend alone (frame
count, silence floor, tone placement, cancellation); `performance_features_test`
(built with `src/inference/performance_features.cc`) exports native mel features
for comparison with the training frontend. `audio-fallback` runs a performance
line through `Process`, including a model switch during a line. Model selection/migration/persistence is covered by
`sex_settings_test`. These are offline tests, not in-game or Steam Deck certification.

`race-settings` also bounds native-expression rebuild work at 30/60/120 Hz and
checks that gradual native ramps leave no smoothing tail. These are operation
budgets, not measured game CPU performance. For repeated game captures:

```sh
python3 tools/game_cpu_regression.py --baseline <4.0-run1> <4.0-run2> <4.0-run3> --candidate <run1> <run2> <run3>
```

Capture format is described in the tool. The gate requires matching initial INI,
modlist and measurement windows; it resamples whole launches and requires the
one-sided CPU-delta bound to be nonpositive for every window. Missing evidence
returns 2; CPU parity not demonstrated returns 1. Review raw stalls, effective
GPU routing, scene/system-load comparability and visual/API behavior separately.

`face-update` checks zero-time pose refreshes, native hook chaining, nested and
concurrent update contexts, and unchanged timing for normal facial updates.

`npc-blacklist` checks override migration, sparse profile inheritance, persistence,
emotion cache isolation, full/light identities across load-order changes and missing
plugins, removal, provider admission, and live speech/expression/morph handoff on
SE/AE/VR layouts. `npc-menu` checks keyboard binding, menu/text guards, target
lifetime, editing/reopening/resetting overrides and stale tasks
with a fake framework/dispatcher. These do not establish in-game UI compatibility.

`alfiq-fallback` checks owner-based race defaults, opt-in, persistence, reset,
speech and conversation admission, and normal Khajiit controls across SE/AE/VR layouts.
`alfiq-23` checks the three owned RACE records in a local Alfiq 2.3 plugin;
supply `"alfiq-23": {"path": "/path/DWAlfiqKhajiit.esp"}` in the fixture map.
Neither contract establishes in-game mouth animation.
