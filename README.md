# Better Facial Animations

An SKSE plugin for Skyrim SE and VR that generates facial animation from speech using NVIDIA Audio2Face. It animates the face and jaw, with tongue cues from the game's speech track. Supports NPC dialogue and optional player speech through Dragonborn ReVoiced.

## Requirements and installation

- Skyrim SE 1.5.97, 1.6.353, 1.6.640, 1.6.659 (GOG), 1.6.1170, 1.6.1179 (GOG), 1.7.99 or 1.7.104, with matching SKSE and Address Library for SKSE Plugins; or Skyrim VR 1.4.15 with SKSEVR (no Address Library needed).
- GPU-first V3 is the normal preset (`ForceCPU=0`), initialized after game entry. If its GPU attempt fails, vanilla lip-sync remains available. CPU runs only when selected; Performance trades facial detail for lower computation cost. The installer separately selects the GPU route, GPU/CPU startup, and V3 or Performance + Potato Mode. Start on CPU preserves the chosen route for later manual GPU opt-in.
- GPU inference defaults to an independent D3D12 device/private queue (`IndependentGpuDevice=1`), requiring a supporting runtime and driver ([specification](https://microsoft.github.io/DirectX-Specs/d3d/IndependentDevices.html)). `IndependentGpuDevice=0` explicitly selects shared GPU compatibility. Change the installer/INI and restart to switch routes. Shared mode retries without metacommands, conditionally through an observed graphics wrapper after device loss, then at 2/4/8/16/30-second intervals. Successful shared variants are remembered per model; sentiment retries after 30 seconds. GPU errors never select CPU automatically: vanilla lip-sync remains available, and **Try GPU again** starts an explicit attempt. CPU is used only with **Force CPU**, selectable in the installer or settings; unchecking it tries GPU in the same session. There is no Auto device probe or automatic GPU-route switch. Runtime DLLs and failed independent resources remain retained. Retired `GpuDevice`, `AutoGpuDevice` and `UseDeviceWrapper` keys are ignored.
- Optional: SKSE Menu Framework for in-game settings.

Install through your mod manager. The FOMOD defaults to independent GPU, GPU startup after game entry, and V3. Shared GPU compatibility and Start on CPU are explicit alternatives; Performance + Potato Mode is suggested for Deck/lower CPU work. Manual installation copies only `SKSE` into `Data` and uses the same defaults. Keep bundled runtime/model files together and launch through SKSE.

The installer includes the [GPU setup and recovery guide](dist/GPU-SETUP.txt), also installed beside the models. It covers Windows/driver and Proton requirements, CPU recovery without launching Skyrim, runtime GPU activation, and useful failure reports.

GPU startup begins after successful game entry, on the inference worker. No recovery file is created or checked; old `gpu-startup.pending` files are ignored. If startup hangs or crashes, reinstall with **Force CPU** or set `ForceCPU=1` in the effective INI. CPU mode avoids GPU inference initialization. An in-process driver call already hanging cannot be cancelled by changing settings; no helper program is used.

Edit `Data/SKSE/Plugins/BetterFacialAnimations.ini` ([settings reference](dist/BetterFacialAnimations.ini)), or open **SKSE Menu Framework → Better Facial Animations → Settings**. Male and Female profiles apply to NPCs; the Player profile independently controls player timing, strength and emotion for either sex. Select Male NPCs, Female NPCs or Player to show that profile's timing, transitions, movement and emotion settings. **Vanilla / scene emotion inputs** (on by default) independently enables authored emotion inputs, including dialogue-mod inputs; Sentiment Analysis remains independent. Both sources off keeps audio-driven lip-sync without these emotion inputs. Each profile has separate **maximum inputs** for the model's ten emotion channels (0–2; default 1; ceilings, not fixed levels or movement multipliers), applied after strength, volume swell and sentiment from the next line. Lower **Maximum happiness input** to soften smiles without reducing anger or sadness. Grief, out of breath and pain are currently unused by Skyrim's emotion mapping. **Sentiment Analysis** (off by default) adds voice sentiment when the optional `emotion_fp16.onnx` is installed. **This is a heavy feature: loading the additional model requires 600 MB of extra VRAM on GPU, on top of the facial animation model, plus substantial additional processing power and longer animation generation time.** Force CPU uses system memory and additional CPU processing instead. Sentiment Analysis only adds voice-detected emotion to the facial model inputs on top of the scene's authored emotions. The Sentiment addition multiplier (0–2, default 1) scales only that extra input: 0 = none, 0.5 = half, 1 = normal, 2 = double. It is available when Sentiment Analysis is enabled and does not replace, reduce or blend away authored emotion. Zero skips sentiment inference; changes apply next line. The combined scene and multiplied sentiment inputs obey the per-emotion maximums; there is no separate sentiment cap. The controls are greyed out without the model; normal animation still works. Disabling Sentiment Analysis unloads its model after current inference. Menu changes save automatically. Model and Force CPU selections switch live on the worker; playing lines finish with their original animation.

**Better Facial Animations → Overrides** provides per-NPC playback, strength, tongue, gaze and emotion settings, including **Sentiment addition multiplier** (0–2). The NPC’s **Sentiment Analysis** checkbox turns it off for that NPC while preserving their multiplier. Both controls inherit the shared settings until edited; global Sentiment Analysis must be enabled to use sentiment. On SE/AE with SKSE Menu Framework 3, choose **Assign hotkey**, press an unused keyboard key, close settings, look at the NPC and press that key to open their editor. No key is assigned by default; Escape closes the editor or cancels binding. **Select NPC under crosshair** also works with older framework versions. Edited controls save automatically; **Use global** restores inheritance for one control, and **Reset NPC to global settings** clears all of that NPC's overrides. Reopen saved entries with **Edit**, or remove them on the Overrides page. Unchecking **Enable facial animations for this NPC** keeps vanilla animation. Global enable and race restrictions still apply. Overrides apply across saves to the NPC's base record (including copies), survive load-order changes, and are preserved by global Defaults. The player and temporary NPC base records cannot be selected. For manual configuration, including VR, add `Plugin.esp|001234=1` under `[Overrides]`, then add `[Override:Plugin.esp|001234]` with `Enabled=0` to disable BFA or profile keys such as `DirectMouth=0.8`, `SmileEyes=1.2`, `VoiceEmotion=0.5`, `SentimentAnalysis=0` (sentiment off), `SentimentMultiplier=0.5` and `MaxJoy=0.4`. Omitted controls inherit the current global profile. Use six-digit hexadecimal local **NPC base** IDs (light IDs padded to six digits) and restart after manual edits. Missing plugins retain entries. Existing `[Blacklist]` entries and `BlacklistHotkey` are read for migration and replaced by the new format when settings save.

## Behavior and compatibility

BFA fits animation to each head's own expression morphs. Custom heads, NPC replacers and High Poly Head can animate when the required morphs are present; heads missing them keep vanilla lip-sync. The Races menu lists detected races with individual checkboxes; unchecked races keep vanilla animation. Creatures and animals without FaceGen heads are skipped. Vanilla, DLC and modded voices work with packed or loose audio, with or without `.lip` files. Potato Mode (`PotatoMode=1`) limits generation and animation to the player and the NPC you are talking to. Both modes initialize inference after the first successful save load or new game, avoiding startup module scans, and keep the model ready until exit. The first lines may use vanilla animation while it warms up. Lines with missing or silent voice audio (including Fuz Ro D'oh dialogue) keep vanilla lip-sync and expressions. BFA takes over only when an animation from audible audio is ready.

Races owned by `DWAlfiqKhajiit.esp` (Creatures of Nirn - Khajiit Alfiq)
default off in the Races menu, including corrupted and boss variants.
They can be enabled individually; compatible face morphs are still required.
Saved race choices are preserved. Ordinary Khajiit remain supported.
Renamed or merged origin plugins need a separate compatibility check.

The face and jaw follow Audio2Face, while the tongue combines partial jaw movement with softened authored speech cues. Playback, region strength (including separate upper and lower lip), and emotion are adjustable separately for male and female heads; `DisableTongue=1` restores the previous model-driven tongue behavior.

Conversation eye gaze adds speech-driven eye motion while the speaking conversation partner looks at the player. It is enabled by default on SE/AE, preserves the game's target, and yields when the actor looks elsewhere. Toggle **Enable eye gaze** in **Better Facial Animations → Settings → Strength**; changes apply immediately. In `[LocalAnimationStrength]`, `ConversationGaze=0` disables it; `GazeStrength` (0–2, default 1) adjusts the motion. VR keeps its existing gaze. Original Skyrim eye meshes have offline visual coverage; in-game and custom-eye calibration remain unverified.

Skyrim 1.6.353, 1.6.640 and GOG 1.6.659 have offline checks but remain untested in-game. VR has only been exercised with a simulated headset; visual quality in a headset is unverified. Offline compatibility checks do not establish in-game compatibility with other mods. See the [player readme](dist/README.txt) for additional details and log locations, and [test documentation](tests/README.md) for coverage and limitations.

In **Settings → Model**, choose one checkbox: **V3** (most expressive, 60 fps,
about 250 MB inference VRAM on GPU), **Performance** (experimental 81k-parameter
Skyrim-trained model, GPU inference or optional CPU inference, less emotive,
30 fps and no eye gaze feature), or **Old model** (legacy v2.3, about 100 MB
VRAM on GPU, slower and less expressive than V3). Hover for tradeoffs. Models switch live; pending old-model work is cancelled while completed lines
finish with their original animation. Performance uses the existing male/female head
mapping; `ForceCPU` applies to all three models. It is not yet validated on
Steam Deck. `UsePerformanceModel=1` selects it in the INI.

## Player dialogue

Dragonborn ReVoiced (DBReV) 1.4.4+ and a compatible voice pack can animate the player. Select the voice in DBReV's MCM and use a third-person or cinematic camera to see the player's face. **General → Player facial animations**, or `[General] PlayerFacialAnimations=0` in the INI, controls player animation independently of NPC animation. Player animation works with Potato Mode and both DBReV and the experimental DBVO 2 support. The master Enabled setting still applies.

For Cinematic Conversation Camera, disable its synthesized fallback lips with `[Performance] bSynthLipSync=0` in `SKSE/Plugins/SD_user.ini`. DBReV retains its own authored lip track when BFA's player feature is off.

### DBVO 2 (experimental)

Dragonborn Voice Over 2 **2.0.2.2** can animate the player through a native speech observer, enabled experimentally on every desktop runtime BFA supports: **1.5.97, 1.6.353, 1.6.640, 1.6.659 (GOG), 1.6.1170, 1.6.1179 (GOG), 1.7.99 and 1.7.104**. BFA detects the installed DBVO plugin automatically. **Player facial animations** controls both DBVO2 and DBReV and is on by default; the old `NativePlayerSpeech` setting is ignored. Keep DBVO's pitch at 100. VR and unknown runtimes are not enabled. DBVO itself must support and load on the selected runtime; the pinned 2.0.2.2 DLL excludes 1.7.99.

The observer does not patch DBVO or replace scripts. It installs only when the entry bytes match the expected prologue and declines when another hook is present. In-game playback has been verified on 1.6.1170; 1.5.97 and 1.7.104 have passed binary hook/ABI and live SKSE/plugin startup checks; player playback still needs in-game validation. 1.6.353, 1.6.640, GOG 1.6.659, GOG 1.6.1179 and 1.7.99 are enabled but unverified. Non-default pitch, audio without LIP, rapid skips and other DBVO builds are untested; short cold lines may keep native lips. See [tests/DBVO.md](tests/DBVO.md).

## Speech providers

Mods that play their own voice lines (AI dialogue, TTS) can animate them through the plain C interface in [BFA_API.h](include/a2f/speech/BFA_API.h), also shipped beside the plugin's settings. Discover `BFA_QueryInterface` at SKSE PostLoad, register a listener, take a reply ID from `NewUtteranceId`, and `Prepare` each segment's WAV bytes with its speaker, reply ID and segment number. Report the audio clock with `Start`, `Pause` and `Seek` (which also unpauses); `Stop` ends a segment. Give all of a reply's segments one reply ID: BFA keeps the face between them. Keep fallback lipsync until `kStarted`; a terminal event returns control and retires the handle, and an animated segment's arrives once BFA hands the face back, after `EndUtterance` and the fade to the engine's expressions. A segment's optional emotion (ten channels) is copied with it and generated into its animation. `EndUtterance` closes the reply, stops its segments, removes its queued/cached audio and animations, and cancels running inference (temporary worker memory is released when cancellation is observed). Playback operations return one `uint32_t` result enum; Prepare and listener registration return an ID plus an output result. Engine access is queued through SKSE. At most 256 handles/replies may remain open; inactive handles/reply holds expire after 120 seconds. SE/AE actor lookup is supported; VR reports UnsupportedRuntime. This unreleased API changes in place: rebuild callers although `BFA_API_VERSION` remains 1. Offline contracts are in `provider-api`; in-game validation is still required.

## Development

Headers live in `include/a2f/`, with implementations in `src/`; both are grouped into `animation`, `inference`, `plugin`, `runtime`, and `speech`. Use includes such as `"a2f/animation/gaze.h"` with `include/` on the header search path.

C++17, cross-compiled with MinGW-w64 and GNU Make. Dependencies: static FFmpeg 8.0.1, ONNX Runtime DirectML 1.24.4, DirectML 1.15.4, and FP16 Audio2Face v2.3/v3.0 models. `./build.sh release` produces matching `-with-sentiment` and `-without-sentiment` archives; both require `EMOTION_MODEL` (default `models/a2e/network_fp16.onnx`) to build. Ordinary builds also work without it. Prepare FP16 sentiment weights with `tools/a2e_fp16.py`. Dependency paths and overrides are in [build.sh](build.sh); model preparation tools are in [tools/](tools/).

```sh
./build.sh                 # Assemble build/mod; no release archive
JOBS=4 ./build.sh          # Limit parallel compilation
./build.sh install <dir>   # Build and copy into a mod folder
./build.sh release         # Create a numbered archive (requires 7z)
./build.sh compat          # Offline compatibility contracts
./build.sh test            # Full suite (requires Wine and test assets)
```

[VERSION](VERSION) sets the release version. Release builds stop after revision 15, the largest revision SKSE can encode; they never wrap or change `VERSION` automatically. See [tests/README.md](tests/README.md) for fixtures and test coverage.

## License

The plugin source code is [GPL-3.0](LICENSE), with required attribution: “Based on Better Facial Animations by Cyberquach”. Bundled third-party libraries and NVIDIA models retain their separate licenses; the GPL does not replace those terms. Audio2Emotion is governed by its [specific Audio2Face-only license](dist/licenses/NVIDIA-Audio2Emotion-License.txt), which prohibits use for emotion recognition. See [NOTICE.txt](dist/NOTICE.txt) and [third-party licenses](dist/licenses/).
