# Better Facial Animations

An SKSE plugin for Skyrim SE and VR that generates facial animation from speech using NVIDIA Audio2Face. It animates the face and jaw, with tongue cues from the game's speech track, for NPC dialogue and optionally the player.

## Requirements

- Skyrim SE 1.5.97, 1.6.353, 1.6.640, 1.6.659 (GOG), 1.6.1170, 1.6.1179 (GOG), 1.7.99 or 1.7.104 with matching SKSE and Address Library; or Skyrim VR 1.4.15 with SKSEVR.
- Microsoft Visual C++ v14 x64 runtime (tested with 14.51).
- A DirectX 12 GPU for the default setup; CPU is available when selected.
- Optional: SKSE Menu Framework for in-game settings.

1.6.353, 1.6.640 and GOG 1.6.659 have offline checks only. VR has been exercised only with a simulated headset.

## Installation

Install through your mod manager and launch through SKSE. The FOMOD selects the facial and emotion models, the processing device for each (**GPU 0–3** or **CPU**), and optional features; the defaults are shared GPU 0, Performance facial animation and Performance emotion. Manual installation copies `SKSE` into `Data` with the same defaults.

Installer choices are defaults; explicit values in `Data/SKSE/Plugins/BetterFacialAnimations.ini` win. A winning INI in MO2 Overwrite or another mod overrides new installer choices.

Inference starts after the first save load (or once a new game leaves the loading screens), so the first lines may play with vanilla animation. GPU failures never fall back to CPU automatically; vanilla lip-sync stays available and **Try GPU again** retries. If startup hangs or crashes, reinstall with **CPU** for both models, or set `ForceCPU=1` and `EmotionDevice=-1`. Driver, Proton and Independent GPU mode requirements are in the [GPU setup and recovery guide](dist/GPU-SETUP.txt).

## Models

**Audio2Face** (facial animation):

- **V3**: full model, 60 fps with speech-driven eye movement; about 270 MB VRAM.
- **Performance** (default): 22M-parameter student trained to reproduce V3 on the training dialogue; unfamiliar lines can be less reliable. 30 fps, GPU or CPU, about 185 MB VRAM.
- **Ultra Performance**: the original 80,729-parameter student, lowest cost and least detail. 30 fps, GPU or CPU, a few MB VRAM.
- **Legacy**: Audio2Face v2.3, 30 fps, about 100 MB VRAM.
- **Disabled**: vanilla animation.

INI: `UsePerformanceModel=1`, `UseUltraPerformanceModel=1` or `UseOldModel=1`; all at 0 selects V3.

**Audio2Emotion** (voice-driven emotion): **Performance** (default, student model, approximate outside its training dialogue, about 31 MB VRAM), **Full** (with-sentiment archive only, about 745 MB VRAM) or **Disabled**. INI: `SentimentAnalysis=0` disables, `UsePerformanceEmotion=0` selects Full.

Model and device changes apply live; playing lines finish with their original animation.

## Settings

Open **SKSE Menu Framework → Better Facial Animations**, or edit the INI ([settings reference](dist/BetterFacialAnimations.ini)). Male, Female and Player profiles each have timing, region strength and emotion caps. **Races** lists detected races; unchecked races keep vanilla animation. `PotatoMode=1` animates only the player and the NPC you are talking to.

**Overrides** holds per-NPC settings. On SE/AE with SKSE Menu Framework 3, assign a hotkey, close the menu, aim at an NPC and press it; **Select NPC under crosshair** also works. Manual entries use `[Overrides]` and `[Override:Plugin.esp|001234]` sections, with six-digit local NPC base IDs; restart after manual edits.

### Eye gaze (SE/AE)

**Settings → Gaze → Enable improved eye gaze** (`ConversationGaze`) is the master switch; off restores the game's own gaze and head tracking. While on, the NPC you talk to looks at you and may look away at pauses, a line's start or a change of emotion, onto nearby objects and people, turning the head for targets well to the side. The male and female profiles and NPC overrides set how firmly they hold your eye, the timing and the chance per emotion; keys are under `[Gaze]`, `[GazeMale]` and `[GazeFemale]`. V3 and Legacy also add speech-driven eye motion (`GazeStrength`, 0–2). VR keeps its existing gaze. Not verified in game.

## Compatibility

BFA fits animation to each head's own expression morphs, so custom heads, NPC replacers and High Poly Head work when the morphs are present; heads without them keep vanilla lip-sync. Packed and loose voices work with or without `.lip` files; silent or missing audio keeps vanilla animation. Creatures without FaceGen heads are skipped. Creatures of Nirn Alfiq races default off in the Races menu.

Incompatible with The Fourth Wall, which writes the same eye values. See the [player readme](dist/README.txt) for other mod interactions and log locations. Offline checks do not establish in-game compatibility.

## Player dialogue

Dragonborn ReVoiced 1.4.4+ with a compatible voice pack, or Dragonborn Voice Over 2 **2.0.2.2** (experimental, desktop runtimes only), animates the player. Toggle with **General → Player facial animations** (`PlayerFacialAnimations`). Use a third-person or cinematic camera to see the face. With Cinematic Conversation Camera, set `bSynthLipSync=0` in `SD_user.ini`. DBVO details and verification status: [tests/DBVO.md](tests/DBVO.md).

## Speech providers

Mods that play their own voice lines (AI dialogue, TTS) can animate them through the C interface in [BFA_API.h](include/a2f/speech/BFA_API.h), shipped beside the plugin's settings. Discover `BFA_QueryInterface` at SKSE PostLoad; the header documents the call sequence. SE/AE only. The API is unreleased and changes in place: rebuild callers.

## Development

Headers are in `include/a2f/` and sources in `src/`, grouped into `animation`, `inference`, `plugin`, `runtime` and `speech`.

C++23 / MSVC 19.51 with CommonLibSSE-NG 8.4.0; Linux builds run MSVC through Wine. Dependency paths and overrides (`COMMONLIB_DEPS`, `BFA_MSVC_ROOT`, `BFA_BUILD_WINEPREFIX`, `AGILITY`, `EMOTION_MODEL`) are in [build.sh](build.sh). Model export and training tools are in [tools/](tools/) and [tools/training](tools/training/README.md).

```sh
./build.sh                 # Assemble build/mod; no release archive
JOBS=4 ./build.sh          # Limit parallel compilation
./build.sh install <dir>   # Build and copy into a mod folder
./build.sh release         # Numbered with/without-sentiment archives (requires 7z and EMOTION_MODEL)
./build.sh compat          # Offline compatibility contracts
./build.sh test            # Full suite (requires Wine and test assets)
```

[VERSION](VERSION) sets the release version; release builds stop at build 15, the largest SKSE can encode. See [tests/README.md](tests/README.md) for fixtures and coverage.

## License

The plugin source code is [GPL-3.0](LICENSE), with required attribution: “Based on Better Facial Animations by Cyberquach”. Bundled third-party libraries and NVIDIA models keep their own licenses. Audio2Emotion is governed by its [Audio2Face-only license](dist/licenses/NVIDIA-Audio2Emotion-License.txt), which prohibits use for emotion recognition. See [NOTICE.txt](dist/NOTICE.txt) and [third-party licenses](dist/licenses/).
