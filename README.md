# Better Facial Animations

An SKSE plugin for Skyrim SE and VR that generates facial animation from speech using NVIDIA Audio2Face. It animates the face and jaw, with tongue cues from the game's speech track. Supports NPC dialogue and optional player speech through Dragonborn ReVoiced.

## Requirements and installation

- Skyrim SE 1.6.1170, 1.6.1179 (GOG), 1.7.99 or 1.7.104, with matching SKSE and Address Library for SKSE Plugins; or Skyrim VR 1.4.15 with SKSEVR (no Address Library needed).
- A DirectX 12 GPU for GPU inference: approximately 250 MB of VRAM with the default model, or 100 MB with `UseOldModel=1`. CPU fallback is available but slower.
- Optional: SKSE Menu Framework for in-game settings.

Install the release archive through your mod manager, or extract its `SKSE` folder into the game's `Data` directory. Keep the bundled model and runtime files together, and launch the game through SKSE.

Edit `Data/SKSE/Plugins/BetterFacialAnimations.ini` ([settings reference](dist/BetterFacialAnimations.ini)), or open **SKSE Menu Framework → Better Facial Animations → Settings**. Menu changes save automatically; model and CPU/GPU changes require a restart.

## Behavior and compatibility

BFA fits animation to each head's own expression morphs. Custom heads, NPC replacers and High Poly Head can animate when the required morphs are present; heads missing them keep vanilla lip-sync. Argonians keep vanilla lip-sync, and creatures and animals are skipped. Vanilla, DLC and modded voices work with packed or loose audio, with or without `.lip` files.

The face and jaw follow Audio2Face, while the tongue combines partial jaw movement with softened authored speech cues. Region strength and emotion are adjustable; `DisableTongue=1` restores the previous model-driven tongue behavior.

Skyrim 1.6.640 has load/hook checks but remains untested in-game. VR has only been exercised with a simulated headset; visual quality in a headset is unverified. Offline compatibility checks do not establish in-game compatibility with other mods. See the [player readme](dist/README.txt) for additional details and log locations, and [test documentation](tests/README.md) for coverage and limitations.

## Player dialogue

Dragonborn ReVoiced (DBReV) 1.4.4+ and a compatible voice pack can animate the player. Select the voice in DBReV's MCM and use a third-person or cinematic camera to see the player's face. **General → Player facial animations**, or `[General] PlayerFacialAnimations=0` in the INI, controls player animation independently of NPC animation. The master Enabled setting still applies.

For Cinematic Conversation Camera, disable its synthesized fallback lips with `[Performance] bSynthLipSync=0` in `SKSE/Plugins/SD_user.ini`. DBReV retains its own authored lip track when BFA's player feature is off.

## Development

C++17, cross-compiled with MinGW-w64 and GNU Make. Dependencies: static FFmpeg 8.0.1, ONNX Runtime DirectML 1.24.4, DirectML 1.15.4, and FP16 Audio2Face v2.3/v3.0 models. Dependency paths and overrides are in [build.sh](build.sh); model preparation tools are in [tools/](tools/).

```sh
./build.sh                 # Assemble build/mod; no release archive
JOBS=4 ./build.sh          # Limit parallel compilation
./build.sh install <dir>   # Build and copy into a mod folder
./build.sh release         # Create a numbered archive (requires 7z)
./build.sh compat          # Offline compatibility contracts
./build.sh test            # Full suite (requires Wine and test assets)
```

[VERSION](VERSION) sets the release version. See [tests/README.md](tests/README.md) for fixtures and test coverage.

## License

[GPL-3.0](LICENSE), with required attribution: “Based on Better Facial Animations by Cyberquach”. See [NOTICE.txt](dist/NOTICE.txt) and [third-party licenses](dist/licenses/).
