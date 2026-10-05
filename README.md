# Better Facial Animations

Real-time NVIDIA Audio2Face-3D (v3.0, or v2.3 Mark) lip-sync for Skyrim SE 1.6.1170, 1.6.1179 (GOG), 1.7.99 and 1.7.104, as an SKSE plugin. 1.6.640 is supported in theory: the plugin loads there and its hooks were checked against the 1.6.640 exe, but it hasn't been tested in-game yet. Skyrim VR 1.4.15 (SKSEVR) is supported too: its functions and layouts were matched against SkyrimVR.exe, whose offsets the plugin carries itself instead of using an Address Library. It was run there only on a simulated headset (lines are picked up, generated and played), so how the faces look in a headset is untested.

When an actor with a FaceGen head speaks, the plugin runs the voice file through the Audio2Face network on the GPU (ONNX Runtime + DirectML). The result deforms the head's face meshes directly (A2F Direct): each head is mapped onto the face the network animates for it (v3.0, the default: Claire for women, Mark for men, at 60 fps; v2.3 with `UseOldModel=1`: Mark for everyone, at 30 fps), the first time it speaks, from its own expression morphs; its brows, beard and eyes follow its skin, and its mouth part takes Mark's jaw and tongue: the lower teeth and gums turn with the network's jaw, and the tongue rides it and bends like Mark's. A line's emotion is fed to the network per frame: the game's emotion value for the line, times `VoiceEmotion`, growing with the voice's loudness by `VoiceSwell`. A head whose .tri lacks a morph the mapping needs keeps vanilla lip-sync. The game's emotion expressions (dialogue, mood, combat) of the NPC the player talks to and of anyone speaking are played on the same meshes from each head's own expression morphs, at half strength on the mouth; they fade out while a line plays, when the network alone shapes the face and the game's emotion reaches it only as the network's emotion input, and back in after it. There is an optional settings page through SKSE Menu Framework.

- `src/better_facial_animations.cc`: hooks, worker thread, line cache, playback
- `src/head_cache.*`: one head -> face mapping per distinct head (and sex, with v3.0), built on its own thread
- `src/settings.cc`, `src/menu.*`, `src/plugin.h`: ini, settings page, what the plugin's files share
- `src/engine.h`: engine object layouts and function types
- `src/pose.*`: coefficient-space continuity across line starts, replacements and endings
- `src/live_core.*`: audio decoding, the line's emotion (game, loudness), Audio2Face network windows, Direct coefficients
- `src/direct.*`: A2FDirect.bin (v2.3: Mark) and A2FDirect3.bin (v3.0: Claire and Mark), the head -> face mapping (skin, jaw, tongue) and per-frame vertex deltas
- `tools/mark_export.py`: writes A2FDirect.bin from the Audio2Face v2.3 Mark package
- `tools/a2f3_fold.py`, `tools/a2f3_ref.py`, `tools/a2f3_bake.py`, `tools/mark_export3.py`: fold the v3.0 network into one ONNX graph, its Python reference, its output for a set of voice lines, and A2FDirect3.bin from those
- `src/ort_net.*`, `src/net_keeper.*`, `src/gpu_state.h`: ONNX Runtime C API wrapper (loaded at runtime), keeping and regaining the GPU session; the session gets Direct3D's own device where ReShade hands out a wrapper of it
- `src/skse.*`: SKSE ABI, log, Address Library, engine layout
- `dist/`: ini, Direct face data (`A2FDirect.bin`, `A2FDirect3.bin`), player readme, notices, model license

## Player dialogue

[Dragonborn ReVoiced](https://www.nexusmods.com/skyrimspecialedition/mods/184221) 1.4.4 or newer can drive the player's face through the same Audio2Face pipeline. Install DBReV and a compatible voice pack, then select the voice in its MCM. BFA's **Player facial animations** setting (General) is independent of NPC animation; the INI equivalent is `[General] PlayerFacialAnimations=0` to disable it. The master Enabled setting still applies. Use a third-person or cinematic dialogue camera to see the player's face. The usual head/race requirements apply.

For Cinematic Conversation Camera, disable its synthesized fallback lips with `[Performance] bSynthLipSync=0` in `SKSE/Plugins/SD_user.ini`. DBReV retains its own authored lip track when BFA's player feature is off.

`src/player_speech.cc` is the optional provider adapter; `src/player_speech.h` is the small owned speech-session interface shared with the renderer. The adapter registers at SKSE PostLoad, checks the public append-only message layout, copies the voice path, and queues starts on the game thread. Ends, replacements and load transitions cancel even a start still waiting in the queue. It uses audio duration rather than the post-dialogue delay. No DBReV DLL linking, internal hooks, script replacements or voice-pack-specific code is needed. Future compatible API extensions are accepted; breaking changes outside that published contract cannot be guaranteed compatible.

## Building

Requires `x86_64-w64-mingw32-g++`, `7z` (for releases) and these dependencies. The paths are defaults; override them with the variables in `build.sh`:

- `deps/ffmpeg/win64` (`FFMPEG`): FFmpeg 8.0.1, from https://ffmpeg.org/releases/ffmpeg-8.0.1.tar.xz, configured with
  `--target-os=mingw32 --arch=x86_64 --cross-prefix=x86_64-w64-mingw32- --enable-static --disable-shared --disable-programs --disable-doc --disable-everything --disable-network --disable-autodetect --disable-debug --enable-small --disable-avdevice --disable-avfilter --disable-swscale --enable-decoder=wmav2,wmav1,pcm_s16le,pcm_u8,pcm_s24le,pcm_f32le --enable-demuxer=xwma,wav --enable-swresample --enable-w32threads`
- `deps/dml/microsoft.ml.onnxruntime.directml` (`ORT`): the NuGet package Microsoft.ML.OnnxRuntime.DirectML 1.24.4, extracted
- `deps/dml/microsoft.ai.directml` (`DML`): the NuGet package Microsoft.AI.DirectML 1.15.4, extracted
- `models/mark/network_fp16.onnx` (`MODEL`): `network.onnx` from [nvidia/Audio2Face-3D-v2.3-Mark](https://huggingface.co/nvidia/Audio2Face-3D-v2.3-Mark), converted with
  `onnxconverter_common.float16.convert_float_to_float16(model, keep_io_types=True)`
- `models/v3/network_fp16.onnx` (`MODEL3`): `network.onnx` from [nvidia/Audio2Face-3D-v3.0](https://huggingface.co/nvidia/Audio2Face-3D-v3.0), folded into one graph with `venv/bin/python tools/a2f3_fold.py models/v3` (`network_folded.onnx`) and converted the same way; shipped as `a2f3_fp16.onnx`. `dist/A2FDirect3.bin` is written by `tools/mark_export3.py`

The version is in `VERSION`; `build.sh` builds it into the plugin and the packaged readme.

Commands:

- `./build.sh` builds into `build/mod`.
- `./build.sh install <dir>` also copies the result into a mod folder.
- `./build.sh release` also writes the release archives to `release/`.
- `./build.sh test` also checks under `wine64` that SKSE would load the plugin on each supported runtime and that every Address Library id resolves. It needs the Address Library for SKSE Plugins (all-in-one) files, extracted to `deps/address-library` (`ADDRLIB`) so that `SKSE/Plugins` is inside it.

## License

GNU General Public License v3.0; see [LICENSE](LICENSE).

Additional term under GPL v3 section 7(b): any copy or modified version must keep the attribution "Based on Better Facial Animations by Cyberquach" in its documentation and on any page it is distributed from.

Bundled third-party components keep their own licenses, which are included in the release archive:

- FFmpeg: LGPL 2.1+
- ONNX Runtime: MIT
- DirectML: Microsoft
- the Audio2Face-3D model: NVIDIA Open Model License (`dist/licenses`)
