# Better Facial Animations

Real-time NVIDIA Audio2Face-3D (v2.3 Mark) lip-sync for Skyrim SE 1.6.1170, 1.6.1179 and 1.7.104, as an SKSE plugin.

When an actor with a FaceGen head speaks, the plugin runs the voice file through the Audio2Face network on the GPU (ONNX Runtime + DirectML). The result deforms the head's face meshes directly (A2F Direct): per-vertex deltas from reference meshes, resampled onto any head topology, with the jaw, teeth and gums on Skyrim's own mouth shapes. The line's emotion also plays as the engine expression morph. There is an optional settings page through SKSE Menu Framework.

- `src/better_facial_animations.cc`: hooks, worker thread, cache, playback, ini and settings page
- `src/live_core.*`: audio decoding, Audio2Face network windows and emotion, Direct coefficients, expression curve
- `src/direct.*`: A2FDirect.bin, mesh mapping, per-frame vertex deltas and the mouth rig
- `src/ort_net.*`: ONNX Runtime C API wrapper (loaded at runtime)
- `src/skse.*`: SKSE ABI, log, Address Library, engine layout
- `dist/`: ini, Direct face data (`A2FDirect.bin`), player readme, notices, model license

## Building

Requires `x86_64-w64-mingw32-g++`, `7z` (for releases) and these dependencies. The paths are defaults; override them with the variables in `build.sh`:

- `deps/ffmpeg/win64` (`FFMPEG`): FFmpeg 8.0.1, from https://ffmpeg.org/releases/ffmpeg-8.0.1.tar.xz, configured with
  `--target-os=mingw32 --arch=x86_64 --cross-prefix=x86_64-w64-mingw32- --enable-static --disable-shared --disable-programs --disable-doc --disable-everything --disable-network --disable-autodetect --disable-debug --enable-small --disable-avdevice --disable-avfilter --disable-swscale --enable-decoder=wmav2,wmav1,pcm_s16le,pcm_u8,pcm_s24le,pcm_f32le --enable-demuxer=xwma,wav --enable-swresample --enable-w32threads`
- `deps/dml/microsoft.ml.onnxruntime.directml` (`ORT`): the NuGet package Microsoft.ML.OnnxRuntime.DirectML 1.24.4, extracted
- `deps/dml/microsoft.ai.directml` (`DML`): the NuGet package Microsoft.AI.DirectML 1.15.4, extracted
- `models/mark/network_fp16.onnx` (`MODEL`): `network.onnx` from [nvidia/Audio2Face-3D-v2.3-Mark](https://huggingface.co/nvidia/Audio2Face-3D-v2.3-Mark), converted with
  `onnxconverter_common.float16.convert_float_to_float16(model, keep_io_types=True)`

Commands:

- `./build.sh` builds into `build/mod`.
- `./build.sh install <dir>` also copies the result into a mod folder.
- `./build.sh release` also writes the release archives to `release/`.

## License

GNU General Public License v3.0; see [LICENSE](LICENSE).

Additional term under GPL v3 section 7(b): any copy or modified version must keep the attribution "Based on Better Facial Animations by Cyberquach" in its documentation and on any page it is distributed from.

Bundled third-party components keep their own licenses, which are included in the release archive:

- FFmpeg: LGPL 2.1+
- ONNX Runtime: MIT
- DirectML: Microsoft
- the Audio2Face-3D model: NVIDIA Open Model License (`dist/licenses`)
