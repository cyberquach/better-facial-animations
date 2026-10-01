# A2F Live

Real-time NVIDIA Audio2Face-3D (v2.3 Mark) lip-sync for Skyrim SE 1.6.1170, as an SKSE plugin.

When an actor with a FaceGen head speaks, the plugin runs the voice file through the Audio2Face network on the GPU (ONNX Runtime + DirectML). It retargets the result onto the vanilla phoneme, modifier and expression morphs, which play in the head's FaceGen update. There is an optional settings page through SKSE Menu Framework.

- `src/A2FLive.cpp`: hooks, worker thread, cache, playback, ini and settings page
- `src/live_core.*`: audio decoding, Audio2Face SDK pipeline (windows, emotion, skin animator), retarget solve
- `src/ort_net.*`: ONNX Runtime C API wrapper (loaded at runtime)
- `src/skse.h`: SKSE ABI, log, Address Library, engine layout
- `dist/`: ini, retarget data (`A2FLive.dat`), player readme, notices, model license

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

GNU General Public License v3.0; see [LICENSE](LICENSE). Bundled third-party components keep their own licenses, which are included in the release archive:

- FFmpeg: LGPL 2.1+
- ONNX Runtime: MIT
- DirectML: Microsoft
- the Audio2Face-3D model: NVIDIA Open Model License (`dist/licenses`)
