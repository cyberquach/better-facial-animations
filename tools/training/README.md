# Performance model training

These local workflows train Audio2Emotion's width-512 temporal student and the
legacy Audio2Face Thin120 student. BFA 6.0 uses a different, stateful Block AttnRes
face architecture defined in `tools/performance_student.py`. The Thin120 training
and export commands below produce legacy candidates, not 6.0-compatible weights.

Export a trusted Block AttnRes checkpoint with (the width is its config's; the
shipped model is width 608, the 6.0 one 256):

```sh
python tools/export_performance.py --checkpoint checkpoint.pt --sha256 CHECKPOINT_SHA256 --out work/face-export
```

This checks fixed-cache ONNX against variable-cache PyTorch and full-sequence
predictions, and writes `performance.fixtures` for native parity. Update that
fixture, the ONNX, metadata, normalization and generated header together.
The generic trainer below does not yet implement this stateful training recipe.
They accept your own audio and teacher files. No SSH, remote worker service,
host discovery, machine-specific paths, or automatic asset downloads are used.

Run commands from the repository root with Python 3.10 or newer. Create a virtual
environment and install `tools/training/requirements.txt`; install FFmpeg separately
and make it available on PATH (or pass `--ffmpeg`). CPU dependencies are pinned;
FFmpeg 8.0.1 was used for verification. On Windows use `.venv-training\Scripts\python`
instead of `.venv-training/bin/python` in the commands below.

```sh
python -m venv .venv-training
.venv-training/bin/python -m pip install -r tools/training/requirements.txt
.venv-training/bin/python -m unittest discover -s tests -p test_portable_training.py
```

The integration tests use generated audio and synthetic teachers to verify the
pipeline, exact CPU interrupted/resumed weights, export parity, and rejection of
changed data/protocols. They do not establish trained model quality.

## Teachers and audio

Supply legally obtained game/mod dialogue or your own recordings. `--audio-root`
accepts a file or directory and may be repeated. WAV, FUZ, XWM, OGG, MP3 and FLAC
are decoded to 16 kHz mono float PCM; BSA v104/v105 audio members are read directly
without extraction. Directory symlinks are not followed. All supported audio under
the explicitly supplied roots is included, independently of mod-manager activation.
Duplicate decoded PCM is labeled once. Decode/inference errors stop the command;
there is no silent exclusion. Use a separate curated root to exclude unwanted audio.

Teachers are self-contained ONNX files, verified against your explicit SHA256.
The Audio2Emotion teacher is the v2.2 model with 60,000-sample input and six logits.
`tools/a2e_fp16.py` prepares the Full model used by the plugin. Audio2Face requires
the folded v3.0 latent teacher produced by `tools/a2f3_fold.py` and its matching
v9 `A2FDirect3.bin` projection. Follow those tools' preparation commands when
starting from upstream weights. Do not mix a newly folded teacher/noise seed with
an unrelated projection. The projection defaults to the pinned hash below;
for a custom matching projection, pass `--projection-sha256` explicitly.
Reference assets used by the shipped model lineage:

| Asset | SHA256 |
|---|---|
| Full emotion FP16 teacher | `29f075c18a559bc203459463cf474f38ef104f142ddedda1e241284c536098a1` |
| Face folded FP16 teacher | `841cc6a2ea39babe3fe6377de21ddf8d90160c33ebac67a2085ce733b3a95a19` |
| Face v9 projection | `8741ff3b457cdd68003f55436f15c7b13de9f3e53d498b3ffaebd789870fec3c` |

The Audio2Emotion model agreement restricts it and its derivatives to the
Audio2Face project, not emotion recognition. Retain the model attribution and
applicable licenses in `dist/licenses/` and `dist/NOTICE.txt`. Training does not
grant permission to redistribute game/mod audio. Weights and audio are not
downloaded or bundled by these scripts.

## Prepare and train

The example asset paths are repository-relative or user-supplied placeholders.
Outputs should remain outside tracked source, for example in the ignored `work/`.

```sh
python -m tools.training.prepare --kind emotion --audio-root /path/to/dialogue \
  --teacher models/a2e/network_fp16.onnx \
  --teacher-sha256 29f075c18a559bc203459463cf474f38ef104f142ddedda1e241284c536098a1 \
  --out work/emotion-data
python -m tools.training.train --data work/emotion-data/packed --out work/emotion-run \
  --epochs 300 --batch 512 --device cpu

python -m tools.training.prepare --kind face --audio-root /path/to/dialogue \
  --teacher models/v3/network_fp16.onnx \
  --teacher-sha256 841cc6a2ea39babe3fe6377de21ddf8d90160c33ebac67a2085ce733b3a95a19 \
  --projection dist/A2FDirect3.bin --out work/face-data
python -m tools.training.train --data work/face-data/packed --out work/face-run \
  --epochs 200 --batch 1024 --device cpu
```

Activate the training environment first, or replace `python` with its executable.
Smaller batches reduce memory requirements. Training also accepts `--device cuda:0`
with a compatible CUDA-enabled PyTorch installation. Teacher preparation accepts
`--provider cpu|cuda|dml --device-id N`. For CUDA teachers, replace the CPU
`onnxruntime` package with `onnxruntime-gpu==1.23.2`; install the corresponding
CUDA libraries. DirectML teachers require ONNX Runtime DirectML on Windows.
Only one ORT package should be installed in an environment. Missing requested
providers fail explicitly; individual unsupported GPU operators may still use
ORT's CPU kernels. Student training supports CPU/CUDA, not DirectML.

Preparation resumes verified per-record labels when rerun with identical options.
It stores labels and then a packed disk-mapped copy: reserve space for both.
Emotion packed arrays use approximately 18,072 bytes per window; face arrays use
452 bytes per frame, plus record metadata. Both retain normalization and source
hashes. Paths, hostnames and adapter names are omitted from manifests; diagnostic
tracebacks may contain local paths, so do not commit generated logs.

Training uses AdamW, zero weight decay, clipping at 1 and cosine learning-rate
decay to 5% of the initial rate. Emotion loss is teacher-to-student KL plus
0.05 times centered-logit MSE. Face loss is normalized position MSE plus 0.25
velocity MSE, using 90 output frames with 17 context frames on either side,
both identities, and masked context/tail targets. Face teacher targets retain
the original deterministic neutral/one-emotion-at-0.5 policy and 60-to-30 Hz
sampling. Normalization uses training records only. Evaluation is every five
epochs and the final epoch; `--evaluate-every` changes this.

The default is closed-corpus fitting: every recording is training data. Its
metrics are teacher agreement/reconstruction, not held-out accuracy. To reserve
data, prepare with `--validation-fraction 0.1 --test-fraction 0.1`, then train with
`--selection-split validation`. Splits are determined by decoded PCM hashes,
prevent exact-duplicate leakage, and are **not speaker-disjoint**. A tiny corpus
may leave a split empty; the requested operation then fails explicitly.

## Resume, evaluate and export

Training writes checksum-verified `latest.json` and `best.json` pointers. Best
means lowest evaluated KL for emotion or normalized MSE for face on the chosen
selection split. Recovery retains the latest, previous and best payloads. Saves
occur at epoch boundaries and every `--checkpoint-steps` updates (default 1000).
Append `--resume` to the unchanged training command to restore optimizer, RNG,
epoch and next-batch cursor. CPU resume equivalence is tested; cross-device or
cross-version bitwise reproduction is not claimed. Use a new run for protocol
changes. `--max-updates N` stops at a cumulative update boundary; alternatively
create an empty `STOP` file in the run directory, then remove it before resuming.
A process killed during an optimizer step resumes from its last durable save.
Run only one writer per output directory. An exclusive `writer.lock` enforces
this. After a forcibly killed writer, confirm it is no longer running before
removing its stale lock and resuming.

```sh
python -m tools.training.evaluate --checkpoint work/emotion-run/best.json \
  --data work/emotion-data/packed --split train --output work/emotion-evaluation.json
python -m tools.training.export --checkpoint work/emotion-run/best.json --out work/emotion-export
python -m tools.training.export --checkpoint work/face-run/best.json --out work/face-export
```

Use `--split test` only when you reserved a test split; it is never used for
training or automatic checkpoint selection. Export verifies ONNX/checkpoint
parity and records timing scope and provenance. Emotion export includes synthetic
native reference fixtures. Face export includes `performance.onnx`,
`normalization.npz`, `performance_stats.h` and `performance-model.json`.

To promote an emotion candidate, update `dist/emotion_performance.onnx`, its JSON
and `tests/emotion_performance.fixtures` together. For the legacy face runtime, update
`dist/performance.onnx`, `dist/performance-model.json`,
`dist/performance-normalization.npz` and `include/a2f/inference/performance_stats.h`
together. The shipped `performance.onnx` stores its weights as float16
(`tools/performance_fp16.py <float32.onnx> <out.onnx>`, needs `onnx`); its
`tests/performance.fixtures` reference uses the same float16-rounded weights. Run `./build.sh compat`, the native model/parity checks in
`tests/README.md`, and matched gameplay quality/performance checks before release.
No training/export command modifies the shipped models automatically.

These are from-scratch retraining recipes, not bit-for-bit reproductions of the
bundled weights. The historical Thin120 was widened from a previously trained
Thin96; the shipped emotion run used a different device/backend and historical
batch changes. New corpora, normalization, initialization and hardware change
the optimization trajectory. The original shipped model provenance remains in
`dist/performance-model.json` and `dist/emotion_performance.json`.
