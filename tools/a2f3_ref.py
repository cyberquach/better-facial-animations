"""Reference front end for Audio2Face-3D v3.0 (diffusion), mirroring the SDK's
offline executor: 1 s windows every 0.5 s, GRU latents chained between
windows, the 30 centre frames of each window kept, output at 60 fps.

  venv/bin/python tools/a2f3_ref.py <audio> <out dir> [--identity N] [--model PATH] [--fresh-noise]

Writes into <out dir>: pcm.f32 (16 kHz mono), noise.f32 (one window,
3x60x88831), pred.f32 (frames x 88831, delta from the identity's neutral)
and info.txt (samples, frames, windows, identity).
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_DIR = os.path.join(ROOT, "sdk/_data/audio2face-models/audio2face-3d-v3.0")
SR, WIN, PAD, STRIDE = 16000, 16000, 16000, 8000
FPS, LEFT, CENTER = 60, 15, 30
SKIP = 45  # frames emitted before t = 0 (SDK: negative timestamps dropped)
DIM = 88831
SKIN, TONGUE, JAW, EYES = 72006, 16806, 15, 4
IDENTITIES = ("Claire", "James", "Mark")


def load_pcm(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def num_frames(n):
    return -(-n * FPS // SR)


def num_windows(frames):
    return (SKIP - 1 + frames) // CENTER + 1


def window(pcm, i, gain=1.0):
    """Samples [STRIDE*i - PAD, STRIDE*i - PAD + WIN), zero outside the clip."""
    out = np.zeros(WIN, np.float32)
    a = STRIDE * i - PAD
    lo, hi = max(a, 0), min(a + WIN, len(pcm))
    if hi > lo:
        out[lo - a:hi - a] = pcm[lo:hi]
    return out * gain


def make_noise(rng):
    return rng.standard_normal((1, 3, 2 * CENTER, DIM), dtype=np.float32)


def run(sess, pcm, identity, emotion=None, noise=None, fresh_noise=False, seed=0, gain=1.0):
    """pred (frames, DIM) for the whole clip. emotion: (frames, 10) or None."""
    frames = num_frames(len(pcm))
    rng = np.random.default_rng(seed)
    if noise is None:
        noise = make_noise(rng)
    ident = np.zeros((1, 3), np.float32)
    ident[0, identity] = 1
    lat = np.zeros((2, 2, 1, 256), np.float32)
    pred = np.zeros((frames, DIM), np.float32)
    for i in range(num_windows(frames)):
        f0 = CENTER * i - SKIP  # output frame of centre frame 0
        emo = np.zeros((1, CENTER, 10), np.float32)
        if emotion is not None:
            idx = np.clip(np.arange(f0, f0 + CENTER), 0, frames - 1)
            emo[0] = emotion[idx]
        if fresh_noise and i:
            noise = make_noise(rng)
        y, lat = sess.run(["prediction", "output_latents"],
                          {"window": window(pcm, i, gain)[None], "identity": ident,
                           "emotion": emo, "input_latents": lat, "noise": noise})
        for j in range(CENTER):
            f = f0 + j
            if 0 <= f < frames:
                pred[f] = y[0, LEFT + j]
    return pred, noise


FOLDED = os.path.join(ROOT, "models/v3/network_folded.onnx")
LATENT = 512 + 512 + 19  # skin latent, tongue latent, jaw/eyes per frame of the folded network

# Skyrim emotion type (1 Anger 2 Disgust 3 Fear 4 Sad 5 Happy 6 Surprise 7 Puzzled) -> the 10 network emotions
# (amazement anger cheekiness disgust fear grief joy outofbreath pain sadness); same map as src/live_core.cc
EMOTION_MAP = {1: {1: 1}, 2: {3: 1}, 3: {4: 1}, 4: {9: 1}, 5: {6: 1}, 6: {0: 1}, 7: {0: 0.3, 2: 0.3}}


def skyrim_emotion(frames, etype, value):
    e = np.zeros((frames, 10), np.float32)
    for k, v in EMOTION_MAP.get(etype, {}).items():
        e[:, k] = v * min(max(value / 100, 0), 1)
    return e


def run_latents(sess, pcm, identity, emotion=None):
    """Folded network over the whole clip -> (frames, LATENT) at 60 fps."""
    frames = num_frames(len(pcm))
    ident = np.eye(3, dtype=np.float32)[identity:identity + 1]
    lat = np.zeros((2, 2, 1, 256), np.float32)
    out = np.zeros((frames, LATENT), np.float32)
    for i in range(num_windows(frames)):
        f0 = CENTER * i - SKIP
        emo = np.zeros((1, CENTER, 10), np.float32)
        if emotion is not None:
            emo[0] = emotion[np.clip(np.arange(f0, f0 + CENTER), 0, frames - 1)]
        s, t, ge, lat = sess.run(None, {"window": window(pcm, i)[None], "identity": ident, "emotion": emo,
                                        "input_latents": lat})
        for j in range(CENTER):
            if 0 <= f0 + j < frames:
                out[f0 + j] = np.concatenate([s[0, j], t[0, j], ge[0, j]])
    return out


def rest_latent(sess, identity, seconds=2.0):
    """Silence rest pose of the folded network: mean over a silent clip."""
    return run_latents(sess, np.zeros(int(SR * seconds), np.float32), identity).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--identity", type=int, default=2)
    ap.add_argument("--model", default=os.path.join(MODEL_DIR, "network.onnx"))
    ap.add_argument("--fresh-noise", action="store_true")
    a = ap.parse_args()
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
    pcm = load_pcm(a.audio)
    pred, noise = run(sess, pcm, a.identity, fresh_noise=a.fresh_noise)
    os.makedirs(a.out, exist_ok=True)
    pcm.tofile(os.path.join(a.out, "pcm.f32"))
    noise.tofile(os.path.join(a.out, "noise.f32"))
    pred.tofile(os.path.join(a.out, "pred.f32"))
    with open(os.path.join(a.out, "info.txt"), "w") as f:
        f.write(f"{len(pcm)} {len(pred)} {num_windows(len(pred))} {a.identity}\n")
    skin = pred[:, :SKIN]
    print(f"{len(pcm)} samples, {len(pred)} frames, {num_windows(len(pred))} windows; "
          f"skin |max| {np.abs(skin).max():.3f}, jaw y range "
          f"{pred[:, SKIN + TONGUE + 1:SKIN + TONGUE + JAW:3].min():.3f}..{pred[:, SKIN + TONGUE + 1:SKIN + TONGUE + JAW:3].max():.3f}")


if __name__ == "__main__":
    sys.exit(main())
