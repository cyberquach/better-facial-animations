"""Folded Audio2Face-3D v3.0 latents for a set of voice lines (training data for the A2F Direct re-PCA).

  venv/bin/python tools/a2f3_bake.py <identity 0-2> 'work/RiverwoodLocation/**/*.wav' [--threads N] [--part i/n]

Writes <line>.a2f3_<Identity>.npy (frames x 1043 at 60 fps) next to each line. Every other line (sorted order) gets a
Skyrim emotion (type 1..7 in turn, value 50..100) so the basis also spans emotional speech; the rest are neutral.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2f3_ref as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("identity", type=int)
    ap.add_argument("pattern")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(R.FOLDED, so, providers=["CPUExecutionProvider"])
    files = sorted(glob.glob(a.pattern, recursive=True))
    i, n = map(int, a.part.split("/"))
    name = R.IDENTITIES[a.identity]
    for k, f in enumerate(files):
        if k % n != i:
            continue
        out = os.path.splitext(f)[0] + f".a2f3_{name}.npy"
        if os.path.exists(out):
            continue
        pcm = R.load_pcm(f)
        emo = R.skyrim_emotion(R.num_frames(len(pcm)), 1 + (k // 2) % 7, 50 + (k * 37) % 51) if k % 2 else None
        np.save(out, R.run_latents(sess, pcm, a.identity, emo).astype(np.float16))
    print(name, a.part, "done")


if __name__ == "__main__":
    sys.exit(main())
