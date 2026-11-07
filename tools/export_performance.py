"""Export the 6.0 Block AttnRes checkpoint with fixed, masked KV memory.

python tools/export_performance.py --checkpoint checkpoint.pt --sha256 HASH --out out
The checkpoint must be trusted. No worker, network or private-path dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.performance_student import ContextStudent, MEMORY, CONTEXT
from tools.training.face import mel_features

HOP = 90
NAMES = ['mel', 'identity', 'emotion', 'past_k', 'past_v', 'past_frames']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def export(checkpoint, expected_sha, out):
    if sha(checkpoint) != expected_sha:
        raise ValueError('Checkpoint SHA-256 mismatch')
    torch.set_num_threads(2)
    torch.manual_seed(613)
    s = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = ContextStudent().eval()
    model.load_state_dict(s['model'], strict=True)
    if not all(torch.isfinite(x).all() for x in model.state_dict().values()):
        raise ValueError('Nonfinite weights')
    out.mkdir(parents=True, exist_ok=True)
    def inputs(valid=0, identity=0):
        return [torch.randn(1, HOP+2*CONTEXT, 80), torch.tensor([identity]),
                torch.rand(1, HOP+2*CONTEXT, 10)*.5,
                torch.randn(6, 1, 4, MEMORY, 64)*.2,
                torch.randn(6, 1, 4, MEMORY, 64)*.2, torch.tensor([valid])]
    artifact = out/'performance.onnx'
    torch.onnx.export(model, tuple(inputs()), artifact, input_names=NAMES,
                      output_names=['normalized_motion', 'next_k', 'next_v'],
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(artifact))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(str(artifact), options, providers=['CPUExecutionProvider'])
    errors = []
    motion_errors, state_errors = [], []
    def compare(a, b, state=False):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=5e-4 if state else 1e-4)
        errors.append(float(np.max(np.abs(a-b))))
        (state_errors if state else motion_errors).append(errors[-1])
    with torch.inference_mode():
        # Full fixed cache must equal the training model's variable cache,
        # including empty/partial history and arbitrary masked leading slots.
        for identity in (0, 1):
            for valid in (0, 1, 90, 180, 255, 256):
                args = inputs(valid, identity)
                sliced = [x[:, :, :, MEMORY-valid:, :] for x in args[3:5]]
                expected = model(*args[:3], *sliced)
                fixed = model(*args)
                actual = runtime.run(None, dict(zip(NAMES, [x.numpy() for x in args])))
                compare(actual[0], expected[0].numpy())
                for i, (x, y) in enumerate(zip(actual, fixed)):
                    compare(x, y.numpy(), state=i > 0)
                for x, y in zip(actual[1:], expected[1:]):
                    compare(x[:, :, :, -y.shape[3]:], y.numpy(), state=True)
        # Seven chunks exceed the memory horizon and include a short final tail.
        # Compare to a single full-sequence forward, not the same export graph.
        count = 577
        mel = torch.randn(1, count+2*CONTEXT, 80)
        emo = torch.rand(1, count+2*CONTEXT, 10)*.5
        for identity in (0, 1):
            ident = torch.tensor([identity])
            empty = model.empty_state(1)
            expected = model(mel, ident, emo, empty, empty)[0].numpy()
            padded_mel = torch.nn.functional.pad(mel, (0, 0, 0, HOP))
            padded_emo = torch.nn.functional.pad(emo, (0, 0, 0, HOP))
            k = v = np.zeros((6, 1, 4, MEMORY, 64), np.float32)
            predictions = []
            for start in range(0, count, HOP):
                feed = [padded_mel[:, start:start+HOP+2*CONTEXT].numpy(), ident.numpy(),
                        padded_emo[:, start:start+HOP+2*CONTEXT].numpy(), k, v,
                        np.array([min(start, MEMORY)], np.int64)]
                motion, k, v = runtime.run(None, dict(zip(NAMES, feed)))
                predictions.append(motion[:, :min(HOP, count-start)])
            compare(np.concatenate(predictions, 1), expected)
    stats = {k: v.numpy() for k, v in s['stats'].items()}
    np.savez(out/'performance-normalization.npz', **stats)
    subprocess.run([sys.executable, str(Path(__file__).with_name('export_performance_stats.py')),
                    str(out/'performance-normalization.npz'), str(out/'performance_stats.h')], check=True)
    # Portable native parity fixtures: synthetic audio long enough to fill/roll
    # the cache, plus both identities. Reference uses the original full forward.
    samples = 160017
    t = np.arange(samples, dtype=np.float64)/16000
    pcm = (.15*np.sin(2*np.pi*(170*t+9*t*t))*(.6+.4*np.sin(2*np.pi*.8*t))
           + .025*np.sin(2*np.pi*730*t)).astype(np.float32)
    features = mel_features(pcm)
    n = len(features)
    raw = np.full((n+2*CONTEXT, 80), np.log(1e-6), np.float32)
    raw[CONTEXT:CONTEXT+n] = features
    mel = torch.from_numpy((raw-stats['mel_mean'])/stats['mel_std'])[None]
    # Matches native Emotion{1,100,1,0}: anger1, no loudness swell.
    emotion = torch.zeros(1, n+2*CONTEXT, 10)
    emotion[:, :, 1] = 1.
    refs = []
    with torch.inference_mode():
        for identity in (0, 1):
            empty = model.empty_state(1)
            pred = model(mel, torch.tensor([identity]), emotion, empty, empty)[0][0].numpy()
            refs.append((pred*stats['target_std']+stats['target_mean']).astype(np.float32))
    with (out/'performance.fixtures').open('wb') as f:
        f.write(np.array([len(pcm), n], '<u4').tobytes())
        f.write(pcm.astype('<f4').tobytes())
        for ref in refs:
            f.write(ref.astype('<f4').tobytes())
    last_eval = next((r for r in reversed(s['history']) if 'evaluation' in r), None)
    receipt = dict(architecture='BlockAttnRes256-6layer-KV256-block2',
        parameters=sum(p.numel() for p in model.parameters()), checkpoint_sha256=expected_sha,
        completed_passes=s['epoch'], optimizer_steps=s['steps'], trained_identity_frames=s['trained_frames'],
        dataset_sha256=s['config']['dataset_sha256'], onnx_sha256=sha(artifact),
        onnx_bytes=artifact.stat().st_size, normalization_sha256=sha(out/'performance-normalization.npz'),
        normalization_header_sha256=sha(out/'performance_stats.h'),
        fixtures_sha256=sha(out/'performance.fixtures'), export_max_abs_error=max(errors),
        normalized_motion_max_abs_error=max(motion_errors), kv_max_abs_error=max(state_errors),
        parity_tolerances=dict(rtol=2e-4, normalized_motion_atol=1e-4, unscaled_kv_atol=5e-4),
        input='16kHz mono -> 30Hz mel80;120input frames;identity0Claire/1Mark;emotion10',
        output='90normalized frames;48skin+10tongue+15jaw;no gaze',
        state=dict(shape=[6,1,4,256,64],inputs=['past_k','past_v','past_frames'],
                   reset='Zero valid frames for every utterance, including after cancellation/failure',
                   update='Keep last256keys/values;valid_frames=min(256,previous+90)'),
        latest_full_evaluation=last_eval,
        limitations='Latest checkpoint, not selected best. Full-corpus evaluation belongs to the earlier pass recorded above. Closed-corpus fit; no in-game or Steam Deck certification.',
        source_sha256={name:sha(Path(__file__).parent/name) for name in
                       ['export_performance.py','performance_student.py']})
    (out/'performance-model.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--sha256', required=True)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    result = export(args.checkpoint, args.sha256, args.out)
    print(json.dumps({k:result[k] for k in ['optimizer_steps','parameters','onnx_bytes','export_max_abs_error']}))
