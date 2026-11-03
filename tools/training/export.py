"""Export a trusted training pointer with ONNX parity and deployment provenance."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import onnx
import torch

from tools.training.common import atomic_json, exclusive_output, finite, load_checkpoint, session, sha
from tools.training.train import create_model


@exclusive_output
def export(args):
    torch.set_num_threads(2)
    state, pointer = load_checkpoint(args.checkpoint)
    model = create_model(state['kind'], state['width']).eval()
    model.load_state_dict(state['model'], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Nonfinite checkpoint')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tools = Path(__file__).resolve().parents[1]
    if state['kind'] == 'emotion':
        artifact = out/'emotion_performance.onnx'
        subprocess.run([sys.executable, str(tools/'export_emotion.py'), '--checkpoint',
                        str(args.checkpoint.parent/pointer['checkpoint']), '--sha256', pointer['sha256'],
                        '--output', str(artifact)], check=True)
        receipt = json.loads(artifact.with_suffix('.json').read_text())
    else:
        artifact = out/'performance.onnx'
        g = torch.Generator().manual_seed(917)
        sample = (torch.randn(1, 124, 80, generator=g), torch.tensor([0]), torch.zeros(1, 124, 10))
        torch.onnx.export(model, sample, str(artifact), input_names=['mel', 'identity', 'emotion'],
                          output_names=['normalized_motion'], opset_version=17, dynamo=False,
                          dynamic_axes={name: {0: 'batch', 1: 'frames'} for name in
                                        ('mel', 'emotion', 'normalized_motion')} | {'identity': {0: 'batch'}})
        onnx.checker.check_model(onnx.load(artifact))
        runtime = session(artifact)
        errors = []
        for length in (1, 30, 124, 287):
            inputs = (torch.randn(2, length, 80, generator=g), torch.tensor([0, 1]),
                      torch.rand(2, length, 10, generator=g)*.5)
            with torch.no_grad():
                expected = model(*inputs).numpy()
            actual = runtime.run(None, dict(zip(('mel', 'identity', 'emotion'), (v.numpy() for v in inputs))))[0]
            finite(actual)
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)
            errors.append(float(np.max(np.abs(actual-expected))))
        feed = dict(zip(('mel', 'identity', 'emotion'), (v.numpy() for v in sample)))
        timings = []
        for i in range(35):
            start = time.perf_counter()
            runtime.run(None, feed)
            if i >= 5:
                timings.append((time.perf_counter()-start)*1000)
        stats = {k: v.numpy() for k, v in state['stats'].items()}
        np.savez(out/'normalization.npz', **stats)
        subprocess.run([sys.executable, str(tools/'export_performance_stats.py'),
                        str(out/'normalization.npz'), str(out/'performance_stats.h')], check=True)
        receipt = {'architecture': f'Thin{state["width"]}', 'parameters': sum(p.numel() for p in model.parameters()),
                   'checkpoint_sha256': pointer['sha256'], 'onnx_sha256': sha(artifact),
                   'onnx_bytes': artifact.stat().st_size, 'export_max_abs_error': max(errors),
                   'normalization_sha256': sha(out/'normalization.npz'),
                   'normalization_header_sha256': sha(out/'performance_stats.h'),
                   'cpu_model_median_ms': float(np.median(timings)),
                   'timing_scope': 'CPU, two threads, 124 frames; excludes frontend and gameplay',
                   'input': 'normalized mel [B,T,80], identity int64 [B] (0 Claire,1 Mark), emotion [B,T,10]',
                   'output': 'normalized motion [B,T,73]:48 skin+10 tongue+15 jaw; no gaze',
                   'license': 'NVIDIA Open Model License; retain teacher attribution', 'in_game_tested': False}
    receipt.update(training={'epochs_completed': state['epoch'], 'steps': state['steps'],
                             'dataset_sha256': state['config']['dataset_sha256'],
                             'last_evaluation': next((r for r in reversed(state['history']) if 'evaluation' in r), None),
                             'initialization': 'from scratch; not the historical shipped checkpoint trajectory'},
                   teacher_sha256=state['config']['teacher_sha256'],
                   projection_sha256=state['config']['projection_sha256'],
                   data_scope=state['config']['data_scope'])
    metadata = artifact.with_suffix('.json') if state['kind'] == 'emotion' else out/'performance-model.json'
    atomic_json(metadata, receipt)
    print(json.dumps(receipt, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='Trusted local latest.json or best.json')
    p.add_argument('--out', type=Path, required=True)
    export(p.parse_args())


if __name__ == '__main__':
    main()
