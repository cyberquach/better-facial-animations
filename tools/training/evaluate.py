"""Evaluate a saved checkpoint on an explicitly named, hash-verified data split."""
import argparse
import json
from pathlib import Path

import torch

from tools.training.common import atomic_json, load_checkpoint, sha
from tools.training.train import Dataset, create_model, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='latest.json or best.json')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'validation', 'test'], required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if min(args.batch, args.threads) < 1:
        raise ValueError('Batch and threads must be positive')
    torch.set_num_threads(args.threads)
    state, pointer = load_checkpoint(args.checkpoint)
    if sha(args.data/'dataset.json') != state['config']['dataset_sha256']:
        raise ValueError('Evaluation corpus differs from the checkpoint manifest')
    data = Dataset(args.data, args.split)
    model = create_model(state['kind'], state['width']).to(args.device)
    model.load_state_dict(state['model'], strict=True)
    result = {'checkpoint_sha256': pointer['sha256'], 'split': args.split,
              'scope': data.manifest['scope'], 'metrics': evaluate(model, data, args.device, args.batch)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
