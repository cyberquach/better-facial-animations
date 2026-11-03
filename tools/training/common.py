"""Portable provenance, atomic writes and explicitly selected local devices."""
import hashlib
from functools import wraps
import json
import os
from pathlib import Path

import numpy as np


def exclusive_output(function):
    @wraps(function)
    def run(args):
        args.out.mkdir(parents=True, exist_ok=True)
        lock = args.out / 'writer.lock'
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise RuntimeError('Output is locked; after an unclean exit, verify no writer remains before removing writer.lock') from None
        os.close(fd)
        try:
            return function(args)
        finally:
            lock.unlink()
    return run


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def code_hashes():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / 'training').glob('*.py'))
    files += [root / n for n in ('emotion_student.py', 'a2f3_ref.py')]
    return {p.relative_to(root).as_posix(): sha(p) for p in files}


def finite(*arrays):
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite input, label or model output')


def session(path, provider='cpu', device_id=0, threads=2):
    import onnxruntime as ort
    names = {'cpu': 'CPUExecutionProvider', 'cuda': 'CUDAExecutionProvider',
             'dml': 'DmlExecutionProvider'}
    name = names[provider]
    if name not in ort.get_available_providers():
        raise RuntimeError(f'Requested provider unavailable: {name}')
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = provider != 'dml'
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if provider == 'cpu':
        # Match the plugin: ENABLE_ALL's layout fusion breaks this FP16 teacher.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    providers = [name] if provider == 'cpu' else [(name, {'device_id': device_id})]
    result = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    result.disable_fallback()
    if result.get_providers()[0] != name:
        raise RuntimeError('Requested execution provider was not selected')
    return result


def save_checkpoint(out, state, best=False):
    """Durable payload before pointers; retain best and two recovery snapshots."""
    import torch
    out = Path(out)
    temp = out / 'checkpoint.tmp'
    with temp.open('wb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    digest = sha(temp)
    name = 'checkpoint-' + digest + '.pt'
    temp.replace(out / name)
    old = json.loads((out / 'latest.json').read_text()) if (out / 'latest.json').exists() else {}
    pointer = {'checkpoint': name, 'sha256': digest, 'epoch': state['epoch'],
               'batch_cursor': state['batch_cursor'], 'steps': state['steps']}
    if best:
        atomic_json(out / 'best.json', pointer)
    pointer['previous'] = old.get('checkpoint')
    atomic_json(out / 'latest.json', pointer)
    keep = {name, pointer['previous']}
    if (out / 'best.json').exists():
        keep.add(json.loads((out / 'best.json').read_text())['checkpoint'])
    for p in out.glob('checkpoint-*.pt'):
        if p.name not in keep:
            p.unlink()
    return pointer


def load_checkpoint(pointer):
    import torch
    pointer = Path(pointer)
    meta = json.loads(pointer.read_text())
    name = 'checkpoint-' + meta['sha256'] + '.pt'
    if meta['checkpoint'] != name or len(meta['sha256']) != 64:
        raise ValueError('Invalid checkpoint pointer')
    path = pointer.parent / name
    if sha(path) != meta['sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    return torch.load(path, map_location='cpu', weights_only=True), meta
