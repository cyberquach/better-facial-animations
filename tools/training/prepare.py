"""Decode local audio/BSA/FUZ, label with a pinned teacher, and pack disk arrays."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import torch

from tools import a2f3_ref
from tools.emotion_student import Frontend, window_starts
from tools.training.archive import BSA
from tools.training.common import atomic_json, code_hashes, exclusive_output, finite, fingerprint, session, sha
from tools.training.face import load_projection, mel_features, project_latents

AUDIO = {'.fuz', '.wav', '.xwm', '.ogg', '.mp3', '.flac'}
PROJECTION_SHA256 = '8741ff3b457cdd68003f55436f15c7b13de9f3e53d498b3ffaebd789870fec3c'


def sources(roots):
    for root in roots:
        root = Path(root).resolve(strict=True)
        if root.is_file():
            paths = [root]
        else:
            paths = (Path(parent) / name for parent, dirs, files in os.walk(root)
                     for name in sorted(files))
        for path in paths:
            if path.is_symlink():
                continue
            if path.suffix.lower() == '.bsa':
                archive = BSA(path)
                for name in sorted(archive.files):
                    if Path(name).suffix.lower() in AUDIO:
                        yield archive.read(name)
            elif path.suffix.lower() in AUDIO:
                yield path.read_bytes()


def decode(raw, ffmpeg):
    if raw[:4] == b'FUZE':
        if len(raw) < 12:
            raise ValueError('Truncated FUZ header')
        offset = 12 + struct.unpack_from('<I', raw, 8)[0]
        if offset >= len(raw):
            raise ValueError('FUZ has no audio payload')
        raw = raw[offset:]
    result = subprocess.run([ffmpeg, '-v', 'error', '-threads', '1', '-i', 'pipe:0',
                             '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1'],
                            input=raw, capture_output=True, timeout=120)
    if result.returncode:
        raise ValueError('Audio decoding failed: ' + result.stderr.decode(errors='replace')[-500:])
    pcm = np.frombuffer(result.stdout, '<f4').copy()
    if not 0 < len(pcm) <= 16000 * 600:
        raise ValueError('Expected nonempty audio of at most ten minutes')
    finite(pcm)
    return pcm


def label_emotion(teacher, pcm):
    starts = window_starts(len(pcm))
    features, targets = [], []
    front = Frontend().eval()
    key = teacher.get_inputs()[0].name
    for start in starts:
        x = np.zeros((1, 60000), np.float32)
        part = pcm[start:start + 60000]
        x[0, :len(part)] = part
        targets.append(teacher.run(None, {key: x})[0].reshape(1, 6))
        with torch.no_grad():
            features.append(front(torch.from_numpy(x)).numpy())
    return {'x': np.concatenate(features).astype(np.float16),
            'y': np.concatenate(targets).astype(np.float32)}


def label_face(teacher, pcm, faces, digest):
    # Same deterministic neutral / one emotion0.5 policy as the original fit.
    condition = int(digest[:8], 16) % 14
    emotion = a2f3_ref.skyrim_emotion(a2f3_ref.num_frames(len(pcm)),
                                    0 if condition < 7 else condition - 6, 50)
    target = [project_latents(a2f3_ref.run_latents(teacher, pcm, identity, emotion), face)[::2]
              for identity, face in zip((0, 2), faces)]
    mel = mel_features(pcm)
    if np.asarray(target).shape != (2, len(mel), 73):
        raise ValueError('Teacher/projection output shape mismatch')
    return {'x': mel.astype(np.float16), 'y': np.stack(target, axis=1).astype(np.float16),
            'emotion': emotion[0]}


def pack(out, config, rows):
    """Memory maps keep the full corpus off RAM; stats use training records only."""
    packed = out / 'packed'
    packed.mkdir(exist_ok=True)
    manifest = packed / 'dataset.json'
    identity = fingerprint({'config': config, 'records': rows})
    if manifest.exists():
        prior = json.loads(manifest.read_text())
        if prior['identity'] != identity:
            raise ValueError('Corpus changed; use a new output directory')
        for name, digest in prior['files'].items():
            if sha(packed / name) != digest:
                raise ValueError('Packed data checksum mismatch')
        return prior
    count = sum(r['count'] for r in rows)
    emotion = config['kind'] == 'emotion'
    shapes = {'x': ((count, 48, 188) if emotion else (count, 80), np.float16),
              'y': ((count, 6) if emotion else (count, 2, 73), np.float32 if emotion else np.float16)}
    arrays = {name: np.lib.format.open_memmap(packed / (name + '.npy'), mode='w+',
                                             dtype=dtype, shape=shape)
              for name, (shape, dtype) in shapes.items()}
    accum = {name: [0, np.zeros(size), np.zeros(size)]
             for name, size in [('mel', 48 if emotion else 80)] + ([] if emotion else [('target', 73)])}
    records, cursor = [], 0
    for row in rows:
        path = out / 'labels' / (row['id'] + '.npz')
        if sha(path) != row['sha256']:
            raise ValueError('Label checksum mismatch')
        with np.load(path, allow_pickle=False) as data:
            n = row['count']
            for key in arrays:
                finite(data[key])
                arrays[key][cursor:cursor+n] = data[key]
            if row['split'] == 'train':
                values = [('mel', data['x'].transpose(0, 2, 1).reshape(-1, 48) if emotion else data['x'])]
                if not emotion:
                    values.append(('target', data['y'].reshape(-1, 73)))
                for key, value in values:
                    v = value.astype(np.float64)
                    accum[key][0] += len(v)
                    accum[key][1] += v.sum(0)
                    accum[key][2] += (v * v).sum(0)
            record = {'id': row['id'], 'offset': cursor, 'count': n, 'split': row['split']}
            if not emotion:
                record['emotion'] = data['emotion'].tolist()
            records.append(record)
            cursor += n
    stats = {}
    for key, (n, total, squared) in accum.items():
        if not n:
            raise ValueError('No training examples; supply more audio or adjust split fractions')
        mean = total / n
        std = np.sqrt(np.maximum(squared/n-mean**2, 0))
        floor = .01 if emotion else (.5 if key == 'mel' else max(float(np.median(std))*.05, 1e-4))
        stats[key + '_mean'] = mean.astype(np.float32)
        stats[key + '_std'] = np.maximum(std, floor).astype(np.float32)
    np.savez(packed / 'normalization.npz', **stats)
    for array in arrays.values():
        array.flush()
    files = {name: sha(packed / name) for name in ('x.npy', 'y.npy', 'normalization.npz')}
    result = {'schema': 1, 'kind': config['kind'], 'identity': identity, 'config': config,
              'files': files, 'records': records, 'count': count,
              'scope': 'closed-corpus fit' if not config['validation_fraction'] and not config['test_fraction']
                       else 'PCM-disjoint splits; not speaker-disjoint'}
    atomic_json(manifest, result)
    return result


@exclusive_output
def prepare(args):
    if args.validation_fraction < 0 or args.test_fraction < 0 or args.validation_fraction + args.test_fraction >= 1:
        raise ValueError('Split fractions must be nonnegative and sum to less than one')
    if sha(args.teacher) != args.teacher_sha256:
        raise ValueError('Teacher SHA256 mismatch')
    if args.kind == 'face' and sha(args.projection) != args.projection_sha256:
        raise ValueError('Projection SHA256 mismatch')
    # The digest must cover all weights, not just an external-data stub.
    import onnx
    import onnxruntime as ort
    graph = onnx.load(args.teacher, load_external_data=False)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
        raise ValueError('Use a self-contained teacher ONNX; external weight files are not accepted')
    del graph
    torch.set_num_threads(args.threads)
    version = subprocess.check_output([args.ffmpeg, '-version'], text=True).splitlines()[0]
    config = {'kind': args.kind, 'teacher_sha256': args.teacher_sha256,
              'projection_sha256': sha(args.projection) if args.kind == 'face' else None,
              'validation_fraction': args.validation_fraction, 'test_fraction': args.test_fraction,
              'provider': args.provider, 'ffmpeg': version, 'code_sha256': code_hashes()}
    config['runtime'] = {'torch': str(torch.__version__), 'numpy': np.__version__, 'onnxruntime': ort.__version__}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text()) != config:
        raise ValueError('Label protocol changed; use a new output directory')
    atomic_json(out / 'config.json', config)
    (out / 'labels').mkdir(exist_ok=True)
    teacher = session(args.teacher, args.provider, args.device_id, args.threads)
    faces = load_projection(args.projection) if args.kind == 'face' else None
    rows, seen = [], set()
    for raw in sources(args.audio_root):
        pcm = decode(raw, args.ffmpeg)
        digest = hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        path = out / 'labels' / (digest + '.npz')
        receipt = path.with_suffix('.json')
        if receipt.exists():
            row = json.loads(receipt.read_text())
            if sha(path) != row['sha256']:
                raise ValueError('Saved label checksum mismatch')
        else:
            data = label_emotion(teacher, pcm) if args.kind == 'emotion' else label_face(teacher, pcm, faces, digest)
            finite(*data.values())
            temp = path.with_suffix('.tmp')
            with temp.open('wb') as f:
                np.savez(f, **data)
            temp.replace(path)
            bucket = int(digest[:16], 16) / 2**64
            split = 'test' if bucket < args.test_fraction else 'validation' if bucket < args.test_fraction + args.validation_fraction else 'train'
            row = {'id': digest, 'sha256': sha(path), 'count': len(data['x']), 'split': split}
            atomic_json(receipt, row)
        rows.append(row)
        print(json.dumps({'recordings': len(rows), 'last_count': row['count']}), flush=True)
    if not rows:
        raise ValueError('No supported audio found')
    result = pack(out, config, sorted(rows, key=lambda r: r['id']))
    print(json.dumps({'packed_count': result['count'], 'recordings': len(rows), 'scope': result['scope']}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['emotion', 'face'], required=True)
    p.add_argument('--audio-root', type=Path, action='append', required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--teacher-sha256', required=True)
    p.add_argument('--projection', type=Path, default=Path('dist/A2FDirect3.bin'))
    p.add_argument('--projection-sha256', default=PROJECTION_SHA256)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--provider', choices=['cpu', 'cuda', 'dml'], default='cpu')
    p.add_argument('--device-id', type=int, default=0)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--validation-fraction', type=float, default=0)
    p.add_argument('--test-fraction', type=float, default=0)
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
