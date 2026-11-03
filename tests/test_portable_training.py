"""CPU integration checks: preparation, resume equivalence, export and tampering.

Run with the training environment: python -m unittest discover -s tests
-p test_portable_training.py. Synthetic teachers test plumbing, not model quality.
"""
import argparse
import json
from pathlib import Path
import tempfile
import struct
import unittest
import wave
import zlib

import numpy as np
import onnx
from onnx import helper as H, TensorProto as T
import torch
import lz4.frame

from tools.training import prepare, train, export
from tools.training.common import load_checkpoint, sha


def archive(path, payload, version):
    folder, name = b'sound\\voice\0', b'line.fuz\0'
    entry = struct.pack('<QIIQ', 0, 1, 0, 0) if version == 105 else struct.pack('<QII', 0, 1, 0)
    encoded = struct.pack('<I', len(payload)) + (lz4.frame.compress(payload) if version == 105 else zlib.compress(payload))
    offset = 36 + len(entry) + 1 + len(folder) + 16 + len(name)
    header = struct.pack('<4s8I', b'BSA\0', version, 36, 7, 1, 1, len(folder), len(name), 0)
    path.write_bytes(header + entry + bytes([len(folder)]) + folder +
                     struct.pack('<QII', 0, len(encoded), offset) + name + encoded)


def tensor(name, shape, dtype=T.FLOAT):
    return H.make_tensor_value_info(name, dtype, shape)


def teacher(path, kind):
    if kind == 'emotion':
        inputs = [tensor('audio', [1, 60000])]
        outputs = [tensor('logits', [1, 6])]
        nodes = [H.make_node('ReduceMean', ['audio'], ['mean'], axes=[1], keepdims=1),
                 H.make_node('MatMul', ['mean', 'weight'], ['scaled']),
                 H.make_node('Add', ['scaled', 'bias'], ['logits'])]
        weights = [H.make_tensor('weight', T.FLOAT, [1, 6], [1, 2, 3, 4, 5, 6]),
                   H.make_tensor('bias', T.FLOAT, [1, 6], [-1, -.5, 0, .5, 1, 1.5])]
    else:
        inputs = [tensor('window', [1, 16000]), tensor('identity', [1, 3]),
                  tensor('emotion', [1, 30, 10]), tensor('input_latents', [2, 2, 1, 256])]
        outputs = [tensor(n, [1, 30, size]) for n, size in [('skin', 512), ('tongue', 512), ('jaw', 19)]]
        outputs += [tensor('output_latents', [2, 2, 1, 256])]
        weights, nodes = [], []
        for name, size in [('skin', 512), ('tongue', 512), ('jaw', 19)]:
            value = H.make_tensor(name+'_value', T.FLOAT, [1, 30, size], np.zeros(30*size, np.float32))
            nodes.append(H.make_node('Constant', [], [name], value=value))
        nodes.append(H.make_node('Identity', ['input_latents'], ['output_latents']))
    model = H.make_model(H.make_graph(nodes, 'synthetic training contract', inputs, outputs, weights),
                        opset_imports=[H.make_opsetid('', 17)])
    model.ir_version = 9
    onnx.save(model, path)


class PortableTraining(unittest.TestCase):
    def test_complete_workflows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root/'audio'
            audio.mkdir()
            for i, length in enumerate((1280, 3000, 50000)):
                pcm = (.2*np.sin(np.arange(length)*(i+1)*.08)*32767).astype('<i2')
                with wave.open(str(audio/f'{i}.wav'), 'wb') as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(16000)
                    f.writeframes(pcm.tobytes())
            fuz = b'FUZE' + struct.pack('<II', 1, 3) + b'lip' + (audio/'0.wav').read_bytes()
            (audio/'duplicate.fuz').write_bytes(fuz)
            for version in (104, 105):
                archive(audio/f'voices{version}.bsa', fuz, version)
            for kind in ('emotion', 'face'):
                with self.subTest(kind=kind):
                    model_path = root/(kind+'.onnx')
                    teacher(model_path, kind)
                    out = root/(kind+'-labels')
                    args = argparse.Namespace(kind=kind, audio_root=[audio], teacher=model_path,
                        teacher_sha256=sha(model_path), projection=Path('dist/A2FDirect3.bin'),
                        projection_sha256=prepare.PROJECTION_SHA256, out=out,
                        provider='cpu', device_id=0, threads=1, ffmpeg='ffmpeg',
                        validation_fraction=0., test_fraction=0.)
                    prepare.prepare(args)
                    self.assertEqual(len(json.loads((out/'packed/dataset.json').read_text())['records']), 3)
                    manifest_hash = sha(out/'packed/dataset.json')
                    prepare.prepare(args)
                    self.assertEqual(manifest_hash, sha(out/'packed/dataset.json'))
                    original = args.teacher_sha256
                    args.teacher_sha256 = '0'*64
                    with self.assertRaisesRegex(ValueError, 'Teacher SHA256'):
                        prepare.prepare(args)
                    args.teacher_sha256 = original
                    base = dict(data=out/'packed', device='cpu', epochs=2, batch=2, width=8,
                        lr=.001, threads=1, seed=123, checkpoint_steps=1, evaluate_every=1,
                        selection_split='train', max_updates=None, resume=False)
                    continuous = root/(kind+'-continuous')
                    train.train(argparse.Namespace(out=continuous, **base))
                    resumed = root/(kind+'-resumed')
                    partial = dict(base, max_updates=1)
                    train.train(argparse.Namespace(out=resumed, **partial))
                    train.train(argparse.Namespace(out=resumed, **dict(base, resume=True)))
                    expected, _ = load_checkpoint(continuous/'latest.json')
                    actual, _ = load_checkpoint(resumed/'latest.json')
                    self.assertEqual(actual['steps'], expected['steps'])
                    self.assertEqual(actual['history'], expected['history'])
                    for name in expected['model']:
                        self.assertTrue(torch.equal(actual['model'][name], expected['model'][name]), name)
                    with self.assertRaisesRegex(ValueError, 'unchanged'):
                        train.train(argparse.Namespace(out=resumed, **dict(base, resume=True, batch=3)))
                    artifact = root/(kind+'-export')
                    export.export(argparse.Namespace(checkpoint=resumed/'best.json', out=artifact))
                    exported = artifact/('emotion_performance.onnx' if kind == 'emotion' else 'performance.onnx')
                    self.assertTrue(exported.is_file())
                    if kind == 'face':
                        self.assertIn('kTargetStd', (artifact/'performance_stats.h').read_text())
                    # A modified packed label must be refused before training.
                    with (out/'packed/y.npy').open('r+b') as f:
                        f.seek(-1, 2)
                        byte = f.read(1)
                        f.seek(-1, 2)
                        f.write(bytes([byte[0] ^ 1]))
                    with self.assertRaisesRegex(ValueError, 'checksum'):
                        train.Dataset(out/'packed', 'train')
                    pointer = json.loads((resumed/'latest.json').read_text())
                    (resumed/pointer['checkpoint']).write_bytes(b'corrupt')
                    with self.assertRaisesRegex(ValueError, 'checksum'):
                        load_checkpoint(resumed/'latest.json')

    def test_validation_does_not_set_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out/'labels').mkdir()
            rows = []
            for name, value, split in [('train', 1, 'train'), ('validation', 100, 'validation')]:
                path = out/'labels'/(name+'.npz')
                np.savez(path, x=np.full((1, 48, 188), value, np.float16), y=np.zeros((1, 6), np.float32))
                rows.append({'id': name, 'sha256': sha(path), 'count': 1, 'split': split})
            prepare.pack(out, {'kind': 'emotion', 'validation_fraction': .1, 'test_fraction': 0}, rows)
            with np.load(out/'packed/normalization.npz') as stats:
                np.testing.assert_array_equal(stats['mel_mean'], np.ones(48))

    def test_writer_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'writer.lock').touch()
            with self.assertRaisesRegex(RuntimeError, 'locked'):
                train.train(argparse.Namespace(out=root))

    def test_face_boundary_mask(self):
        # Verify context masking independently from the trainer's loss formula.
        pred = torch.zeros(1, 124, 73)
        target = torch.ones_like(pred)
        mask = torch.zeros(1, 124)
        mask[:, 17:20] = 1
        p, n, v, pairs = train.face_parts(pred, target, mask)
        self.assertEqual((p.item(), n.item(), v.item(), pairs.item()), (219, 3, 0, 2))
        pred[:, :17] = 1000
        self.assertEqual(train.face_parts(pred, target, mask)[0].item(), 219)


if __name__ == '__main__':
    unittest.main()
