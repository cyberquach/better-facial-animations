"""Train/evaluate the shipped student architectures on local memory-mapped labels."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from tools.emotion_student import distillation_loss, make_student
from tools.training.common import atomic_json, code_hashes, exclusive_output, finite, load_checkpoint, save_checkpoint, sha
from tools.training.face import ThinStudent


class Dataset:
    def __init__(self, root, split):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'dataset.json').read_text())
        self.kind = self.manifest['kind']
        for name, digest in self.manifest['files'].items():
            if sha(self.root / name) != digest:
                raise ValueError('Dataset checksum mismatch: ' + name)
        self.x = np.load(self.root / 'x.npy', mmap_mode='r')
        self.y = np.load(self.root / 'y.npy', mmap_mode='r')
        with np.load(self.root / 'normalization.npz', allow_pickle=False) as stats:
            self.stats = {k: stats[k].copy() for k in stats.files}
        self.records = [r for r in self.manifest['records'] if r['split'] == split]
        counts = [r['count'] if self.kind == 'emotion' else 2 * math.ceil(r['count']/90)
                  for r in self.records]
        self.ends = np.cumsum(counts, dtype=np.int64)
        if not len(self.ends):
            raise ValueError('No records in requested split: ' + split)

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        record = int(np.searchsorted(self.ends, index, side='right'))
        local = index - (int(self.ends[record-1]) if record else 0)
        row = self.records[record]
        offset = row['offset']
        if self.kind == 'emotion':
            return self.x[offset+local].astype(np.float32), self.y[offset+local].astype(np.float32)
        start, identity = (local // 2) * 90, local % 2
        positions = np.arange(start-17, start+90+17)
        valid = (positions >= 0) & (positions < row['count'])
        x = np.full((124, 80), np.log(1e-6), np.float32)
        y = np.zeros((124, 73), np.float32)
        x[valid] = self.x[offset+positions[valid]]
        y[valid] = self.y[offset+positions[valid], identity]
        mask = valid.astype(np.float32)
        mask[:17] = mask[-17:] = 0
        stats = self.stats
        return ((x-stats['mel_mean'])/stats['mel_std'], np.int64(identity),
                np.broadcast_to(np.asarray(row['emotion'], np.float32), (124, 10)).copy(),
                (y-stats['target_mean'])/stats['target_std'], mask)

    def batch(self, ids, device):
        return [torch.from_numpy(np.stack(values)).to(device)
                for values in zip(*(self[int(i)] for i in ids))]


def face_parts(pred, target, mask):
    position_sum = ((pred-target).square()*mask[..., None]).sum()
    velocity_mask = mask[:, 1:]*mask[:, :-1]
    velocity_sum = ((torch.diff(pred, dim=1)-torch.diff(target, dim=1)).square()
                    * velocity_mask[..., None]).sum()
    frames = mask.sum()
    pairs = velocity_mask.sum()
    return position_sum, frames, velocity_sum, pairs


def loss(model, batch, kind):
    if kind == 'emotion':
        return distillation_loss(model(batch[0]), batch[1])
    p, n, v, pairs = face_parts(model(*batch[:3]), batch[3], batch[4])
    return p/(n.clamp_min(1)*73) + .25*v/(pairs.clamp_min(1)*73)


@torch.no_grad()
def evaluate(model, data, device, batch_size):
    model.eval()
    total = np.zeros(9, np.float64)
    classes = np.zeros((6, 2), np.int64)
    for begin in range(0, len(data), batch_size):
        batch = data.batch(range(begin, min(begin+batch_size, len(data))), device)
        if data.kind == 'emotion':
            pred = model(batch[0]).double()
            teacher = batch[1].double()
            p, q = pred.softmax(-1), teacher.softmax(-1)
            chosen, actual = q.argmax(-1), p.argmax(-1)
            correct = chosen == actual
            confident = q.max(-1).values >= .7
            total += [len(p), correct.sum().item(), (p-q).abs().sum().item(),
                      (p-q).square().sum().item(),
                      (q*(q.clamp_min(1e-30).log()-p.clamp_min(1e-30).log())).sum().item(),
                      confident.sum().item(), correct[confident].sum().item(), 0, 0]
            for c in range(6):
                selected = chosen == c
                classes[c] += [selected.sum().item(), correct[selected].sum().item()]
        else:
            pred = model(*batch[:3])
            p, n, v, pairs = face_parts(pred, batch[3], batch[4])
            raw = ((pred-batch[3])*torch.as_tensor(data.stats['target_std'], device=device)).square()
            total[:4] += [p.item(), n.item(), v.item(), pairs.item()]
            for j, sl in enumerate((slice(0, 48), slice(48, 58), slice(58, 73))):
                total[4+j] += (raw[..., sl].mean(-1)*batch[4]).sum().item()
        finite(total)
    if data.kind == 'emotion':
        n = total[0]
        return {'n': int(n), 'agreement': total[1]/n, 'probability_mae': total[2]/(6*n),
                'probability_rmse': math.sqrt(total[3]/(6*n)), 'kl': total[4]/n,
                'confident_n': int(total[5]), 'confident_agreement': total[6]/total[5] if total[5] else None,
                'per_class': {str(c): {'n': int(a), 'agreement': float(b/a) if a else None}
                              for c, (a, b) in enumerate(classes)}}
    return {'frames': int(total[1]), 'normalized_mse': total[0]/(73*total[1]),
            'velocity_mse': total[2]/(73*total[3]) if total[3] else 0,
            **{key: math.sqrt(total[4+j]/total[1])
               for j, key in enumerate(('skin_rmse', 'tongue_rmse', 'jaw_rmse'))}}


def create_model(kind, width):
    return make_student(width, 'temporal') if kind == 'emotion' else ThinStudent(width)


@exclusive_output
def train(args):
    if min(args.epochs, args.batch, args.checkpoint_steps, args.evaluate_every, args.threads) < 1:
        raise ValueError('Epochs, batch, checkpoint/evaluation intervals and threads must be positive')
    if args.max_updates is not None and args.max_updates < 1:
        raise ValueError('max-updates must be positive')
    if not args.device.startswith(('cpu', 'cuda')):
        raise ValueError('Training supports cpu or cuda[:index]; teacher inference also supports DirectML')
    device = torch.device(args.device)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    data = Dataset(args.data, 'train')
    width = args.width or (512 if data.kind == 'emotion' else 120)
    lr = args.lr if args.lr is not None else (.001 if data.kind == 'emotion' else .0003)
    if width < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Width and learning rate must be positive')
    selection = args.selection_split
    scoring = data if selection == 'train' else Dataset(args.data, selection)
    config = {'kind': data.kind, 'width': width, 'epochs': args.epochs, 'batch': args.batch,
              'lr': lr, 'seed': args.seed, 'device': args.device, 'threads': args.threads,
              'evaluate_every': args.evaluate_every, 'selection_split': selection,
              'dataset_sha256': sha(args.data/'dataset.json'), 'code_sha256': code_hashes(),
              'teacher_sha256': data.manifest['config']['teacher_sha256'],
              'projection_sha256': data.manifest['config']['projection_sha256'],
              'data_scope': data.manifest['scope'],
              'torch': str(torch.__version__), 'numpy': np.__version__}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    exists = (out/'run.json').exists()
    if exists != args.resume:
        raise ValueError('Existing run requires --resume; a new run must omit it')
    if exists and json.loads((out/'run.json').read_text()) != config:
        raise ValueError('Resume requires unchanged corpus, code and training protocol')
    atomic_json(out/'run.json', config)
    model = create_model(data.kind, width).to(device)
    if data.kind == 'emotion':
        model.mean.copy_(torch.as_tensor(data.stats['mel_mean'][None, :, None], device=device))
        model.std.copy_(torch.as_tensor(data.stats['mel_std'][None, :, None], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0, foreach=False)
    epoch = cursor = steps = 0
    loss_sum = loss_count = 0
    best_score = float('inf')
    history = []
    if args.resume:
        state, _ = load_checkpoint(out/'latest.json')
        if state['config'] != config:
            raise ValueError('Checkpoint protocol mismatch')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        epoch, cursor, steps = state['epoch'], state['batch_cursor'], state['steps']
        loss_sum, loss_count = state['loss_sum'], state['loss_count']
        best_score, history = state['best_score'], state['history']
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])

    def save(best=False):
        # Save tensors as tensors (no pickle-only NumPy objects in checkpoints).
        def cpu(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu(v) for v in value]
            return value
        state = {'model': cpu(model.state_dict()), 'optimizer': cpu(optimizer.state_dict()),
                 'width': width, 'architecture': 'temporal' if data.kind == 'emotion' else 'thin',
                 'kind': data.kind, 'stats': {k: torch.from_numpy(v) for k, v in data.stats.items()},
                 'epoch': epoch, 'batch_cursor': cursor, 'steps': steps, 'config': config,
                 'loss_sum': loss_sum, 'loss_count': loss_count, 'best_score': best_score,
                 'history': history, 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
        save_checkpoint(out, cpu(state), best)
        atomic_json(out/'history.json', history)

    # An initial durable state makes early interrupts recoverable too.
    if not args.resume:
        save()
    while epoch < args.epochs:
        order = np.random.default_rng(args.seed+epoch).permutation(len(data))
        batches = math.ceil(len(data)/args.batch)
        for group in optimizer.param_groups:
            group['lr'] = lr*(.05+.95*.5*(1+math.cos(math.pi*epoch/args.epochs)))
        model.train()
        while cursor < batches:
            if (out/'STOP').exists() or (args.max_updates is not None and steps >= args.max_updates):
                save()
                atomic_json(out/'status.json', {'phase': 'stopped', 'epoch': epoch, 'steps': steps})
                return
            ids = order[cursor*args.batch:(cursor+1)*args.batch]
            batch = data.batch(ids, device)
            optimizer.zero_grad(set_to_none=True)
            value = loss(model, batch, data.kind)
            if not torch.isfinite(value):
                raise ValueError('Nonfinite training loss; resume from the last durable checkpoint')
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            cursor += 1
            loss_sum += value.item()*len(ids)
            loss_count += len(ids)
            if steps % args.checkpoint_steps == 0:
                save()
            atomic_json(out/'status.json', {'phase': 'training', 'epoch': epoch,
                                           'batch_cursor': cursor, 'steps': steps, 'loss': value.item()})
        epoch += 1
        cursor = 0
        record = {'epoch': epoch, 'steps': steps, 'training_loss': loss_sum/loss_count}
        loss_sum = loss_count = 0
        improved = False
        if epoch % args.evaluate_every == 0 or epoch == args.epochs:
            score = evaluate(model, scoring, device, args.batch)
            record.update(evaluation=score, split=selection)
            metric = score['kl'] if data.kind == 'emotion' else score['normalized_mse']
            improved = metric < best_score
            if improved:
                best_score = metric
        history.append(record)
        save(improved)
        print(json.dumps(record), flush=True)
    atomic_json(out/'status.json', {'phase': 'complete', 'epoch': epoch, 'steps': steps,
                                   'scope': data.manifest['scope']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--width', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261101)
    p.add_argument('--checkpoint-steps', type=int, default=1000)
    p.add_argument('--evaluate-every', type=int, default=5)
    p.add_argument('--selection-split', choices=['train', 'validation'], default='train')
    p.add_argument('--max-updates', type=int, help='Stop at this cumulative update count; resumable')
    p.add_argument('--resume', action='store_true')
    train(p.parse_args())


if __name__ == '__main__':
    main()
