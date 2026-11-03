"""Skyrim Audio2Face-only emotion distillation: fixed frontend and compact students.

Input is the same 60,000-sample, 16 kHz window as Audio2Emotion v2.2.
Exported students consume frontend output [batch,48,188], and return six logits
in anger/disgust/fear/joy/neutral/sadness order. Include frontend cost in timings.
NVIDIA's Audio2Emotion restrictions apply; see dist/licenses/.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WINDOW = 60000
MELS = 48
FRAMES = 188


def window_starts(samples):
    if samples < 1:
        raise ValueError('empty audio')
    frames = (samples * 60 + 15999) // 16000
    count = (frames - 1) // 30 + 2
    last = max(0, samples - WINDOW)
    return sorted({min(max(i * 8000 - WINDOW // 2, 0), last) for i in range(count)})


class Frontend(nn.Module):
    """48 HTK log-mels, 32ms FFT, 25ms Hann, 20ms stride; per-window RMS norm."""
    def __init__(self):
        super().__init__()
        hz = np.linspace(0, 8000, 257)
        edges = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 8000 / 700), MELS+2)/2595)-1)
        bank = np.maximum(0, np.minimum(
            (hz[None]-edges[:-2, None])/(edges[1:-1]-edges[:-2])[:, None],
            (edges[2:, None]-hz[None])/(edges[2:]-edges[1:-1])[:, None]))
        self.register_buffer('bank', torch.tensor(bank, dtype=torch.float32))
        self.register_buffer('window', torch.hann_window(400))

    def forward(self, pcm):
        x = pcm - pcm.mean(-1, keepdim=True)
        x = x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-7)
        spec = torch.stft(x, 512, hop_length=320, win_length=400,
                          window=self.window, center=True, pad_mode='constant',
                          return_complex=True).abs().square()
        return torch.log(torch.matmul(self.bank, spec) + 1e-6)


class Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.depthwise = nn.Conv1d(width, width, 5, padding=2*dilation,
                                   dilation=dilation, groups=width)
        self.expand = nn.Conv1d(width, width*2, 1)
        self.project = nn.Conv1d(width*2, width, 1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x):
        return x + self.project(F.silu(self.expand(self.norm(self.depthwise(x)))))


class Student(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.register_buffer('mean', torch.zeros(1, MELS, 1))
        self.register_buffer('std', torch.ones(1, MELS, 1))
        self.stem = nn.Conv1d(MELS, width, 5, stride=2, padding=2)
        self.blocks = nn.Sequential(*(Block(width, d) for d in (1, 2, 4, 8)))
        self.head = nn.Sequential(nn.Linear(width*4, width), nn.SiLU(), nn.Linear(width, 6))

    def forward(self, mel):
        x = F.silu(self.stem((mel-self.mean)/self.std))
        x = F.avg_pool1d(x, 2)
        x = self.blocks(x)
        # Preserve coarse time order and dispersion, without a large dense head.
        halves = x.split((24, 23), dim=-1)
        stats = torch.cat([t.mean(-1) for t in halves] +
                          [torch.sqrt(t.var(-1, unbiased=False)+1e-5) for t in halves], dim=1)
        return self.head(stats)


class SummaryStudent(nn.Module):
    """No temporal convolutions: four time bins of spectral mean/dispersion + MLP."""
    def __init__(self, width=192):
        super().__init__()
        self.width=width
        self.register_buffer('mean',torch.zeros(1,MELS,1))
        self.register_buffer('std',torch.ones(1,MELS,1))
        self.head=nn.Sequential(nn.Linear(MELS*8,width),nn.SiLU(),
                                nn.Linear(width,width//2),nn.SiLU(),nn.Linear(width//2,6))

    def forward(self,mel):
        x=(mel-self.mean)/self.std
        mean=F.avg_pool1d(x,47,47)
        variance=(F.avg_pool1d(x.square(),47,47)-mean.square()).clamp_min(0)
        features=torch.cat((mean,torch.sqrt(variance+1e-5)),dim=1).flatten(1)
        return self.head(features)


def make_student(width,architecture='temporal'):
    if architecture not in ('temporal','summary'):raise ValueError('unknown architecture')
    model=(Student if architecture=='temporal' else SummaryStudent)(width)
    model.architecture=architecture
    return model


def distillation_loss(student, teacher):
    # Match distribution for facial controls and centered logits for softer ranks.
    kl = F.kl_div(F.log_softmax(student, -1), F.softmax(teacher, -1), reduction='batchmean')
    centered = F.mse_loss(student-student.mean(-1, keepdim=True),
                          teacher-teacher.mean(-1, keepdim=True))
    return kl + 0.05*centered


def metrics(prediction, teacher):
    p = torch.as_tensor(prediction, dtype=torch.float64).softmax(-1)
    q = torch.as_tensor(teacher, dtype=torch.float64).softmax(-1)
    chosen = q.argmax(-1)
    agrees = p.argmax(-1) == chosen
    per_class = {str(c): {'n': int((chosen == c).sum()),
                          'agreement': float(agrees[chosen == c].double().mean()) if (chosen == c).any() else None}
                 for c in range(6)}
    confidence = q.max(-1).values >= .7
    return {'n': len(p), 'agreement': agrees.double().mean().item(),
            'probability_mae': (p-q).abs().mean().item(),
            'probability_rmse': (p-q).square().mean().sqrt().item(),
            'kl': (q*(q.clamp_min(1e-30).log()-p.clamp_min(1e-30).log())).sum(-1).mean().item(),
            'confident_n': int(confidence.sum()),
            'confident_agreement': float(agrees[confidence].double().mean()) if confidence.any() else None,
            'per_class': per_class}
