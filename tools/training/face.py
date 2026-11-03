"""Small 30 Hz facial-motion student and versioned log-mel frontend."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SR = 16000
FPS = 30
MELS = 80
DILATIONS = (1, 2, 4, 8, 1, 1)
CONTEXT = sum(DILATIONS)


def mel_features(pcm):
    """Centered 32ms STFT / 10ms hop, HTK triangular mel, natural log.

    Interpolate onto exact 30 Hz frame timestamps; zero-pad waveform boundaries.
    Uses only fixed transforms. Training and deployment must retain this frontend.
    """
    pcm = np.asarray(pcm, dtype=np.float32)
    frames = math.ceil(len(pcm) * FPS / SR)
    x = torch.from_numpy(pcm)
    spec = torch.stft(x, 512, hop_length=160, win_length=400,
                      window=torch.hann_window(400), center=True,
                      pad_mode='constant', return_complex=True).abs().square()
    hz = np.linspace(0, SR / 2, 257)
    edges = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 8000 / 700), MELS + 2) / 2595) - 1)
    bank = np.maximum(0, np.minimum((hz[None] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None],
                                    (edges[2:, None] - hz[None]) / (edges[2:] - edges[1:-1])[:, None]))
    mel = torch.log(torch.from_numpy(bank.astype(np.float32)) @ spec + 1e-6).numpy().T
    pos = np.arange(frames, dtype=np.float64) * 100 / FPS
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(mel) - 1)
    frac = (pos - lo).astype(np.float32)[:, None]
    return mel[lo] * (1 - frac) + mel[hi] * frac


def load_projection(path):
    """Read the exact packaged v9 canonical bases and rest subtraction."""
    raw = open(path, 'rb').read()
    offset = 0
    def take(dtype, count):
        nonlocal offset
        a = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).copy()
        offset += np.dtype(dtype).itemsize * count
        return a
    assert raw[:4] == b'A2FD'
    offset = 4
    assert list(take('<u4', 5)) == [9, 2, 512, 512, 1043]
    faces = []
    for _ in range(2):
        nv = int(take('<u4', 1)[0])
        take('<f4', nv * 3); take('<f2', nv); take('<f4', 24)
        k, nm = map(int, take('<u4', 2))
        assert k == 48
        take('<u4', nm); take('<f2', nm * k * 3)
        proj = take('<f4', k * 512).reshape(k, 512)
        rest = take('<f4', 1043)
        take('<f4', 15); take('<f4', 1)
        nt = int(take('<u4', 1)[0])
        take('<f4', nt * 3); take('<f4', 3); take('<f2', nt * 10 * 3)
        tongue = take('<f4', 10 * 512).reshape(10, 512)
        faces.append((proj, tongue, rest))
    assert offset == len(raw), (offset, len(raw))
    # File order is female / male in tools/mark_export3.py.
    return faces


def project_latents(y, face):
    proj, tongue, rest = face
    return np.concatenate(((y[:, :512] - rest[:512]) @ proj.T,
                           (y[:, 512:1024] - rest[512:1024]) @ tongue.T,
                           y[:, 1024:1039] - rest[1024:1039]), axis=-1)


class ThinBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.temporal=nn.Conv1d(width,width,3,padding=dilation,dilation=dilation,groups=width)
        self.mix=nn.Conv1d(width,width,1)
    def forward(self,x):return x+self.mix(F.silu(self.temporal(x)))


class ThinStudent(nn.Module):
    """Four temporal blocks; no channel expansion or LayerNorm at inference."""
    def __init__(self,width=96):
        super().__init__();self.width=width
        self.identity=nn.Embedding(2,8)
        self.input=nn.Conv1d(98,width,1)
        self.blocks=nn.Sequential(*(ThinBlock(width,d) for d in (1,2,4,10)))
        self.output=nn.Conv1d(width,73,1)
    def forward(self,mel,identity,emotion):
        ident=self.identity(identity)[:,None].expand(-1,mel.shape[1],-1)
        x=torch.cat((mel,ident,emotion),dim=-1).transpose(1,2)
        return self.output(self.blocks(F.silu(self.input(x)))).transpose(1,2)
