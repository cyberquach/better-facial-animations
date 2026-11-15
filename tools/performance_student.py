"""CPU-portable convolution/attention student with bounded persistent KV memory.

ContextStudent(width): 6 attention blocks, 4 heads, FFN = 3 x width, four local
blocks, 13 depth mixes, 73 outputs. The shipped dist/performance.onnx is width
608 (dist/performance-model.json); 256 is the 6.0 model.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 256
LAYERS = 6
HEADS = 4
MEMORY = 256
CONTEXT = 15


def parameters(width):
    return 60 * width * width + 296 * width + 89


class LocalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            width, width, 3, padding=dilation, dilation=dilation, groups=width
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return x + F.silu(
            self.conv(self.norm(x).transpose(1, 2))
        ).transpose(1, 2)


class AttentionBlock(nn.Module):
    def __init__(self, width=WIDTH, heads=HEADS, hidden=768):
        super().__init__()
        self.heads = heads
        self.dim = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)
        self.register_buffer(
            'slopes',
            torch.tensor([2 ** (-2 * (i + 1)) for i in range(heads)]).reshape(
                1, heads, 1, 1
            ),
        )

    def attention(self, x, past_k, past_v, past_frames=None):
        b, t, c = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .reshape(b, t, 3, self.heads, self.dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        k = torch.cat((past_k.to(k.dtype), k), dim=2)
        v = torch.cat((past_v.to(v.dtype), v), dim=2)
        qpos = torch.arange(t, device=x.device) + past_k.shape[2]
        kpos = torch.arange(k.shape[2], device=x.device)
        distance = qpos[:, None] - kpos[None, :]
        allowed = (distance >= 0) & (distance <= MEMORY)
        if past_frames is not None:
            allowed = allowed & (
                kpos[None, :] >= MEMORY - past_frames.reshape(1, 1)
            )
        bias = (
            (-distance.float()[None, None] * self.slopes)
            .masked_fill(~allowed[None, None], float('-inf'))
            .to(q.dtype)
        )
        if torch.onnx.is_in_onnx_export():
            weights = torch.softmax(
                (q @ k.transpose(-1, -2)) / math.sqrt(self.dim) + bias, dim=-1
            )
            y = weights @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, dropout_p=0.
            )
        return (
            self.out(y.transpose(1, 2).reshape(b, t, c)),
            k[:, :, -MEMORY:],
            v[:, :, -MEMORY:],
        )


class DepthMix(nn.Module):
    """One input-dependent depth mixture; pseudo-query per sublayer."""

    def __init__(self, width=WIDTH):
        super().__init__()
        self.query = nn.Parameter(torch.randn(width) * .01)
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, values):
        v = torch.stack(values, 0)
        k = (
            v
            * torch.rsqrt(v.float().square().mean(-1, keepdim=True) + 1e-6).to(
                v.dtype
            )
            * self.scale
        )
        weights = (k * self.query).sum(-1).float().softmax(0).to(v.dtype)
        return (weights[..., None] * v).sum(0)


class ContextStudent(nn.Module):
    def __init__(self, width=WIDTH):
        super().__init__()
        if width % HEADS:
            raise ValueError('Width must divide by the head count')
        self.width = width
        self.identity = nn.Embedding(2, 8)
        self.input = nn.Linear(98, width)
        self.local = nn.Sequential(
            *(LocalBlock(width, d) for d in (1, 2, 4, 8))
        )
        self.blocks = nn.ModuleList(
            AttentionBlock(width, HEADS, 3 * width) for _ in range(LAYERS)
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 73)
        self.mix = nn.ModuleList(DepthMix(width) for _ in range(13))

    def empty_state(self, batch, device='cpu', dtype=torch.float32):
        return torch.empty(
            (LAYERS, batch, HEADS, 0, self.width // HEADS),
            device=device, dtype=dtype
        )

    def forward(self, mel, identity, emotion, past_k, past_v, past_frames=None):
        ident = self.identity(identity)[:, None].expand(-1, mel.shape[1], -1)
        x = self.local(
            F.silu(self.input(torch.cat((mel, ident, emotion), dim=-1)))
        )[:, CONTEXT:-CONTEXT]
        ks = []
        vs = []
        values = [x]
        partial = None
        for i, block in enumerate(self.blocks):
            # Two transformer layers per block, four attention/FFN sublayers.
            if i % 2 == 0:
                if partial is not None:
                    values.append(partial)
                partial = None
            h = self.mix[2 * i](
                values if partial is None else values + [partial]
            )
            a, k, v = block.attention(h, past_k[i], past_v[i], past_frames)
            partial = a if partial is None else partial + a
            h = self.mix[2 * i + 1](values + [partial])
            partial = partial + block.down(F.silu(block.up(block.norm2(h))))
            ks.append(k)
            vs.append(v)
        x = self.mix[12](values + [partial])
        return self.head(self.norm(x)), torch.stack(ks), torch.stack(vs)
