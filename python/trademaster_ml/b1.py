"""
B1 — cross-stock attention model. Each instrument (stocks and ETFs) is a token; attention lets every prediction
use the state of every other instrument at the same bar. Written with plain Linear/softmax ops so it exports
cleanly to ONNX and runs on any ONNX Runtime backend (CPU now).

Inputs:  x    [B, N, F] normalized features, mask [B, N] (1 = valid instrument at this bar)
Output:  out  [B, N, 2] = (mu in bps over the horizon, log-variance of that return in bps^2)
"""
import math

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, d, heads, dropout=0.1):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, N, D = x.shape
        z = self.ln1(x)
        q = self.q(z).view(B, N, self.h, self.dk).transpose(1, 2)
        k = self.k(z).view(B, N, self.h, self.dk).transpose(1, 2)
        v = self.v(z).view(B, N, self.h, self.dk).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dk)                 # [B, H, N, N]
        s = s + (1.0 - mask)[:, None, None, :] * -1e9                    # ignore invalid instruments as keys
        a = torch.softmax(s, dim=-1)
        x = x + self.drop(self.o((a @ v).transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class B1(nn.Module):
    def __init__(self, n_instruments, n_features, d=32, heads=4, layers=2, dropout=0.1, id_embedding=True):
        super().__init__()
        self.inp = nn.Linear(n_features, d)
        self.emb = nn.Parameter(torch.zeros(n_instruments, d)) if id_embedding else None
        self.blocks = nn.ModuleList(Block(d, heads, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def forward(self, x, mask):
        h = self.inp(x)
        if self.emb is not None:
            h = h + self.emb[None]
        for b in self.blocks:
            h = b(h, mask)
        out = self.head(self.ln(h))
        return torch.stack([out[..., 0], out[..., 1].clamp(-8.0, 12.0)], dim=-1)
