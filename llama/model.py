# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from llama.workaround import matmul_complex, triu


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 1024


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-2], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = matmul_complex(xq_, freqs_cis).flatten(3)
    xk_out = matmul_complex(xk_, freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )


    def forward(
        self,
        x: torch.Tensor,
        start_pos: torch.IntTensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        hidden_state: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        cache_k = hidden_state[0].to(xq)
        cache_v = hidden_state[1].to(xq)

        cache_k[:bsz, start_pos[0] : start_pos[0] + seqlen] = xk
        cache_v[:bsz, start_pos[0] : start_pos[0] + seqlen] = xv

        hidden_state = torch.stack([cache_k, cache_v], dim=0)

        keys = cache_k[:bsz, : start_pos[0] + seqlen]
        values = cache_v[:bsz, : start_pos[0] + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output), hidden_state


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: torch.IntTensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        hidden_state: torch.Tensor,
    ):
        attn, hidden_state = self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask, hidden_state
        )
        h = x + attn
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out, hidden_state


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.n_heads = params.n_heads
        self.head_dim = params.dim // params.n_heads
        self.max_batch_size = params.max_batch_size
        self.max_seq_len = params.max_seq_len

        self.tok_embeddings = torch.nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for i in range(params.n_layers):
            self.layers.append(TransformerBlock(params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def init_hidden_state(self):
        return torch.zeros(
            (self.n_layers, 2, self.max_batch_size, self.max_seq_len, self.n_heads, self.head_dim)
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: torch.IntTensor, hidden_state: torch.Tensor):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos[0] : start_pos[0] + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = triu(mask, diagonal=start_pos[0] + 1).type_as(h)

        for index, layer in enumerate(self.layers):
            h = h.to(layer.parameters().__next__().device)
            h, hidden_state[index] = layer(h, start_pos[0], freqs_cis, mask, hidden_state[index])

        h = h.to(self.norm.parameters().__next__().device)
        h = self.norm(h)

        hl = h[:, -1, :]
        hl = hl.to(self.output.parameters().__next__().device)
        output = self.output(hl)
        return output.float(), hidden_state