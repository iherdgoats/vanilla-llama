# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
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

    def apply_rotary_embedding(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        bsz: int, 
        seqlen: int,
        freqs_cis: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = xq.float().reshape(bsz, seqlen, self.n_local_heads, self.head_dim // 2, 2)
        xk_ = xk.float().reshape(bsz, seqlen, self.n_local_heads, self.head_dim // 2, 2)
        freqs_cis = freqs_cis.view(1, seqlen, 1, self.head_dim // 2, 2)
        xq_out = matmul_complex(xq_, freqs_cis).flatten(3)
        xk_out = matmul_complex(xk_, freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
        hidden_state: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = self.apply_rotary_embedding(xq, xk, bsz, seqlen, freqs_cis)

        prev_k = hidden_state[0].to(xq)
        prev_v = hidden_state[1].to(xq)

        new_k = torch.cat((prev_k[:bsz, :start_pos.long()], xk, prev_k[:bsz, start_pos.long() + seqlen:]), dim=1)
        new_v = torch.cat((prev_v[:bsz, :start_pos.long()], xv, prev_v[:bsz, start_pos.long() + seqlen:]), dim=1)

        hidden_state = torch.stack([new_k, new_v], dim=0)

        keys = new_k[:bsz, : start_pos.long() + seqlen]
        values = new_v[:bsz, : start_pos.long() + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
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
        start_pos: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
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

        self.freqs_cis = nn.Parameter(self.precompute_freqs_cis())

    def precompute_freqs_cis(self, theta: float = 10000.0):
        dim = self.params.dim // self.params.n_heads
        end = self.params.max_seq_len * 2
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: torch.Tensor, mask: torch.Tensor, hidden_state: torch.Tensor):
        seqlen = tokens.shape[1]
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos.long() : start_pos.long() + seqlen]

        new_hidden_state = []
        for index, layer in enumerate(self.layers):
            h = h.to(layer.parameters().__next__().device)
            h, layer_hidden_state = layer(h, start_pos, freqs_cis, mask, hidden_state[index])
            new_hidden_state.append(layer_hidden_state)

        h = h.to(self.norm.parameters().__next__().device)
        h = self.norm(h)

        hl = h[:, -1, :]
        hl = hl.to(self.output.parameters().__next__().device)
        output = self.output(hl)

        return output.float(), torch.stack(new_hidden_state, dim=0)


def initial_hidden_state(args: ModelArgs):
    head_dim = args.dim // args.n_heads
    return torch.zeros(
        (args.n_layers, 2, args.max_batch_size, args.max_seq_len, args.n_heads, head_dim)
    )


def attention_mask(start_pos: int, seqlen: int) -> torch.Tensor:
    if seqlen > 1:
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"))
        return triu(mask, diagonal=start_pos + 1)

    return torch.zeros((1, 1, 1, 1))