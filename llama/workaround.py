import torch


def matmul_complex(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_real = a[..., 0]
    a_imag = a[..., 1]
    b_real = b[..., 0]
    b_imag = b[..., 1]
    return torch.stack([
        a_real * b_real - a_imag * b_imag, 
        a_real * b_imag + a_imag * b_real], dim=-1)


def triu(x, diagonal=0):
    l = x.shape[-1]
    arange = torch.arange(l, device=x.device)
    mask = arange.expand(l, l)
    arange = arange.unsqueeze(-1)
    if diagonal:
        arange = arange + diagonal
    mask = mask >= arange
    return x.masked_fill(mask == 0, 0)