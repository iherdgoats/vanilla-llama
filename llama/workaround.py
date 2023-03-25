import torch


def triu(x, diagonal=0):
    l = x.shape[-1]
    arange = torch.arange(l, device=x.device)
    mask = arange.expand(l, l)
    arange = arange.unsqueeze(-1)
    if diagonal:
        arange = arange + diagonal
    mask = mask >= arange
    return x.masked_fill(mask == 0, 0)