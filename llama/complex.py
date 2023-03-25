import torch
from dataclasses import dataclass


@dataclass
class ComplexTensorPair:
    def to(self, *args, **kwargs):
        return ComplexTensorPair(
            self.real.to(*args, **kwargs),
            self.imag.to(*args, **kwargs)
        )
    
    def __getitem__(self, idx):
        return ComplexTensorPair(
            self.real[idx, ...],
            self.imag[idx, ...]
        )
    
    @property
    def ndim(self):
        return self.real.ndim

    real: torch.Tensor
    imag: torch.Tensor

    @property
    def shape(self):
        return self.real.shape

    def view(self, *args, **kwargs):
        return ComplexTensorPair(
            self.real.view(*args, **kwargs),
            self.imag.view(*args, **kwargs)
        )

    def __mul__(self, other: 'ComplexTensorPair'):
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real
        result = ComplexTensorPair(real, imag)
        return result


def _view_as_complex(tensor: torch.Tensor):
    assert tensor.shape[-1] == 2
    return ComplexTensorPair(
        real=tensor[..., 0],
        imag=tensor[..., 1]
    )


def _view_as_real(complex_tensor: ComplexTensorPair):
    return torch.stack(
        [complex_tensor.real, complex_tensor.imag],
        dim=-1
    )