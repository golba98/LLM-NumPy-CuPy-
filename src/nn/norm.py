import numpy as np
from src.nn.module import Module
from src.parameter import Parameter
from src.tensor import Tensor

class LayerNorm(Module):
    """
    Layer Normalization:
    centers feature activations by subtracting the mean and normalizes by variance.
    y = gamma * (x - mu) / sqrt(var + eps) + beta
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = Parameter(np.ones(dim, dtype=np.float64))
        self.beta = Parameter(np.zeros(dim, dtype=np.float64))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        x_hat = (x - mean) / ((var + self.eps).sqrt())
        return x_hat * self.gamma + self.beta

    def __repr__(self) -> str:
        return f"LayerNorm(dim={self.dim}, eps={self.eps})"


class RMSNorm(Module):
    """
    Root Mean Square Normalization (RMSNorm):
    Normalizes feature activations using root-mean-square magnitude without centering.
    y = gamma * x / sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = Parameter(np.ones(dim, dtype=np.float64))

    def forward(self, x: Tensor) -> Tensor:
        mean_sq = (x ** 2).mean(axis=-1, keepdims=True)
        rms = (mean_sq + self.eps).sqrt()
        x_hat = x / rms
        return x_hat * self.gamma

    def __repr__(self) -> str:
        return f"RMSNorm(dim={self.dim}, eps={self.eps})"
