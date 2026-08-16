from typing import List
import numpy as np
from src.backend import array_module
from src.parameter import Parameter

class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer with optional momentum.
    """
    def __init__(self, params: List[Parameter], lr: float = 0.01, momentum: float = 0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.momentum = momentum
        self.velocities = [array_module(p.data).zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """Updates parameters based on accumulated gradients."""
        for p, v in zip(self.params, self.velocities):
            if p.grad is None:
                continue
            if self.momentum != 0.0:
                v[...] = self.momentum * v + p.grad
                p.data -= self.lr * v
            else:
                p.data -= self.lr * p.grad

    def zero_grad(self) -> None:
        """Clears accumulated gradients across all parameters."""
        for p in self.params:
            p.zero_grad()
