from typing import List, Tuple
import numpy as np
from src.backend import array_module
from src.parameter import Parameter

class Adam:
    """
    Adam (Adaptive Moment Estimation) optimizer with first/second moment tracking & bias correction.
    """
    def __init__(
        self,
        params: List[Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        # First moment vector m_t and second raw moment vector v_t
        self.m = [array_module(p.data).zeros_like(p.data) for p in self.params]
        self.v = [array_module(p.data).zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """Updates parameters using bias-corrected moment estimates."""
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad
            # Update biased 1st moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            
            # Update biased 2nd raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g ** 2)

            # Compute bias-corrected 1st moment estimate: m_hat = m_t / (1 - beta1^t)
            m_hat = self.m[i] / (1.0 - (self.beta1 ** self.t))
            
            # Compute bias-corrected 2nd raw moment estimate: v_hat = v_t / (1 - beta2^t)
            v_hat = self.v[i] / (1.0 - (self.beta2 ** self.t))

            # Update parameters: theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
            p.data -= self.lr * m_hat / (array_module(p.data).sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Clears accumulated gradients across all parameters."""
        for p in self.params:
            p.zero_grad()
