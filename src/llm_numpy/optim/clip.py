import numpy as np
from llm_numpy.backend import array_module
from typing import Iterable
from llm_numpy.parameter import Parameter

def clip_grad_norm_(parameters: Iterable[Parameter], max_norm: float) -> float:
    """
    Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together as if they were concatenated into a single vector.

    Args:
        parameters: An iterable of Parameters that have gradients.
        max_norm: Maximum norm threshold.

    Returns:
        Total L2 norm of the parameters (before clipping).
    """
    params = [p for p in parameters if p.requires_grad and p.grad is not None]
    if len(params) == 0:
        return 0.0

    # Calculate global L2 norm
    # FP16 squared gradients can overflow during the global reduction even
    # when individual gradients are finite. Accumulate the norm in FP32.
    total_norm_sq = sum(
        array_module(p.grad).sum(p.grad.astype(np.float32) ** 2, dtype=np.float32)
        for p in params
    )
    total_norm = float(total_norm_sq.item() if hasattr(total_norm_sq, "item") else total_norm_sq) ** 0.5

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in params:
            p.grad *= clip_coef

    return total_norm
