import numpy as np  # type-only annotations
from src.backend import array_module
from typing import Union
from src.tensor import Tensor
from src.ops.basic import to_tensor

def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Numerically stable Softmax calculation.
    """
    x_t = to_tensor(x)
    xp = array_module(x_t.data)
    max_x = xp.max(x_t.data, axis=axis, keepdims=True)
    exp_x = xp.exp(x_t.data - max_x)
    probs = exp_x / xp.sum(exp_x, axis=axis, keepdims=True)
    out = Tensor(probs, requires_grad=x_t.requires_grad, _parents=(x_t,), _op="Softmax")

    def _backward():
        if out.grad is None:
            return
        if x_t.requires_grad:
            # Jacobian of softmax: dS_i/dx_j = S_i (delta_ij - S_j)
            # sum_k (dL/dS_k * S_k * (delta_ki - S_i)) = S_i * (dL/dS_i - sum_k dL/dS_k * S_k)
            sum_grad_p = xp.sum(out.grad * probs, axis=axis, keepdims=True)
            grad_x = probs * (out.grad - sum_grad_p)
            x_t.grad = grad_x if x_t.grad is None else x_t.grad + grad_x

    out._backward = _backward
    return out

def cross_entropy_loss(logits: Tensor, targets: Union[Tensor, np.ndarray, list],
                       sample_weights: Union[np.ndarray, list, None] = None) -> Tensor:
    """
    Computes numerically stable Softmax Cross-Entropy loss.
    
    Args:
        logits: Unnormalized class predictions of shape (N, C) or (C,).
        targets: Target class indices of shape (N,) or target one-hot probabilities of shape (N, C).
    """
    logits_t = to_tensor(logits)
    xp = array_module(logits_t.data)
    z = logits_t.data
    if z.ndim == 1:
        z = z.reshape(1, -1)
        
    N, C = z.shape

    # Target matrix formulation
    if isinstance(targets, Tensor):
        target_data = targets.data
    else:
        target_data = xp.asarray(targets)

    if target_data.ndim == 1 or (target_data.ndim == 2 and target_data.shape[1] == 1):
        target_indices = target_data.astype(int).ravel()
        one_hot_targets = xp.zeros((N, C), dtype=z.dtype)
        one_hot_targets[xp.arange(N), target_indices] = 1.0
    else:
        one_hot_targets = xp.asarray(target_data, dtype=z.dtype)

    if sample_weights is None:
        weights = xp.ones(N, dtype=z.dtype)
    else:
        weights = xp.asarray(sample_weights, dtype=z.dtype).reshape(-1)
        if weights.size != N or bool(xp.any(weights < 0)):
            raise ValueError("sample_weights must be non-negative and align with logits")
        if not bool(xp.any(weights > 0)):
            raise ValueError("sample_weights must select at least one sample")
    weight_total = xp.sum(weights)

    # Numerically stable log-softmax using max subtraction
    shift_z = z - xp.max(z, axis=1, keepdims=True)
    exp_z = xp.exp(shift_z)
    probs = exp_z / xp.sum(exp_z, axis=1, keepdims=True)

    # Compute loss value: -sum(targets * log(probs)) / N
    log_probs = shift_z - xp.log(xp.sum(exp_z, axis=1, keepdims=True))
    loss_val = -xp.sum(one_hot_targets * log_probs * weights[:, None]) / weight_total

    out = Tensor(loss_val, requires_grad=logits_t.requires_grad, _parents=(logits_t,), _op="CrossEntropy")

    def _backward():
        if out.grad is None:
            return
        if logits_t.requires_grad:
            # Weighted fused softmax + cross-entropy derivative.
            grad_z = out.grad * (probs - one_hot_targets) * weights[:, None] / weight_total
            if logits_t.data.ndim == 1:
                grad_z = grad_z.reshape(-1)
            logits_t.grad = grad_z if logits_t.grad is None else logits_t.grad + grad_z

    out._backward = _backward
    return out
