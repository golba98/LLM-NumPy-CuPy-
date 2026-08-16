from typing import Union
import numpy as np  # type-only annotations
from src.backend import array_module
from src.tensor import Tensor, unbroadcast
from src.ops.basic import to_tensor

def matmul(a: Union[Tensor, float, np.ndarray], b: Union[Tensor, float, np.ndarray]) -> Tensor:
    """
    Matrix multiplication of two tensors (2D or batched N-D).
    Calculates forward computation using np.matmul and reverse-mode analytical derivatives:
    dA = dC @ B.T
    dB = A.T @ dC
    """
    a_tensor = to_tensor(a)
    b_tensor = to_tensor(b, like=a_tensor)
    xp = array_module(a_tensor.data)
    out_data = xp.matmul(a_tensor.data, b_tensor.data)
    req_grad = a_tensor.requires_grad or b_tensor.requires_grad
    out = Tensor(out_data, requires_grad=req_grad, _parents=(a_tensor, b_tensor), _op="@")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            # Swap last two axes of B for transposition in matrix multiplication
            if b_tensor.ndim == 1:
                # 1D vector case handling
                b_t = b_tensor.data[:, None]
                grad_a_raw = xp.matmul(out.grad[..., None], b_t.T)
            else:
                b_t = xp.swapaxes(b_tensor.data, -1, -2)
                grad_a_raw = xp.matmul(out.grad, b_t)
            grad_a = unbroadcast(grad_a_raw, a_tensor.shape)
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

        if b_tensor.requires_grad:
            if a_tensor.ndim == 1:
                a_t = a_tensor.data[None, :]
                grad_b_raw = xp.matmul(a_t.T, out.grad[None, ...])
            else:
                a_t = xp.swapaxes(a_tensor.data, -1, -2)
                grad_b_raw = xp.matmul(a_t, out.grad)
            grad_b = unbroadcast(grad_b_raw, b_tensor.shape)
            b_tensor.grad = grad_b if b_tensor.grad is None else b_tensor.grad + grad_b

    out._backward = _backward
    return out
