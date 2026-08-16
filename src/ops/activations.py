import numpy as np  # type-only annotations
from src.backend import array_module
from typing import Union
from src.tensor import Tensor
from src.ops.basic import to_tensor

def relu(a: Union[Tensor, float, np.ndarray]) -> Tensor:
    a_tensor = to_tensor(a)
    xp = array_module(a_tensor.data)
    out_data = xp.maximum(0.0, a_tensor.data)
    out = Tensor(out_data, requires_grad=a_tensor.requires_grad, _parents=(a_tensor,), _op="ReLU")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            grad_a = out.grad * (a_tensor.data > 0.0)
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

    out._backward = _backward
    return out

def sigmoid(a: Union[Tensor, float, np.ndarray]) -> Tensor:
    a_tensor = to_tensor(a)
    xp = array_module(a_tensor.data)
    # Clip for numerical stability
    x_clipped = xp.clip(a_tensor.data, -500, 500)
    out_data = 1.0 / (1.0 + xp.exp(-x_clipped))
    out = Tensor(out_data, requires_grad=a_tensor.requires_grad, _parents=(a_tensor,), _op="Sigmoid")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            grad_a = out.grad * out_data * (1.0 - out_data)
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

    out._backward = _backward
    return out

def tanh(a: Union[Tensor, float, np.ndarray]) -> Tensor:
    a_tensor = to_tensor(a)
    xp = array_module(a_tensor.data)
    out_data = xp.tanh(a_tensor.data)
    out = Tensor(out_data, requires_grad=a_tensor.requires_grad, _parents=(a_tensor,), _op="Tanh")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            grad_a = out.grad * (1.0 - out_data ** 2)
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

    out._backward = _backward
    return out

def silu(a: Union[Tensor, float, np.ndarray]) -> Tensor:
    """SiLU (Swish) activation function: x * sigmoid(x)."""
    a_tensor = to_tensor(a)
    xp = array_module(a_tensor.data)
    x = a_tensor.data
    x_clipped = xp.clip(x, -500, 500)
    sig = 1.0 / (1.0 + xp.exp(-x_clipped))
    out_data = x * sig
    out = Tensor(out_data, requires_grad=a_tensor.requires_grad, _parents=(a_tensor,), _op="SiLU")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            # d(x*sig)/dx = sig + x * sig * (1 - sig)
            grad_a = out.grad * (sig + x * sig * (1.0 - sig))
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

    out._backward = _backward
    return out

def gelu(a: Union[Tensor, float, np.ndarray]) -> Tensor:
    """
    GELU (Gaussian Error Linear Unit) activation function (tanh approximation):
    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    a_tensor = to_tensor(a)
    xp = array_module(a_tensor.data)
    x = a_tensor.data
    k = xp.sqrt(2.0 / xp.pi)
    inner = k * (x + 0.044715 * (x ** 3))
    tanh_inner = xp.tanh(inner)
    out_data = 0.5 * x * (1.0 + tanh_inner)
    out = Tensor(out_data, requires_grad=a_tensor.requires_grad, _parents=(a_tensor,), _op="GELU")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            d_inner_dx = k * (1.0 + 3.0 * 0.044715 * (x ** 2))
            dtanh = 1.0 - (tanh_inner ** 2)
            dx = 0.5 * (1.0 + tanh_inner) + 0.5 * x * dtanh * d_inner_dx
            grad_a = out.grad * dx
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

    out._backward = _backward
    return out
