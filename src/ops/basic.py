import numpy as np  # type-only annotations; operations dispatch via array_module
from typing import Union, Tuple, Optional
from src.backend import array_module
from src.tensor import Tensor, unbroadcast

def to_tensor(x, like: Optional[Tensor] = None) -> Tensor:
    """Helper to convert inputs to Tensor if they are not already."""
    if isinstance(x, Tensor):
        return x
    if like is not None:
        x = array_module(like.data).asarray(x, dtype=like.data.dtype)
    return Tensor(x, requires_grad=False)

def add(a: Union[Tensor, float, np.ndarray], b: Union[Tensor, float, np.ndarray]) -> Tensor:
    a_tensor = to_tensor(a)
    b_tensor = to_tensor(b, like=a_tensor)
    xp = array_module(a_tensor.data)
    out_data = a_tensor.data + b_tensor.data
    req_grad = a_tensor.requires_grad or b_tensor.requires_grad
    out = Tensor(out_data, requires_grad=req_grad, _parents=(a_tensor, b_tensor), _op="+")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            grad_a = unbroadcast(out.grad, a_tensor.shape)
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a
        if b_tensor.requires_grad:
            grad_b = unbroadcast(out.grad, b_tensor.shape)
            b_tensor.grad = grad_b if b_tensor.grad is None else b_tensor.grad + grad_b

    out._backward = _backward
    return out

def sub(a: Union[Tensor, float, np.ndarray], b: Union[Tensor, float, np.ndarray]) -> Tensor:
    a_tensor = to_tensor(a)
    b_tensor = to_tensor(b, like=a_tensor)
    xp = array_module(a_tensor.data)
    out_data = a_tensor.data - b_tensor.data
    req_grad = a_tensor.requires_grad or b_tensor.requires_grad
    out = Tensor(out_data, requires_grad=req_grad, _parents=(a_tensor, b_tensor), _op="-")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            grad_a = unbroadcast(out.grad, a_tensor.shape)
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a
        if b_tensor.requires_grad:
            grad_b = unbroadcast(-out.grad, b_tensor.shape)
            b_tensor.grad = grad_b if b_tensor.grad is None else b_tensor.grad + grad_b

    out._backward = _backward
    return out

def mul(a: Union[Tensor, float, np.ndarray], b: Union[Tensor, float, np.ndarray]) -> Tensor:
    a_tensor = to_tensor(a)
    b_tensor = to_tensor(b, like=a_tensor)
    xp = array_module(a_tensor.data)
    out_data = a_tensor.data * b_tensor.data
    req_grad = a_tensor.requires_grad or b_tensor.requires_grad
    out = Tensor(out_data, requires_grad=req_grad, _parents=(a_tensor, b_tensor), _op="*")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            grad_a = unbroadcast(out.grad * b_tensor.data, a_tensor.shape)
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a
        if b_tensor.requires_grad:
            grad_b = unbroadcast(out.grad * a_tensor.data, b_tensor.shape)
            b_tensor.grad = grad_b if b_tensor.grad is None else b_tensor.grad + grad_b

    out._backward = _backward
    return out

def div(a: Union[Tensor, float, np.ndarray], b: Union[Tensor, float, np.ndarray]) -> Tensor:
    a_tensor = to_tensor(a)
    b_tensor = to_tensor(b, like=a_tensor)
    xp = array_module(a_tensor.data)
    out_data = a_tensor.data / b_tensor.data
    req_grad = a_tensor.requires_grad or b_tensor.requires_grad
    out = Tensor(out_data, requires_grad=req_grad, _parents=(a_tensor, b_tensor), _op="/")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            grad_a = unbroadcast(out.grad / b_tensor.data, a_tensor.shape)
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a
        if b_tensor.requires_grad:
            grad_b = unbroadcast(-out.grad * a_tensor.data / (b_tensor.data ** 2), b_tensor.shape)
            b_tensor.grad = grad_b if b_tensor.grad is None else b_tensor.grad + grad_b

    out._backward = _backward
    return out

def neg(a: Union[Tensor, float, np.ndarray]) -> Tensor:
    a_tensor = to_tensor(a)
    xp = array_module(a_tensor.data)
    out_data = -a_tensor.data
    out = Tensor(out_data, requires_grad=a_tensor.requires_grad, _parents=(a_tensor,), _op="neg")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            grad_a = -out.grad
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

    out._backward = _backward
    return out

def pow_op(a: Union[Tensor, float, np.ndarray], power: Union[int, float]) -> Tensor:
    a_tensor = to_tensor(a)
    xp = array_module(a_tensor.data)
    out_data = a_tensor.data ** power
    out = Tensor(out_data, requires_grad=a_tensor.requires_grad, _parents=(a_tensor,), _op=f"**{power}")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            grad_a = out.grad * (power * (a_tensor.data ** (power - 1)))
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

    out._backward = _backward
    return out

def sum_op(a: Union[Tensor, float, np.ndarray], axis=None, keepdims: bool = False) -> Tensor:
    a_tensor = to_tensor(a)
    xp = array_module(a_tensor.data)
    out_data = a_tensor.data.sum(axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=a_tensor.requires_grad, _parents=(a_tensor,), _op="sum")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            grad = out.grad
            if not keepdims and axis is not None:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                shape = list(a_tensor.shape)
                for ax in axes:
                    ax_idx = ax % a_tensor.ndim
                    shape[ax_idx] = 1
                grad = grad.reshape(shape)
            grad_a = xp.broadcast_to(grad, a_tensor.shape)
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

    out._backward = _backward
    return out

def mean_op(a: Union[Tensor, float, np.ndarray], axis=None, keepdims: bool = False) -> Tensor:
    a_tensor = to_tensor(a)
    xp = array_module(a_tensor.data)
    out_data = a_tensor.data.mean(axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=a_tensor.requires_grad, _parents=(a_tensor,), _op="mean")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            if axis is None:
                n = a_tensor.data.size
            elif isinstance(axis, int):
                n = a_tensor.shape[axis % a_tensor.ndim]
            else:
                n = int(xp.prod([a_tensor.shape[ax % a_tensor.ndim] for ax in axis]))
            
            grad = out.grad / n
            if not keepdims and axis is not None:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                shape = list(a_tensor.shape)
                for ax in axes:
                    ax_idx = ax % a_tensor.ndim
                    shape[ax_idx] = 1
                grad = grad.reshape(shape)
            grad_a = xp.broadcast_to(grad, a_tensor.shape)
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

    out._backward = _backward
    return out

def reshape_op(a: Union[Tensor, float, np.ndarray], shape: Tuple[int, ...]) -> Tensor:
    a_tensor = to_tensor(a)
    out_data = a_tensor.data.reshape(shape)
    out = Tensor(out_data, requires_grad=a_tensor.requires_grad, _parents=(a_tensor,), _op="reshape")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            grad_a = out.grad.reshape(a_tensor.shape)
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

    out._backward = _backward
    return out

def transpose_op(a: Union[Tensor, float, np.ndarray], axes=None) -> Tensor:
    a_tensor = to_tensor(a)
    ndim = a_tensor.ndim
    if axes is None:
        perm = tuple(reversed(range(ndim)))
    else:
        if isinstance(axes, int):
            axes = (axes,)
        axes = tuple(ax % ndim if ax < 0 else ax for ax in axes)
        if len(axes) == 2 and ndim != 2:
            perm = list(range(ndim))
            perm[axes[0]], perm[axes[1]] = perm[axes[1]], perm[axes[0]]
            perm = tuple(perm)
        else:
            perm = axes

    out_data = a_tensor.data.transpose(perm)
    out = Tensor(out_data, requires_grad=a_tensor.requires_grad, _parents=(a_tensor,), _op="transpose")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            # ``perm`` is Python shape metadata, not a device array.  Keeping
            # the inverse permutation on the host avoids asking CuPy to sort a
            # tuple and does not move any tensor data.
            inv_perm = tuple(np.argsort(perm))
            grad_a = out.grad.transpose(inv_perm)
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

    out._backward = _backward
    return out


def exp_op(a: Union[Tensor, float, np.ndarray]) -> Tensor:
    a_tensor = to_tensor(a)
    xp = array_module(a_tensor.data)
    out_data = xp.exp(a_tensor.data)
    out = Tensor(out_data, requires_grad=a_tensor.requires_grad, _parents=(a_tensor,), _op="exp")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            grad_a = out.grad * out_data
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

    out._backward = _backward
    return out

def log_op(a: Union[Tensor, float, np.ndarray]) -> Tensor:
    a_tensor = to_tensor(a)
    xp = array_module(a_tensor.data)
    out_data = xp.log(a_tensor.data)
    out = Tensor(out_data, requires_grad=a_tensor.requires_grad, _parents=(a_tensor,), _op="log")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            grad_a = out.grad / a_tensor.data
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

    out._backward = _backward
    return out

def sqrt_op(a: Union[Tensor, float, np.ndarray]) -> Tensor:
    a_tensor = to_tensor(a)
    xp = array_module(a_tensor.data)
    out_data = xp.sqrt(a_tensor.data)
    out = Tensor(out_data, requires_grad=a_tensor.requires_grad, _parents=(a_tensor,), _op="sqrt")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            grad_a = out.grad / (2.0 * out_data)
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

    out._backward = _backward
    return out

def getitem_op(a: Union[Tensor, float, np.ndarray], idx) -> Tensor:
    a_tensor = to_tensor(a)
    xp = array_module(a_tensor.data)
    out_data = a_tensor.data[idx]
    out = Tensor(out_data, requires_grad=a_tensor.requires_grad, _parents=(a_tensor,), _op="getitem")

    def _backward():
        if out.grad is None:
            return
        if a_tensor.requires_grad:
            grad_a = xp.zeros_like(a_tensor.data)
            xp.add.at(grad_a, idx, out.grad)
            a_tensor.grad = grad_a if a_tensor.grad is None else a_tensor.grad + grad_a

    out._backward = _backward
    return out

def concat_op(tensors: Union[Tuple[Tensor, ...], list], axis: int = 0) -> Tensor:
    tensor_list = [to_tensor(t) for t in tensors]
    xp = array_module(tensor_list[0].data)
    out_data = xp.concatenate([t.data for t in tensor_list], axis=axis)
    req_grad = any(t.requires_grad for t in tensor_list)
    out = Tensor(out_data, requires_grad=req_grad, _parents=tuple(tensor_list), _op="concat")

    def _backward():
        if out.grad is None:
            return
        split_sizes = [t.shape[axis] for t in tensor_list]
        # Split boundaries are shape metadata; keep them as host integers so
        # NumPy and CuPy receive the same indexing representation.
        split_indices = tuple(np.cumsum(split_sizes)[:-1].tolist())
        grads = xp.split(out.grad, split_indices, axis=axis)
        for t, g in zip(tensor_list, grads):
            if t.requires_grad:
                t.grad = g if t.grad is None else t.grad + g

    out._backward = _backward
    return out
