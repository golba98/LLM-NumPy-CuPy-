from typing import Union, Tuple, List, Optional, Set
from llm_numpy.backend import array_module, current_backend, is_cuda_array, to_cpu, to_device

def unbroadcast(grad, target_shape: Tuple[int, ...]):
    """
    Reduces gradient tensor `grad` to match `target_shape` by summing along
    dimensions that were broadcasted during forward computation.

    Args:
        grad: The incoming gradient array.
        target_shape: The target array shape to reduce `grad` to.

    Returns:
        Gradient array matching `target_shape`.
    """
    if grad is None:
        return None
    if grad.shape == target_shape:
        return grad

    # Calculate difference in dimensions
    ndim_added = grad.ndim - len(target_shape)
    axes_to_sum = list(range(ndim_added))

    # Check dimensions that were broadcast from size 1
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[ndim_added + i] > 1:
            axes_to_sum.append(ndim_added + i)

    if axes_to_sum:
        grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)

    if ndim_added > 0:
        grad = grad.squeeze(axis=tuple(range(ndim_added)))

    return grad.reshape(target_shape)


_GRAD_ENABLED = True

def is_grad_enabled() -> bool:
    return _GRAD_ENABLED

def set_grad_enabled(mode: bool) -> None:
    global _GRAD_ENABLED
    _GRAD_ENABLED = mode

class no_grad:
    """Context manager that temporarily disables gradient recording and autograd graph construction."""
    def __enter__(self):
        self.prev = is_grad_enabled()
        set_grad_enabled(False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_grad_enabled(self.prev)

class Tensor:
    """
    Lightweight N-dimensional Tensor with automatic differentiation engine.
    Supports basic mathematical operations and reverse-mode autograd.
    """
    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _parents: Tuple['Tensor', ...] = (),
        _op: str = ""
    ):
        if isinstance(data, Tensor):
            data = data.data
        if not hasattr(data, "shape") or not hasattr(data, "dtype"):
            data = current_backend().xp.array(data, dtype=current_backend().xp.float64)
        # Keep the existing FP32/FP64 behavior while preserving CuPy arrays.
        if str(data.dtype) not in {"float16", "float32", "float64"}:
            data = data.astype(array_module(data).float64)

        self.data = data
        self.requires_grad: bool = requires_grad and is_grad_enabled()
        self.grad = None

        self._parents: Tuple['Tensor', ...] = _parents if is_grad_enabled() else ()
        self._op: str = _op
        self._backward = lambda: None


    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self) -> str:
        return "cuda" if is_cuda_array(self.data) else "cpu"

    def to(self, device: str, dtype=None) -> "Tensor":
        self.data = to_device(self.data, device)
        if dtype is not None:
            self.data = self.data.astype(dtype)
        if self.grad is not None:
            self.grad = to_device(self.grad, device)
            if dtype is not None:
                self.grad = self.grad.astype(dtype)
        return self

    def cpu(self):
        return to_cpu(self.data)

    def zero_grad(self) -> None:
        """Clear current gradient."""
        self.grad = None

    def backward(self, grad: Optional[Union[np.ndarray, 'Tensor']] = None) -> None:
        """
        Executes reverse-mode automatic differentiation starting from this tensor.
        Computes gradients for all ancestor tensors in the computational graph with requires_grad=True.
        """
        if not self.requires_grad:
            return

        # Handle root gradient initialization
        if grad is None:
            if self.shape == () or self.shape == (1,):
                self.grad = array_module(self.data).ones_like(self.data, dtype=self.data.dtype)
            else:
                self.grad = array_module(self.data).ones_like(self.data, dtype=self.data.dtype)
        else:
            if isinstance(grad, Tensor):
                grad = grad.data
            xp = array_module(self.data)
            self.grad = xp.asarray(grad, dtype=self.data.dtype)

        # Build topological order using Depth First Search (DFS)
        topo: List['Tensor'] = []
        visited: Set['Tensor'] = set()

        def build_topo(v: 'Tensor'):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # Execute backward functions in reverse topological order
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        req_grad_str = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        op_str = f", op='{self._op}'" if self._op else ""
        return f"Tensor({self.data}{req_grad_str}{op_str})"

    # Python dunder method operator bindings will be registered from llm_numpy.ops
    def __add__(self, other):
        from llm_numpy.ops.basic import add
        return add(self, other)

    def __radd__(self, other):
        from llm_numpy.ops.basic import add
        return add(other, self)

    def __sub__(self, other):
        from llm_numpy.ops.basic import sub
        return sub(self, other)

    def __rsub__(self, other):
        from llm_numpy.ops.basic import sub
        return sub(other, self)

    def __mul__(self, other):
        from llm_numpy.ops.basic import mul
        return mul(self, other)

    def __rmul__(self, other):
        from llm_numpy.ops.basic import mul
        return mul(other, self)

    def __truediv__(self, other):
        from llm_numpy.ops.basic import div
        return div(self, other)

    def __rtruediv__(self, other):
        from llm_numpy.ops.basic import div
        return div(other, self)

    def __neg__(self):
        from llm_numpy.ops.basic import neg
        return neg(self)

    def __pow__(self, power: Union[int, float]):
        from llm_numpy.ops.basic import pow_op
        return pow_op(self, power)

    def __matmul__(self, other):
        from llm_numpy.ops.matrix import matmul
        return matmul(self, other)

    def sum(self, axis=None, keepdims=False):
        from llm_numpy.ops.basic import sum_op
        return sum_op(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        from llm_numpy.ops.basic import mean_op
        return mean_op(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape):
        from llm_numpy.ops.basic import reshape_op
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape_op(self, tuple(shape))

    @property
    def T(self):
        from llm_numpy.ops.basic import transpose_op
        return transpose_op(self)

    def transpose(self, *axes):
        from llm_numpy.ops.basic import transpose_op
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        return transpose_op(self, axes=axes)

    def relu(self):
        from llm_numpy.ops.activations import relu
        return relu(self)

    def sigmoid(self):
        from llm_numpy.ops.activations import sigmoid
        return sigmoid(self)

    def tanh(self):
        from llm_numpy.ops.activations import tanh
        return tanh(self)

    def gelu(self):
        from llm_numpy.ops.activations import gelu
        return gelu(self)

    def silu(self):
        from llm_numpy.ops.activations import silu
        return silu(self)

    def exp(self):
        from llm_numpy.ops.basic import exp_op
        return exp_op(self)

    def log(self):
        from llm_numpy.ops.basic import log_op
        return log_op(self)

    def sqrt(self):
        from llm_numpy.ops.basic import sqrt_op
        return sqrt_op(self)

    def __getitem__(self, idx):
        from llm_numpy.ops.basic import getitem_op
        return getitem_op(self, idx)

    @staticmethod
    def concat(tensors, axis=0):
        from llm_numpy.ops.basic import concat_op
        return concat_op(tensors, axis=axis)
