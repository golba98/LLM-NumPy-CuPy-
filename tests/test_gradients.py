import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
from llm_numpy.tensor import Tensor
from llm_numpy.utils.gradcheck import gradcheck

def test_gradcheck_addition():
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert gradcheck(lambda x, y: x + y, [a, b])

def test_gradcheck_broadcast_addition():
    a = Tensor(np.random.randn(5, 4), requires_grad=True)
    b = Tensor(np.random.randn(4,), requires_grad=True)
    assert gradcheck(lambda x, y: x + y, [a, b])

def test_gradcheck_multiplication():
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert gradcheck(lambda x, y: x * y, [a, b])

def test_gradcheck_division():
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.uniform(0.5, 2.0, size=(3, 4)), requires_grad=True)
    assert gradcheck(lambda x, y: x / y, [a, b])

def test_gradcheck_matmul_2d():
    A = Tensor(np.random.randn(4, 5), requires_grad=True)
    B = Tensor(np.random.randn(5, 3), requires_grad=True)
    assert gradcheck(lambda x, y: x @ y, [A, B])

def test_gradcheck_matmul_batched():
    A = Tensor(np.random.randn(2, 4, 5), requires_grad=True)
    B = Tensor(np.random.randn(2, 5, 3), requires_grad=True)
    assert gradcheck(lambda x, y: x @ y, [A, B])

def test_gradcheck_relu():
    x = Tensor(np.array([-2.0, -0.5, 0.5, 2.0]), requires_grad=True)
    assert gradcheck(lambda t: t.relu(), [x])

def test_gradcheck_sigmoid():
    x = Tensor(np.random.randn(4, 4), requires_grad=True)
    assert gradcheck(lambda t: t.sigmoid(), [x])

def test_gradcheck_tanh():
    x = Tensor(np.random.randn(4, 4), requires_grad=True)
    assert gradcheck(lambda t: t.tanh(), [x])

def test_gradcheck_silu():
    x = Tensor(np.random.randn(4, 4), requires_grad=True)
    assert gradcheck(lambda t: t.silu(), [x])

def test_gradcheck_gelu():
    x = Tensor(np.random.randn(4, 4), requires_grad=True)
    assert gradcheck(lambda t: t.gelu(), [x])

def test_gradcheck_composed_expression():
    # y = (A @ B + bias).relu().sum()
    A = Tensor(np.random.randn(3, 4), requires_grad=True)
    B = Tensor(np.random.randn(4, 2), requires_grad=True)
    bias = Tensor(np.random.randn(2,), requires_grad=True)
    
    def comp(a, b, c):
        return (a @ b + c).relu().sum()

    assert gradcheck(comp, [A, B, bias])
