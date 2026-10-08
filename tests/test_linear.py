import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
from llm_numpy.tensor import Tensor
from llm_numpy.nn.linear import Linear
from llm_numpy.utils.gradcheck import gradcheck

def test_linear_initialization_and_forward():
    layer = Linear(10, 5)
    assert layer.weight.shape == (10, 5)
    assert layer.bias.shape == (5,)
    assert layer.weight.requires_grad
    assert layer.bias.requires_grad

    x = Tensor(np.random.randn(32, 10))
    out = layer(x)
    assert out.shape == (32, 5)

def test_linear_gradcheck():
    layer = Linear(4, 3)
    x = Tensor(np.random.randn(2, 4), requires_grad=True)

    def func(inp, w, b):
        return inp @ w + b

    assert gradcheck(func, [x, layer.weight, layer.bias])
