import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
from llm_numpy.tensor import Tensor
from llm_numpy.nn.ffn import SwiGLU
from llm_numpy.utils.gradcheck import gradcheck

def test_swiglu_forward_and_params():
    swiglu = SwiGLU(dim=8, hidden_dim=24)
    x = Tensor(np.random.randn(2, 4, 8), requires_grad=True)

    out = swiglu(x)
    assert out.shape == (2, 4, 8)
    assert np.all(np.isfinite(out.data))

def test_swiglu_gradcheck():
    swiglu = SwiGLU(dim=4, hidden_dim=8)
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)

    def func(inp, w_g, w_v, w_d):
        gate = (inp @ w_g).silu()
        val = inp @ w_v
        return ((gate * val) @ w_d).sum()

    assert gradcheck(func, [x, swiglu.w_gate.weight, swiglu.w_value.weight, swiglu.w_down.weight])
