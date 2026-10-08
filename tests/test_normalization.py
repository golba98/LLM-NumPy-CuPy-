import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
from llm_numpy.tensor import Tensor
from llm_numpy.nn.norm import LayerNorm, RMSNorm
from llm_numpy.utils.gradcheck import gradcheck

def test_layernorm_forward_and_params():
    ln = LayerNorm(8)
    assert ln.gamma.shape == (8,)
    assert ln.beta.shape == (8,)

    x = Tensor(np.random.randn(2, 4, 8), requires_grad=True)
    out = ln(x)
    assert out.shape == (2, 4, 8)

    # Check zero mean and unit variance per feature vector
    means = out.data.mean(axis=-1)
    vars_ = out.data.var(axis=-1)
    np.testing.assert_allclose(means, 0.0, atol=1e-5)
    np.testing.assert_allclose(vars_, 1.0, atol=1e-3)

def test_layernorm_gradcheck():
    ln = LayerNorm(4)
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)

    def func(inp, g, b):
        mean = inp.mean(axis=-1, keepdims=True)
        var = ((inp - mean) ** 2).mean(axis=-1, keepdims=True)
        x_hat = (inp - mean) / ((var + 1e-5).sqrt())
        return (x_hat * g + b).sum()

    assert gradcheck(func, [x, ln.gamma, ln.beta])

def test_rmsnorm_forward_and_params():
    rms_norm = RMSNorm(8)
    assert rms_norm.gamma.shape == (8,)

    x = Tensor(np.random.randn(3, 7, 8), requires_grad=True)
    out = rms_norm(x)
    assert out.shape == (3, 7, 8)

    # Check root mean square magnitude is ~1 before scaling by gamma=1
    mean_sq = np.mean(out.data ** 2, axis=-1)
    np.testing.assert_allclose(mean_sq, 1.0, atol=1e-3)

def test_rmsnorm_gradcheck():
    rms_norm = RMSNorm(4)
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)

    def func(inp, g):
        rms = ((inp ** 2).mean(axis=-1, keepdims=True) + 1e-5).sqrt()
        return ((inp / rms) * g).sum()

    assert gradcheck(func, [x, rms_norm.gamma])
