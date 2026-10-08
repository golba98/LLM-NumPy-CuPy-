import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
from llm_numpy.tensor import Tensor
from llm_numpy.nn.attention import softmax
from llm_numpy.utils.gradcheck import gradcheck

def test_softmax_properties():
    x = Tensor(np.random.randn(4, 8), requires_grad=True)
    probs = softmax(x, axis=-1)
    
    # Check probabilities sum to 1 along last axis
    sums = probs.data.sum(axis=-1)
    np.testing.assert_allclose(sums, np.ones(4), atol=1e-6)
    
    # Check output range [0, 1]
    assert np.all(probs.data >= 0.0) and np.all(probs.data <= 1.0)

def test_softmax_large_logits_stability():
    # Large positive and negative logits should not produce NaN or Inf
    x = Tensor(np.array([[1000.0, -1000.0, 500.0]]), requires_grad=True)
    probs = softmax(x, axis=-1)
    
    assert np.all(np.isfinite(probs.data))
    np.testing.assert_allclose(probs.data.sum(axis=-1), 1.0)

def test_softmax_gradcheck():
    x = Tensor(np.random.randn(3, 5), requires_grad=True)
    assert gradcheck(lambda t: softmax(t, axis=-1), [x])

def test_softmax_3d_batch_inputs():
    x = Tensor(np.random.randn(2, 4, 6), requires_grad=True)
    probs = softmax(x, axis=-1)
    assert probs.shape == (2, 4, 6)
    sums = probs.data.sum(axis=-1)
    np.testing.assert_allclose(sums, np.ones((2, 4)), atol=1e-6)
