import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
from llm_numpy.tensor import Tensor
from llm_numpy.losses.mse import mse_loss
from llm_numpy.losses.cross_entropy import cross_entropy_loss, softmax
from llm_numpy.utils.gradcheck import gradcheck

def test_mse_loss():
    y_pred = Tensor([2.0, 3.0], requires_grad=True)
    y_true = Tensor([1.0, 1.0])
    loss = mse_loss(y_pred, y_true)
    # loss = ((1)^2 + (2)^2) / 2 = 2.5
    assert abs(loss.data.item() - 2.5) < 1e-6
    loss.backward()
    # dL/dy_pred = 2 * (y_pred - y_true) / N = 2 * [1, 2] / 2 = [1, 2]
    np.testing.assert_allclose(y_pred.grad, [1.0, 2.0])

def test_softmax_gradcheck():
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    assert gradcheck(lambda t: softmax(t), [x])

def test_cross_entropy_gradcheck():
    logits = Tensor(np.random.randn(4, 5), requires_grad=True)
    targets = np.array([0, 2, 1, 4])
    
    assert gradcheck(lambda l: cross_entropy_loss(l, targets), [logits])
