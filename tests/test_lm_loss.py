import pytest
import numpy as np
from src.tensor import Tensor
from src.losses.cross_entropy import cross_entropy_loss
from src.nn.model import language_model_loss, perplexity
from src.utils.gradcheck import gradcheck

def test_hand_calculated_cross_entropy_reference():
    # Logits: [2.0, 1.0, 0.0], target: 0
    logits = Tensor([2.0, 1.0, 0.0], requires_grad=True)
    target = np.array([0])

    loss = cross_entropy_loss(logits, target)

    # Hand calculation:
    # exp([0, -1, -2]) = [1, 0.36787944117, 0.13533528323]
    # sum_exp = 1.5032147244
    # p0 = 1 / 1.5032147244 = 0.66524095577
    # loss = -ln(p0) = 0.40760596444
    expected_loss = 0.40760596444
    np.testing.assert_allclose(loss.data.item(), expected_loss, atol=1e-6)

    # Hand calculation of derivative dL/dz = (p - y_one_hot) / N
    # p = [0.66524095577, 0.244728471, 0.090030573]
    # target one-hot = [1.0, 0.0, 0.0]
    # grad = p - y_one_hot = [-0.33475904423, 0.244728471, 0.090030573]
    loss.backward()
    expected_grad = np.array([-0.33475904423, 0.244728471, 0.090030573])
    np.testing.assert_allclose(logits.grad, expected_grad, atol=1e-6)

def test_perplexity_computation():
    loss_val = np.log(10.0)
    ppl = perplexity(loss_val)
    np.testing.assert_allclose(ppl, 10.0, atol=1e-6)

def test_lm_loss_gradcheck():
    logits = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    targets = np.array([[0, 2, 1], [3, 1, 0]])

    assert gradcheck(lambda l: language_model_loss(l, targets), [logits])
