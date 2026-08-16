import pytest
import numpy as np
from src.parameter import Parameter
from src.optim.clip import clip_grad_norm_

def test_clip_grad_norm_above_threshold():
    p1 = Parameter([3.0, 4.0]) # norm = 5.0
    p1.grad = np.array([3.0, 4.0])

    p2 = Parameter([0.0, 0.0]) # norm = 0.0
    p2.grad = np.array([0.0, 0.0])

    total_norm = clip_grad_norm_([p1, p2], max_norm=1.0)
    assert abs(total_norm - 5.0) < 1e-6

    # Scaled by 1/5 -> [0.6, 0.8]
    np.testing.assert_allclose(p1.grad, [0.6, 0.8], atol=1e-5)

def test_clip_grad_norm_below_threshold_unchanged():
    p1 = Parameter([0.3, 0.4]) # norm = 0.5 < max_norm=1.0
    p1.grad = np.array([0.3, 0.4])

    total_norm = clip_grad_norm_([p1], max_norm=1.0)
    assert abs(total_norm - 0.5) < 1e-6
    np.testing.assert_allclose(p1.grad, [0.3, 0.4], atol=1e-6)
