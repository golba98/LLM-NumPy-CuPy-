import pytest
import numpy as np
from src.tensor import Tensor, unbroadcast
from src.ops.basic import add, sub, mul, div, neg, pow_op, sum_op, mean_op, reshape_op, transpose_op

def test_unbroadcast_scalar():
    grad = np.ones((3, 3), dtype=np.float64)
    reduced = unbroadcast(grad, ())
    assert reduced.shape == ()
    assert reduced.item() == 9.0

def test_unbroadcast_vector_from_2d():
    grad = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    reduced = unbroadcast(grad, (2,))
    assert reduced.shape == (2,)
    np.testing.assert_allclose(reduced, [4.0, 6.0])

def test_unbroadcast_leading_dim():
    grad = np.ones((4, 2, 3), dtype=np.float64)
    reduced = unbroadcast(grad, (2, 3))
    assert reduced.shape == (2, 3)
    np.testing.assert_allclose(reduced, np.full((2, 3), 4.0))

def test_addition_forward_backward():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = a + b
    np.testing.assert_allclose(c.data, [5.0, 7.0, 9.0])
    
    c.backward([1.0, 1.0, 1.0])
    np.testing.assert_allclose(a.grad, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(b.grad, [1.0, 1.0, 1.0])

def test_multiplication_gradient_accumulation():
    x = Tensor(3.0, requires_grad=True)
    y = x * x + x  # dy/dx = 2x + 1 = 7.0
    y.backward()
    np.testing.assert_allclose(x.grad, 7.0)

def test_power_op():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = (x ** 3).sum()
    y.backward()
    np.testing.assert_allclose(x.grad, [12.0, 27.0])

def test_reshape_transpose():
    x = Tensor(np.arange(6, dtype=np.float64).reshape(2, 3), requires_grad=True)
    y = x.T.reshape(6)
    z = y.sum()
    z.backward()
    assert x.grad.shape == (2, 3)
    np.testing.assert_allclose(x.grad, np.ones((2, 3)))
