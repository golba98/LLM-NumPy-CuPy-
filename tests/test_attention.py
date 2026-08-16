import pytest
import numpy as np
from src.tensor import Tensor
from src.nn.attention import scaled_dot_product_attention, MultiHeadSelfAttention, get_causal_mask
from src.utils.gradcheck import gradcheck

def test_causal_mask_structure():
    mask = get_causal_mask(4)
    expected = np.array([
        [   0., -1e9, -1e9, -1e9],
        [   0.,    0., -1e9, -1e9],
        [   0.,    0.,    0., -1e9],
        [   0.,    0.,    0.,    0.]
    ])
    np.testing.assert_allclose(mask.data, expected)

def test_scaled_dot_product_attention_shapes_and_causality():
    B, T, D = 2, 4, 8
    q = Tensor(np.random.randn(B, T, D), requires_grad=True)
    k = Tensor(np.random.randn(B, T, D), requires_grad=True)
    v = Tensor(np.random.randn(B, T, D), requires_grad=True)

    out = scaled_dot_product_attention(q, k, v, causal=True)
    assert out.shape == (B, T, D)
    assert np.all(np.isfinite(out.data))

def test_scaled_dot_product_attention_gradcheck():
    q = Tensor(np.random.randn(1, 3, 4), requires_grad=True)
    k = Tensor(np.random.randn(1, 3, 4), requires_grad=True)
    v = Tensor(np.random.randn(1, 3, 4), requires_grad=True)

    assert gradcheck(lambda q_, k_, v_: scaled_dot_product_attention(q_, k_, v_, causal=True), [q, k, v])

def test_multi_head_attention_shapes_and_gradcheck():
    mha = MultiHeadSelfAttention(dim=8, num_heads=2, max_seq_len=64)
    x = Tensor(np.random.randn(2, 4, 8), requires_grad=True)

    out = mha(x, causal=True)
    assert out.shape == (2, 4, 8)
    assert np.all(np.isfinite(out.data))

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert mha.q_proj.weight.grad is not None
    assert mha.out_proj.weight.grad is not None

def test_multi_head_attention_invalid_dim_raises():
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(dim=10, num_heads=3) # 10 not divisible by 3
