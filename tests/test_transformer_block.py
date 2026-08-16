import pytest
import numpy as np
from src.tensor import Tensor
from src.nn.transformer import TransformerBlock
from src.nn.module import count_parameters
from src.utils.gradcheck import gradcheck

def test_transformer_block_forward_backward_and_params():
    block = TransformerBlock(dim=8, num_heads=2, hidden_dim=16, max_seq_len=64)
    x = Tensor(np.random.randn(2, 4, 8), requires_grad=True)

    y = block(x, causal=True)
    assert y.shape == (2, 4, 8)
    assert np.all(np.isfinite(y.data))

    loss = y.sum()
    loss.backward()

    # Input gradient must exist and be finite
    assert x.grad is not None
    assert np.all(np.isfinite(x.grad))

    # All trainable parameters must receive finite gradients
    params = block.parameters()
    assert len(params) > 0
    for p in params:
        assert p.grad is not None
        assert np.all(np.isfinite(p.grad))

def test_transformer_block_causal_prefix_invariance():
    # End-to-end causal test: Changing future tokens in sequence must NOT affect earlier tokens
    block = TransformerBlock(dim=8, num_heads=2, hidden_dim=16, max_seq_len=64)
    
    seq_A = np.random.randn(1, 6, 8)
    seq_B = seq_A.copy()
    seq_B[0, 4:, :] = np.random.randn(1, 2, 8) # Mutate tokens at t=4 and t=5

    x_A = Tensor(seq_A, requires_grad=False)
    x_B = Tensor(seq_B, requires_grad=False)

    out_A = block(x_A, causal=True)
    out_B = block(x_B, causal=True)

    # Tokens t=0..3 must be identical
    np.testing.assert_allclose(out_A.data[0, :4, :], out_B.data[0, :4, :], atol=1e-12)

def test_tiny_transformer_block_sampled_gradcheck():
    # Sampled finite-difference check on tiny TransformerBlock configuration
    block = TransformerBlock(dim=4, num_heads=2, hidden_dim=8, max_seq_len=16)
    x = Tensor(np.random.randn(1, 2, 4), requires_grad=True)

    def func(inp):
        return block(inp, causal=True).sum()

    assert gradcheck(func, [x], eps=1e-5, atol=1e-3, rtol=1e-2)
