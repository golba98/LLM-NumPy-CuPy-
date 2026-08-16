import numpy as np

from src.nn.attention import MultiHeadSelfAttention
from src.nn.ffn import SwiGLU
from src.tensor import Tensor
from src.utils.seed import set_seed


def _copy_parameters(source, target):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data[...] = source_param.data


def test_fused_qkv_matches_three_projection_path_forward_and_backward():
    set_seed(7)
    fused = MultiHeadSelfAttention(16, 4, max_seq_len=8, fused_qkv=True)
    unfused = MultiHeadSelfAttention(16, 4, max_seq_len=8, fused_qkv=False)
    _copy_parameters(fused, unfused)
    x_fused = Tensor(np.random.default_rng(2).normal(size=(2, 8, 16)).astype(np.float32), requires_grad=True)
    x_unfused = Tensor(x_fused.data.copy(), requires_grad=True)

    fused_out = fused(x_fused)
    unfused_out = unfused(x_unfused)
    np.testing.assert_allclose(fused_out.data, unfused_out.data, rtol=2e-5, atol=2e-5)
    fused_out.sum().backward()
    unfused_out.sum().backward()
    np.testing.assert_allclose(x_fused.grad, x_unfused.grad, rtol=3e-5, atol=3e-5)
    for fused_param, unfused_param in zip(fused.parameters(), unfused.parameters()):
        np.testing.assert_allclose(fused_param.grad, unfused_param.grad, rtol=3e-5, atol=3e-5)


def test_fused_swiglu_matches_two_projection_path_forward_and_backward():
    set_seed(11)
    fused = SwiGLU(16, 32, fused_input=True)
    unfused = SwiGLU(16, 32, fused_input=False)
    _copy_parameters(fused, unfused)
    values = np.random.default_rng(3).normal(size=(2, 8, 16)).astype(np.float32)
    x_fused = Tensor(values.copy(), requires_grad=True)
    x_unfused = Tensor(values.copy(), requires_grad=True)

    fused_out = fused(x_fused)
    unfused_out = unfused(x_unfused)
    np.testing.assert_allclose(fused_out.data, unfused_out.data, rtol=2e-5, atol=2e-5)
    fused_out.sum().backward()
    unfused_out.sum().backward()
    np.testing.assert_allclose(x_fused.grad, x_unfused.grad, rtol=3e-5, atol=3e-5)
    for fused_param, unfused_param in zip(fused.parameters(), unfused.parameters()):
        np.testing.assert_allclose(fused_param.grad, unfused_param.grad, rtol=3e-5, atol=3e-5)
