import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
from llm_numpy.tensor import Tensor
from llm_numpy.nn.rope import RotaryEmbedding
from llm_numpy.utils.gradcheck import gradcheck

def test_rope_odd_head_dim_raises():
    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim=7) # Odd head dimension must raise error

def test_rope_rotation_and_norm_preservation():
    head_dim = 8
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=64)
    
    q = Tensor(np.random.randn(2, 2, 4, head_dim), requires_grad=True)
    k = Tensor(np.random.randn(2, 2, 4, head_dim), requires_grad=True)

    q_rot, k_rot = rope(q, k)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

    # Check norm preservation: ||q_rot||^2 == ||q||^2
    orig_q_norm = np.linalg.norm(q.data, axis=-1)
    rot_q_norm = np.linalg.norm(q_rot.data, axis=-1)
    np.testing.assert_allclose(rot_q_norm, orig_q_norm, atol=1e-6)

def test_rope_position_0_is_identity():
    head_dim = 4
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=64)
    
    # At position 0, cos(0)=1, sin(0)=0 so rotation is identity
    q = Tensor(np.random.randn(1, 1, 1, head_dim), requires_grad=True)
    k = Tensor(np.random.randn(1, 1, 1, head_dim), requires_grad=True)

    q_rot, k_rot = rope(q, k)
    np.testing.assert_allclose(q_rot.data, q.data, atol=1e-6)
    np.testing.assert_allclose(k_rot.data, k.data, atol=1e-6)

def test_rope_gradcheck():
    head_dim = 4
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=16)
    q = Tensor(np.random.randn(1, 1, 2, head_dim), requires_grad=True)
    k = Tensor(np.random.randn(1, 1, 2, head_dim), requires_grad=True)

    def func(q_, k_):
        q_r, k_r = rope(q_, k_)
        return (q_r + k_r).sum()

    assert gradcheck(func, [q, k])
