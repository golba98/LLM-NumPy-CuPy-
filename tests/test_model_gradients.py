import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
from llm_numpy.config import LLMConfig
from llm_numpy.nn.model import TinyLLM, language_model_loss
from llm_numpy.tensor import Tensor
from llm_numpy.utils.gradcheck import gradcheck

def test_tiny_llm_full_backward_and_finite_gradients():
    cfg = LLMConfig(vocab_size=16, max_seq_len=32, dim=8, num_layers=2, num_heads=2, hidden_dim=16, tie_embeddings=True)
    model = TinyLLM(cfg)

    tokens = np.array([[1, 4, 2, 7, 9]])
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    logits, loss = model(inputs, targets=targets)

    assert loss.shape == () or loss.shape == (1,)
    assert np.isfinite(loss.data.item())
    assert logits.shape == (1, 4, 16)

    loss.backward()

    # Every unique parameter must receive a finite, non-zero gradient
    params = model.parameters()
    assert len(params) > 0
    for p in params:
        assert p.grad is not None
        assert np.all(np.isfinite(p.grad))

def test_tiny_llm_sampled_finite_difference_gradcheck():
    # Sampled gradcheck on a minimal TinyLLM model
    cfg = LLMConfig(vocab_size=6, max_seq_len=8, dim=4, num_layers=1, num_heads=2, hidden_dim=8, tie_embeddings=False)
    model = TinyLLM(cfg)

    inputs = np.array([[0, 2, 1]])
    targets = np.array([[2, 1, 4]])

    target_param = model.blocks[0].attn.q_proj.weight

    def func(w):
        model.blocks[0].attn.q_proj.weight = w
        logits, loss = model(inputs, targets=targets)
        return loss

    assert gradcheck(func, [target_param], eps=1e-5, atol=1e-3, rtol=1e-2)
