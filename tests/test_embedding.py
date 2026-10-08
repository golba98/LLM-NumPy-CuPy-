import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
from llm_numpy.tensor import Tensor
from llm_numpy.nn.embedding import Embedding
from llm_numpy.utils.gradcheck import gradcheck

def test_embedding_lookup_shape_and_params():
    emb = Embedding(num_embeddings=100, embedding_dim=16)
    assert emb.weight.shape == (100, 16)

    token_ids = np.array([[1, 5, 42], [99, 0, 12]])
    out = emb(token_ids)
    assert out.shape == (2, 3, 16)

def test_embedding_repeated_index_gradient_accumulation():
    # Crucial test: Token '4' appears twice in sequence [4, 7, 4, 2]
    emb = Embedding(num_embeddings=10, embedding_dim=4)
    token_ids = np.array([4, 7, 4, 2])
    
    out = emb(token_ids)
    loss = out.sum()
    loss.backward()

    # Token 4 was selected twice, so its weight gradient must equal 2 * ones = [2, 2, 2, 2]
    np.testing.assert_allclose(emb.weight.grad[4], [2.0, 2.0, 2.0, 2.0])
    # Tokens 7, 2 were selected once -> [1, 1, 1, 1]
    np.testing.assert_allclose(emb.weight.grad[7], [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(emb.weight.grad[2], [1.0, 1.0, 1.0, 1.0])
    # Unselected tokens -> [0, 0, 0, 0]
    np.testing.assert_allclose(emb.weight.grad[0], [0.0, 0.0, 0.0, 0.0])

def test_embedding_gradcheck():
    emb = Embedding(num_embeddings=5, embedding_dim=3)
    indices = np.array([0, 2, 0, 4])

    def func(w):
        return w[indices].sum()

    assert gradcheck(func, [emb.weight])
