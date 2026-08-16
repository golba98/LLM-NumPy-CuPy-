import pytest
import os
import numpy as np
from src.config import LLMConfig
from src.nn.model import TinyLLM
from src.utils.serialization import save_weights, load_weights

def test_save_load_weights_roundtrip(tmp_path):
    cfg = LLMConfig(vocab_size=16, max_seq_len=32, dim=8, num_layers=1, num_heads=2, hidden_dim=16, tie_embeddings=True)
    model_orig = TinyLLM(cfg)

    save_file = str(tmp_path / "model_weights.npz")
    save_weights(model_orig, save_file)
    assert os.path.exists(save_file)

    model_loaded = TinyLLM(cfg)
    load_weights(model_loaded, save_file)

    # Check identical forward pass output logits
    tokens = np.array([[1, 4, 7]])
    orig_logits = model_orig(tokens).data
    loaded_logits = model_loaded(tokens).data

    np.testing.assert_allclose(loaded_logits, orig_logits, atol=1e-12)

    # Check that tied weight sharing is preserved after reloading
    assert id(model_loaded.tok_embeddings.weight) == id(model_loaded.lm_head.weight)
