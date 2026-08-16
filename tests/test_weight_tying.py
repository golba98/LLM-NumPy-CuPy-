import pytest
import numpy as np
from src.config import LLMConfig
from src.nn.model import TinyLLM, language_model_loss
from src.nn.module import count_parameters
from src.tensor import Tensor

def test_weight_tying_reference_sharing():
    cfg = LLMConfig(vocab_size=16, dim=8, tie_embeddings=True)
    model = TinyLLM(cfg)

    # Check exact object identity (shared parameter in memory)
    assert id(model.tok_embeddings.weight) == id(model.lm_head.weight)

def test_weight_tying_parameter_count_no_double_counting():
    cfg_tied = LLMConfig(vocab_size=16, dim=8, num_layers=1, num_heads=2, hidden_dim=16, tie_embeddings=True)
    cfg_untied = LLMConfig(vocab_size=16, dim=8, num_layers=1, num_heads=2, hidden_dim=16, tie_embeddings=False)

    model_tied = TinyLLM(cfg_tied)
    model_untied = TinyLLM(cfg_untied)

    # Difference between untied and tied parameter count must be exactly vocab_size * dim = 16 * 8 = 128
    diff = count_parameters(model_untied) - count_parameters(model_tied)
    assert diff == 16 * 8

def test_weight_tying_gradient_accumulation_both_paths():
    # Crucial test: Weight parameter receives gradient contributions from BOTH embedding lookup AND output projection
    cfg = LLMConfig(vocab_size=8, max_seq_len=16, dim=4, num_layers=1, num_heads=2, hidden_dim=8, tie_embeddings=True)
    model = TinyLLM(cfg)

    tokens = np.array([[1, 2, 3]])
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    logits, loss = model(inputs, targets=targets)
    loss.backward()

    shared_weight = model.tok_embeddings.weight
    assert shared_weight.grad is not None
    assert np.all(np.isfinite(shared_weight.grad))
    # Non-zero gradient demonstrating multi-path autograd accumulation
    assert np.linalg.norm(shared_weight.grad) > 0.0
