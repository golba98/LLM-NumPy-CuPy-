import pytest
import numpy as np
from src.config import LLMConfig
from src.nn.model import TinyLLM

def test_full_model_causal_prefix_invariance():
    # End-to-End causal proof through multiple Transformer blocks:
    # Changing future tokens in sequence B must NOT alter logits for earlier tokens
    cfg = LLMConfig(vocab_size=32, max_seq_len=64, dim=16, num_layers=2, num_heads=2, hidden_dim=32)
    model = TinyLLM(cfg)

    seq_A = np.array([[1, 5, 10, 15, 20, 25]])
    seq_B = seq_A.copy()
    seq_B[0, 4:] = [2, 3] # Mutate tokens at t=4 and t=5

    logits_A = model(seq_A)
    logits_B = model(seq_B)

    # Logits for positions t=0..3 must be identical
    diff_prefix = np.max(np.abs(logits_A.data[0, :4, :] - logits_B.data[0, :4, :]))
    assert diff_prefix < 1e-12

def test_batch_isolation():
    # Changing batch item 1 must NOT alter predictions for batch item 0
    cfg = LLMConfig(vocab_size=32, max_seq_len=64, dim=16, num_layers=1, num_heads=2, hidden_dim=32)
    model = TinyLLM(cfg)

    batch_1 = np.array([[1, 2, 3], [4, 5, 6]])
    batch_2 = np.array([[1, 2, 3], [7, 8, 9]]) # Item 1 mutated

    logits_1 = model(batch_1)
    logits_2 = model(batch_2)

    diff_item_0 = np.max(np.abs(logits_1.data[0] - logits_2.data[0]))
    assert diff_item_0 < 1e-12
