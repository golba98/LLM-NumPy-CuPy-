import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
from llm_numpy.config import LLMConfig
from llm_numpy.nn.model import TinyLLM, language_model_loss
from llm_numpy.utils.seed import set_seed
from llm_numpy.utils.diagnostics import inspect_weights

def test_initialization_seed_reproducibility():
    cfg = LLMConfig(vocab_size=32, dim=16, num_layers=1, num_heads=2, hidden_dim=32)

    set_seed(123)
    model1 = TinyLLM(cfg)

    set_seed(123)
    model2 = TinyLLM(cfg)

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        np.testing.assert_array_equal(p1.data, p2.data)

def test_initial_logits_and_random_loss_baseline():
    set_seed(42)
    V = 64
    cfg = LLMConfig(vocab_size=V, max_seq_len=128, dim=32, num_layers=2, num_heads=4, hidden_dim=88)
    model = TinyLLM(cfg)

    tokens = np.array([[1, 5, 10, 15, 20]])
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    logits, loss = model(inputs, targets=targets)

    # Initial loss of a randomly initialized model over vocabulary size V should be roughly in the vicinity of ln(V)
    expected_baseline = np.log(V) # ln(64) = 4.15888
    actual_loss = loss.data.item()

    assert np.isfinite(actual_loss)
    # Sanity check: loss should be reasonable (e.g. between 3.0 and 6.0 for V=64)
    assert 3.0 <= actual_loss <= 6.0

def test_inspect_weights_finite_and_valid():
    cfg = LLMConfig(vocab_size=16, dim=8, num_layers=1, num_heads=2, hidden_dim=16)
    model = TinyLLM(cfg)

    reports = inspect_weights(model)
    assert len(reports) > 0
    for r in reports:
        assert r["finite"]
        assert not np.isnan(r["mean"])
