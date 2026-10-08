import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest
from llm_numpy.config import LLMConfig
from llm_numpy.nn.model import TinyLLM
from llm_numpy.nn.module import count_parameters

def test_parameter_count_formula_matches_programmatic_count():
    # Tied configuration
    cfg_tied = LLMConfig(vocab_size=64, max_seq_len=128, dim=32, num_layers=2, num_heads=4, hidden_dim=88, tie_embeddings=True)
    model_tied = TinyLLM(cfg_tied)

    V, C, L, F = cfg_tied.vocab_size, cfg_tied.dim, cfg_tied.num_layers, cfg_tied.hidden_dim
    block_params = 4 * (C * C) + 3 * (C * F) + 2 * C
    expected_tied = V * C + L * block_params + C

    assert count_parameters(model_tied) == expected_tied

    # Untied configuration
    cfg_untied = LLMConfig(vocab_size=64, max_seq_len=128, dim=32, num_layers=2, num_heads=4, hidden_dim=88, tie_embeddings=False)
    model_untied = TinyLLM(cfg_untied)

    expected_untied = 2 * V * C + L * block_params + C
    assert count_parameters(model_untied) == expected_untied
