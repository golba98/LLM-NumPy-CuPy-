import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest
from llm_numpy.config import LLMConfig

def test_valid_llm_config():
    cfg = LLMConfig(vocab_size=100, max_seq_len=64, dim=32, num_layers=2, num_heads=4, hidden_dim=88)
    assert cfg.head_dim == 8

def test_invalid_llm_configs_raise():
    with pytest.raises(ValueError):
        LLMConfig(vocab_size=0)
    with pytest.raises(ValueError):
        LLMConfig(max_seq_len=-5)
    with pytest.raises(ValueError):
        LLMConfig(dim=33, num_heads=4) # 33 not divisible by 4
    with pytest.raises(ValueError):
        LLMConfig(dim=6, num_heads=2)  # head_dim = 3 (odd, fails RoPE even requirement)
