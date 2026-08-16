import pytest
import numpy as np
from src.config import LLMConfig
from src.nn.model import TinyLLM
from src.utils.seed import set_seed

def test_untrained_greedy_autoregressive_generation():
    set_seed(42)
    cfg = LLMConfig(vocab_size=16, max_seq_len=32, dim=16, num_layers=1, num_heads=2, hidden_dim=32)
    model = TinyLLM(cfg)

    prompt = [1, 5, 2]
    generated = model.generate(prompt, max_new_tokens=5, greedy=True)

    assert generated.shape == (1, 8) # 3 prompt + 5 generated = 8
    # Prefix must match prompt
    np.testing.assert_array_equal(generated[0, :3], prompt)
    # Generated token IDs must be in valid vocabulary range
    assert np.all(generated >= 0) and np.all(generated < cfg.vocab_size)

def test_generation_temperature_scaling():
    set_seed(42)
    cfg = LLMConfig(vocab_size=16, max_seq_len=32, dim=16, num_layers=1, num_heads=2, hidden_dim=32)
    model = TinyLLM(cfg)

    prompt = [0, 1]
    gen_hot = model.generate(prompt, max_new_tokens=4, temperature=2.0, greedy=False)
    assert gen_hot.shape == (1, 6)
