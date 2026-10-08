import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest

from llm_numpy.backend import cuda_available, to_cpu
from llm_numpy.config import LLMConfig
from llm_numpy.nn.model import TinyLLM
from llm_numpy.optim.adamw import AdamW


pytestmark = pytest.mark.skipif(not cuda_available(), reason="CuPy/CUDA unavailable")


def test_cuda_fp16_keeps_fp32_adamw_state_and_finite_loss():
    config = LLMConfig(vocab_size=32, max_seq_len=8, dim=16, num_layers=1,
                       num_heads=4, hidden_dim=32)
    model = TinyLLM(config).to("cuda", dtype=np.float16)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0).to("cuda", dtype=np.float16)
    inputs = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    targets = np.array([[2, 3, 4, 5, 6, 7]], dtype=np.int64)
    losses = []
    for _ in range(5):
        _, loss = model(inputs, targets=targets)
        losses.append(float(to_cpu(loss.data)))
        assert np.isfinite(losses[-1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    assert losses[-1] < losses[0]
    for group in optimizer.param_groups:
        assert all(str(moment.dtype) == "float32" for moment in group["m"])
        assert all(str(moment.dtype) == "float32" for moment in group["v"])
