import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from llm_numpy.config import LLMConfig
from llm_numpy.nn.model import TinyLLM, language_model_loss
from llm_numpy.utils.seed import set_seed

def test_microbatch_gradient_accumulation_equivalence():
    set_seed(42)
    cfg = LLMConfig(vocab_size=16, max_seq_len=16, dim=8, num_layers=1, num_heads=2, hidden_dim=16)
    model1 = TinyLLM(cfg)

    inputs = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    targets = np.array([
        [2, 3, 4],
        [5, 6, 7],
        [8, 9, 10],
        [11, 12, 13]
    ])

    logits1, loss1 = model1(inputs, targets=targets)
    loss1.backward()

    set_seed(42)
    model2 = TinyLLM(cfg)

    mb1_inp, mb1_tgt = inputs[:2], targets[:2]
    mb2_inp, mb2_tgt = inputs[2:], targets[2:]

    _, loss_mb1 = model2(mb1_inp, targets=mb1_tgt)
    (loss_mb1 / 2.0).backward()

    _, loss_mb2 = model2(mb2_inp, targets=mb2_tgt)
    (loss_mb2 / 2.0).backward()

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert p1.grad is not None and p2.grad is not None
        np.testing.assert_allclose(p2.grad, p1.grad, atol=1e-5)
