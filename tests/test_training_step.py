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
from llm_numpy.optim.adamw import AdamW
from llm_numpy.optim.clip import clip_grad_norm_

def test_single_training_step_parameter_mutation_and_loss_reduction():
    cfg = LLMConfig(vocab_size=16, max_seq_len=16, dim=16, num_layers=1, num_heads=2, hidden_dim=32)
    model = TinyLLM(cfg)
    optimizer = AdamW(model.parameters(), lr=0.01)

    inputs = np.array([[1, 4, 2, 7, 9]])
    targets = np.array([[4, 2, 7, 9, 3]])

    w_before = model.tok_embeddings.weight.data.copy()

    optimizer.zero_grad()
    logits_1, loss_1 = model(inputs, targets=targets)
    loss_1_val = loss_1.data.item()
    loss_1.backward()

    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    w_after = model.tok_embeddings.weight.data.copy()

    assert not np.allclose(w_before, w_after)

    optimizer.zero_grad()
    logits_2, loss_2 = model(inputs, targets=targets)
    loss_2_val = loss_2.data.item()

    assert loss_2_val < loss_1_val
