import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from llm_numpy.config import LLMConfig
from llm_numpy.nn.model import TinyLLM
from llm_numpy.optim.adamw import AdamW
from llm_numpy.optim.clip import clip_grad_norm_
from llm_numpy.utils.seed import set_seed

def test_one_batch_overfitting_critical():
    set_seed(42)
    V = 16
    cfg = LLMConfig(vocab_size=V, max_seq_len=16, dim=32, num_layers=2, num_heads=4, hidden_dim=64, tie_embeddings=True)
    model = TinyLLM(cfg)
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.0)

    inputs = np.array([[1, 5, 2, 8, 3, 10], [4, 9, 0, 7, 11, 2]])
    targets = np.array([[5, 2, 8, 3, 10, 14], [9, 0, 7, 11, 2, 6]])

    initial_logits, initial_loss = model(inputs, targets=targets)
    initial_loss_val = initial_loss.data.item()

    assert 2.0 <= initial_loss_val <= 4.0

    for step in range(120):
        optimizer.zero_grad()
        logits, loss = model(inputs, targets=targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    final_logits, final_loss = model(inputs, targets=targets)
    final_loss_val = final_loss.data.item()

    preds = np.argmax(final_logits.data, axis=-1)
    acc = np.mean(preds == targets) * 100.0

    assert final_loss_val < 0.1
    assert acc > 95.0

def test_tiny_corpus_overfitting():
    set_seed(42)
    V = 32
    cfg = LLMConfig(vocab_size=V, max_seq_len=16, dim=32, num_layers=2, num_heads=4, hidden_dim=64)
    model = TinyLLM(cfg)
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.0)

    inputs = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])
    targets = np.array([
        [2, 3, 4, 5],
        [6, 7, 8, 9],
        [10, 11, 12, 13]
    ])

    for _ in range(100):
        optimizer.zero_grad()
        logits, loss = model(inputs, targets=targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    final_logits, final_loss = model(inputs, targets=targets)
    assert final_loss.data.item() < 0.2
