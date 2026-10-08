import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
import os
import numpy as np
from llm_numpy.config import LLMConfig
from llm_numpy.nn.model import TinyLLM
from llm_numpy.optim.adamw import AdamW
from llm_numpy.data import DataLoader, LanguageModelDataset
from llm_numpy.training.checkpoint import save_checkpoint, load_checkpoint
from llm_numpy.training.config import TrainingConfig
from llm_numpy.training.trainer import Trainer
from llm_numpy.utils.seed import set_seed

def test_full_checkpoint_resume_equivalence(tmp_path):
    cfg = LLMConfig(vocab_size=16, max_seq_len=16, dim=8, num_layers=1, num_heads=2, hidden_dim=16, tie_embeddings=True)
    inputs = np.array([[1, 4, 2, 7, 9]])
    targets = np.array([[4, 2, 7, 9, 3]])

    set_seed(42)
    model_A = TinyLLM(cfg)
    opt_A = AdamW(model_A.parameters(), lr=0.01)

    for _ in range(20):
        opt_A.zero_grad()
        _, loss = model_A(inputs, targets=targets)
        loss.backward()
        opt_A.step()

    loss_A = model_A(inputs, targets=targets)[1].data.item()

    set_seed(42)
    model_B = TinyLLM(cfg)
    opt_B = AdamW(model_B.parameters(), lr=0.01)

    for step in range(10):
        opt_B.zero_grad()
        _, loss = model_B(inputs, targets=targets)
        loss.backward()
        opt_B.step()

    ckpt_path = str(tmp_path / "checkpoint.npz")
    save_checkpoint(ckpt_path, model_B, opt_B, step=10, epoch=1, tokens_processed=50)

    model_C = TinyLLM(cfg)
    opt_C = AdamW(model_C.parameters(), lr=0.01)
    step_c, epoch_c, tokens_c = load_checkpoint(ckpt_path, model_C, opt_C)

    assert step_c == 10
    assert epoch_c == 1
    assert tokens_c == 50

    for _ in range(10):
        opt_C.zero_grad()
        _, loss = model_C(inputs, targets=targets)
        loss.backward()
        opt_C.step()

    loss_C = model_C(inputs, targets=targets)[1].data.item()

    np.testing.assert_allclose(loss_C, loss_A, atol=1e-6)
    for p_a, p_c in zip(model_A.parameters(), model_C.parameters()):
        np.testing.assert_allclose(p_c.data, p_a.data, atol=1e-6)


def test_periodic_checkpoint_persists_dataset_manifest_hash(tmp_path):
    cfg = LLMConfig(vocab_size=16, max_seq_len=4, dim=8, num_layers=1, num_heads=2, hidden_dim=16)
    model = TinyLLM(cfg)
    optimizer = AdamW(model.parameters(), lr=0.01)
    dataset = LanguageModelDataset(np.arange(12), context_length=4)
    loader = DataLoader(dataset, batch_size=1, seed=42)
    checkpoint_dir = tmp_path / "checkpoints"
    manifest_hash = "a" * 64
    training_config = TrainingConfig(
        batch_size=1, context_length=4, max_steps=1, eval_interval=1,
        checkpoint_interval=1, checkpoint_dir=str(checkpoint_dir),
        dataset_manifest_sha256=manifest_hash,
    )

    Trainer(model, optimizer, loader, config=training_config).train()

    with np.load(checkpoint_dir / "checkpoint_step_000001.npz", allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
    assert metadata["dataset_manifest_sha256"] == manifest_hash
