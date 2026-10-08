import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
from llm_numpy.data.dataset import LanguageModelDataset
from llm_numpy.data.dataloader import DataLoader

def test_dataloader_batch_shapes_and_reproducible_shuffling():
    tokens = list(range(100))
    ds = LanguageModelDataset(tokens, context_length=8, stride=8)

    loader1 = DataLoader(ds, batch_size=2, shuffle=True, seed=42)
    loader2 = DataLoader(ds, batch_size=2, shuffle=True, seed=42)

    batches1 = list(loader1)
    batches2 = list(loader2)

    assert len(batches1) == len(batches2)
    for (inp1, tgt1), (inp2, tgt2) in zip(batches1, batches2):
        assert inp1.shape == (2, 8)
        assert tgt1.shape == (2, 8)
        np.testing.assert_array_equal(inp1, inp2)
        np.testing.assert_array_equal(tgt1, tgt2)

def test_dataloader_drop_last():
    tokens = list(range(100))
    ds = LanguageModelDataset(tokens, context_length=4, stride=4)
    # Total samples = (100 - 4 - 1) // 4 + 1 = 95 // 4 + 1 = 24 samples
    loader = DataLoader(ds, batch_size=5, drop_last=True)
    batches = list(loader)

    # 24 // 5 = 4 batches of size 5
    assert len(batches) == 4
    for inp, tgt in batches:
        assert inp.shape == (5, 4)
