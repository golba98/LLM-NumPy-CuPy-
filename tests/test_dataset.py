import pytest
import numpy as np
from src.data.dataset import LanguageModelDataset

def test_dataset_windowing_and_target_shifting():
    tokens = list(range(10, 25)) # 15 tokens
    ctx_len = 4
    stride = 4

    ds = LanguageModelDataset(tokens, context_length=ctx_len, stride=stride)
    
    # samples: 0: 10..14, 1: 14..18, 2: 18..22 are complete windows.
    assert len(ds) == 3

    inp0, tgt0 = ds[0]
    np.testing.assert_array_equal(inp0, [10, 11, 12, 13])
    np.testing.assert_array_equal(tgt0, [11, 12, 13, 14])
    np.testing.assert_array_equal(ds[2][0], [18, 19, 20, 21])

def test_dataset_out_of_bounds_raises():
    tokens = list(range(10))
    ds = LanguageModelDataset(tokens, context_length=4)
    with pytest.raises(IndexError):
        _ = ds[100]
