import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
from llm_numpy.nn.model import shift_for_next_token

def test_shift_for_next_token_alignment():
    tokens = np.array([[10, 20, 30, 40, 50]])
    inputs, targets = shift_for_next_token(tokens)

    np.testing.assert_array_equal(inputs, [[10, 20, 30, 40]])
    np.testing.assert_array_equal(targets, [[20, 30, 40, 50]])

def test_shift_for_next_token_invalid_len_raises():
    with pytest.raises(ValueError):
        shift_for_next_token([[10]]) # length <= 1 raises
