import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest

from llm_numpy.training.config import TrainingConfig


def test_training_config_rejects_invalid_gradient_guard_values():
    with pytest.raises(ValueError, match="min_gradient_norm"):
        TrainingConfig(min_gradient_norm=-1.0)
    with pytest.raises(ValueError, match="max_consecutive_small_gradient_steps"):
        TrainingConfig(max_consecutive_small_gradient_steps=-1)


def test_training_config_allows_zero_gradient_guard_to_be_disabled():
    config = TrainingConfig()
    assert config.min_gradient_norm == 0.0
    assert config.max_consecutive_small_gradient_steps == 0
