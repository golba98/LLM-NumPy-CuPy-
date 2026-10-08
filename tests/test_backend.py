import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest

from llm_numpy.backend import cuda_available, describe, get_backend


def test_cpu_backend_is_lazy_and_explicit():
    backend = get_backend("cpu")
    assert backend.name == "NumPy"
    assert backend.device == "cpu"
    assert backend.xp is np
    assert describe("cpu")["gpu"] == "none"


def test_invalid_device_is_actionable():
    with pytest.raises(ValueError, match="cpu, cuda, auto"):
        get_backend("tpu")


@pytest.mark.skipif(not cuda_available(), reason="CuPy/CUDA unavailable")
def test_cuda_backend_reports_device():
    backend = get_backend("cuda")
    assert backend.name == "CuPy"
    assert backend.device == "cuda"
    assert backend.gpu_name
