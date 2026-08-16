import numpy as np
import random
from src.backend import current_backend

def set_seed(seed_val: int = 42) -> None:
    """
    Set random seeds for NumPy and Python standard library for reproducibility.
    """
    np.random.seed(seed_val)
    random.seed(seed_val)
    if current_backend().is_cuda:
        try:
            import cupy as cp
            cp.random.seed(seed_val)
        except Exception:
            # CPU runs remain reproducible even if CuPy is installed without
            # the optional cuRAND runtime.
            pass
