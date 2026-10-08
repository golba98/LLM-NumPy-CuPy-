import numpy as np
from typing import List, Dict, Any
from llm_numpy.nn.module import Module

def inspect_gradients(module: Module) -> List[Dict[str, Any]]:
    """
    Inspects gradient statistics across all unique parameters in module.
    """
    reports = []
    for name, value in module.__dict__.items():
        if hasattr(value, "grad") and value.grad is not None:
            g = value.grad
            reports.append({
                "name": name,
                "shape": value.shape,
                "grad_norm": float(np.linalg.norm(g)),
                "finite": bool(np.all(np.isfinite(g)))
            })
    # Also traverse submodules
    for p in module.parameters():
        if p.grad is not None:
            g = p.grad
            reports.append({
                "shape": p.shape,
                "grad_norm": float(np.linalg.norm(g)),
                "finite": bool(np.all(np.isfinite(g)))
            })
    return reports

def inspect_weights(module: Module) -> List[Dict[str, Any]]:
    """
    Inspects weight statistics (mean, std, min, max, L2 norm) for initialization sanity checks.
    """
    reports = []
    for p in module.parameters():
        d = p.data
        reports.append({
            "shape": p.shape,
            "mean": float(np.mean(d)),
            "std": float(np.std(d)),
            "min": float(np.min(d)),
            "max": float(np.max(d)),
            "l2_norm": float(np.linalg.norm(d)),
            "finite": bool(np.all(np.isfinite(d)))
        })
    return reports
