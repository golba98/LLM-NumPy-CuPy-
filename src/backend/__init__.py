"""Array backends for the from-scratch LLM.

NumPy remains the default CPU backend. CuPy is imported lazily only when a
CUDA backend is requested, so CPU-only installations do not need CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np


_DEFAULT_DEVICE = "cpu"


@dataclass(frozen=True)
class Backend:
    name: str
    device: str
    xp: Any

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def gpu_name(self) -> str | None:
        if not self.is_cuda:
            return None
        try:
            name = self.xp.cuda.runtime.getDeviceProperties(
                self.xp.cuda.Device().id
            )["name"]
            return name.decode() if isinstance(name, bytes) else str(name)
        except Exception:
            return "CUDA device"


def _load_cupy():
    try:
        return importlib.import_module("cupy")
    except ImportError as exc:
        raise RuntimeError(
            "CUDA was requested but CuPy is not installed. Install the matching "
            "package, for example: python3 -m pip install cupy-cuda12x"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "CuPy could not initialize the CUDA runtime. Check the NVIDIA driver "
            "and CuPy/CUDA compatibility."
        ) from exc


def cuda_available() -> bool:
    try:
        cp = _load_cupy()
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except (RuntimeError, Exception):
        return False


def get_backend(device: str = "auto") -> Backend:
    normalized = str(device).lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: cpu, cuda, auto")
    if normalized == "auto":
        normalized = "cuda" if cuda_available() else "cpu"
    if normalized == "cpu":
        return Backend("NumPy", "cpu", np)
    cp = _load_cupy()
    try:
        if int(cp.cuda.runtime.getDeviceCount()) <= 0:
            raise RuntimeError("CUDA was requested but no CUDA device was found.")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "CUDA was requested but the CUDA runtime is unavailable or the driver "
            "is incompatible."
        ) from exc
    return Backend("CuPy", "cuda", cp)


def set_default_device(device: str) -> Backend:
    global _DEFAULT_DEVICE
    backend = get_backend(device)
    _DEFAULT_DEVICE = backend.device
    return backend


def current_backend() -> Backend:
    return get_backend(_DEFAULT_DEVICE)


def array_module(value: Any):
    """Return NumPy or CuPy for an array without importing CuPy eagerly."""
    module_name = type(value).__module__.split(".")[0]
    if module_name == "cupy":
        return _load_cupy()
    return np


def is_cuda_array(value: Any) -> bool:
    return type(value).__module__.split(".")[0] == "cupy"


def to_device(value: Any, device: str):
    backend = get_backend(device)
    if backend.is_cuda:
        return backend.xp.asarray(value)
    if is_cuda_array(value):
        return backend.xp.asarray(value.get())
    return backend.xp.asarray(value)


def to_cpu(value: Any):
    if is_cuda_array(value):
        return value.get()
    return np.asarray(value)


def synchronize(device: str = "cuda") -> None:
    if device == "cuda":
        get_backend("cuda").xp.cuda.Stream.null.synchronize()


def describe(device: str = "auto") -> dict[str, str]:
    backend = get_backend(device)
    result = {"backend": backend.name, "device": backend.device}
    if backend.is_cuda:
        result["gpu"] = backend.gpu_name or "unknown"
        result["cuda_runtime"] = str(backend.xp.cuda.runtime.runtimeGetVersion())
    else:
        result["gpu"] = "none"
    return result


def memory_stats(device: str = "cuda") -> dict[str, int]:
    """Return device and CuPy pool memory in bytes without forcing CPU use."""
    backend = get_backend(device)
    if not backend.is_cuda:
        return {"used_bytes": 0, "free_bytes": 0, "pool_used_bytes": 0, "pool_total_bytes": 0}
    free_bytes, total_bytes = backend.xp.cuda.Device().mem_info
    pool = backend.xp.get_default_memory_pool()
    return {"used_bytes": int(total_bytes - free_bytes), "free_bytes": int(free_bytes),
            "pool_used_bytes": int(pool.used_bytes()), "pool_total_bytes": int(pool.total_bytes())}


__all__ = [
    "Backend", "array_module", "cuda_available", "current_backend", "describe",
    "get_backend", "is_cuda_array", "memory_stats", "set_default_device", "synchronize", "to_cpu",
    "to_device",
]
