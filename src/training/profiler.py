"""Optional, low-overhead timing for the NumPy training reference."""

from __future__ import annotations

import platform
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Iterator, List

import numpy as np


class StepProfiler:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.samples: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.samples[name].append((time.perf_counter() - started) * 1000.0)

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {name: {"mean_ms": float(np.mean(values)),
                       "median_ms": float(np.median(values)),
                       "min_ms": float(np.min(values)),
                       "max_ms": float(np.max(values)),
                       "std_ms": float(np.std(values)),
                       "count": len(values)}
                for name, values in self.samples.items()}

    @staticmethod
    def environment() -> Dict[str, str]:
        return {"python": sys.version.split()[0], "numpy": np.__version__,
                "platform": platform.platform(), "machine": platform.processor() or platform.machine()}
