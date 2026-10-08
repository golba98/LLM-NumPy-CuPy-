import numpy as np
from typing import Sequence, Tuple, Optional


class LanguageModelDataset:
    """Window a token stream into aligned input/next-token target pairs."""

    def __init__(self, token_ids: Sequence[int], context_length: int, stride: Optional[int] = None):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self.token_ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        self.context_length = int(context_length)
        self.stride = int(context_length if stride is None else stride)
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        available = len(self.token_ids) - self.context_length - 1
        self.num_samples = max(0, available // self.stride + 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        if index < 0:
            index += self.num_samples
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"Index {index} out of range for dataset of size {self.num_samples}")
        start = index * self.stride
        window = self.token_ids[start : start + self.context_length + 1]
        return window[:-1].copy(), window[1:].copy()
