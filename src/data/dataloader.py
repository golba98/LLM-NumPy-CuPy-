import numpy as np
from typing import Iterator, Tuple
from src.data.dataset import LanguageModelDataset

class DataLoader:
    """
    Custom DataLoader yielding mini-batches of inputs and targets of shape (B, T).
    Supports deterministic shuffling using seed + epoch.
    """
    def __init__(
        self,
        dataset: LanguageModelDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(self.dataset)
        indices = np.arange(n)

        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(indices)

        current_epoch = self.epoch
        self.epoch += 1

        for i in range(0, n, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                continue

            batch_inputs = []
            batch_targets = []
            for idx in batch_idx:
                inp, tgt = self.dataset[idx]
                batch_inputs.append(inp)
                batch_targets.append(tgt)

            yield np.array(batch_inputs, dtype=int), np.array(batch_targets, dtype=int)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
