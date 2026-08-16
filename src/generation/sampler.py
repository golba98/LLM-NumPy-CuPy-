import numpy as np
from typing import Optional


def _validate_temperature(temperature: float) -> None:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities)


def greedy_decode(logits: np.ndarray) -> int:
    return int(np.argmax(np.asarray(logits), axis=-1))


def top_k_sampling(logits: np.ndarray, k: int = 50, temperature: float = 1.0,
                   rng: Optional[np.random.Generator] = None) -> int:
    values = np.asarray(logits, dtype=float)
    if values.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if k <= 0 or k > values.size:
        raise ValueError("k must be in the range [1, vocab_size]")
    _validate_temperature(temperature)
    if k == 1:
        return greedy_decode(values)
    indices = np.argsort(values)[-k:]
    probabilities = _softmax(values[indices] / temperature)
    generator = rng if rng is not None else np.random.default_rng()
    return int(generator.choice(indices, p=probabilities))


def top_p_sampling(logits: np.ndarray, p: float = 0.9, temperature: float = 1.0,
                   rng: Optional[np.random.Generator] = None) -> int:
    values = np.asarray(logits, dtype=float)
    if values.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if not 0 < p <= 1:
        raise ValueError("p must satisfy 0 < p <= 1")
    _validate_temperature(temperature)
    sorted_indices = np.argsort(values)[::-1]
    sorted_probabilities = _softmax(values[sorted_indices] / temperature)
    cumulative = np.cumsum(sorted_probabilities)
    cutoff = int(np.searchsorted(cumulative, p, side="left")) + 1
    selected_indices = sorted_indices[:cutoff]
    probabilities = sorted_probabilities[:cutoff]
    probabilities = probabilities / np.sum(probabilities)
    generator = rng if rng is not None else np.random.default_rng()
    return int(generator.choice(selected_indices, p=probabilities))
