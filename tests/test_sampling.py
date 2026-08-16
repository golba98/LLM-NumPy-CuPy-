import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from src.generation.sampler import greedy_decode, top_k_sampling, top_p_sampling

def test_greedy_decode():
    logits = np.array([1.0, 5.0, 2.0, 0.5])
    assert greedy_decode(logits) == 1

def test_top_k_k_equals_1_is_greedy():
    logits = np.array([1.0, 5.0, 2.0, 0.5])
    token_id = top_k_sampling(logits, k=1, temperature=1.0)
    assert token_id == 1

def test_top_k_retains_k_candidates():
    logits = np.array([10.0, 9.0, 1.0, 0.0])
    rng = np.random.default_rng(42)
    samples = [top_k_sampling(logits, k=2, temperature=1.0, rng=rng) for _ in range(50)]
    assert all(s in [0, 1] for s in samples)

def test_top_p_sampling_cum_threshold():
    logits = np.array([10.0, 0.0, -10.0, -20.0])
    rng = np.random.default_rng(42)
    token_id = top_p_sampling(logits, p=0.9, temperature=1.0, rng=rng)
    assert token_id == 0
