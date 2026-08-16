import numpy as np
from src.backend import array_module
from typing import Tuple
from src.nn.module import Module
from src.tensor import Tensor

class RotaryEmbedding(Module):
    """
    Rotary Position Embeddings (RoPE).
    Applies position-dependent 2D rotations to Query and Key feature pairs.
    Constrained to even head dimensions.
    """
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding requires even head_dim, but got {head_dim}")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute theta frequencies: base^(-2i / head_dim)
        inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
        
        # Position indices m = 0, 1, ..., max_seq_len - 1
        t = np.arange(max_seq_len, dtype=np.float64)
        
        # Outer product: freqs[m, i] = m * inv_freq[i]
        freqs = np.outer(t, inv_freq) # (max_seq_len, head_dim // 2)

        # Precompute sine and cosine lookup tables
        self.cos_data = np.cos(freqs)
        self.sin_data = np.sin(freqs)

    def _rotate(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # x shape: (B, H, T, D)
        half_dim = self.head_dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]

        # Rotate pairs: x1' = x1 * cos - x2 * sin, x2' = x1 * sin + x2 * cos
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        # Concatenate along last dimension to restore shape (B, H, T, D)
        return Tensor.concat([x1_rot, x2_rot], axis=-1)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        # q and k shape: (B, H, T, D)
        T = q.shape[-2]
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        # Reshape cos and sin for broadcasting: (1, 1, T, D//2)
        xp = array_module(q.data)
        cos_table = Tensor(xp.asarray(self.cos_data[:T], dtype=q.data.dtype).reshape(1, 1, T, self.head_dim // 2), requires_grad=False)
        sin_table = Tensor(xp.asarray(self.sin_data[:T], dtype=q.data.dtype).reshape(1, 1, T, self.head_dim // 2), requires_grad=False)

        q_rot = self._rotate(q, cos_table, sin_table)
        k_rot = self._rotate(k, cos_table, sin_table)

        return q_rot, k_rot

    def __repr__(self) -> str:
        return f"RotaryEmbedding(head_dim={self.head_dim}, max_seq_len={self.max_seq_len}, base={self.base})"
