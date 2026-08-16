from dataclasses import dataclass

@dataclass
class LLMConfig:
    """
    Configuration for TinyLLM language model architecture.
    Enforces strict architectural invariants during initialization.
    """
    vocab_size: int = 64
    max_seq_len: int = 128
    dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    hidden_dim: int = 176
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len}")
        if self.dim <= 0:
            raise ValueError(f"dim must be > 0, got {self.dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})")
        
        head_dim = self.dim // self.num_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE compatibility")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads
