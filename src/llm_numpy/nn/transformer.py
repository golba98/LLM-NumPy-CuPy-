from llm_numpy.nn.module import Module
from llm_numpy.nn.norm import RMSNorm
from llm_numpy.nn.attention import MultiHeadSelfAttention
from llm_numpy.nn.ffn import SwiGLU
from llm_numpy.tensor import Tensor

class TransformerBlock(Module):
    """
    Pre-Norm Decoder-Only Transformer Block:
    1. h = x + Attention(RMSNorm(x))
    2. y = h + SwiGLU(RMSNorm(h))
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        eps: float = 1e-5
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.attn_norm = RMSNorm(dim, eps=eps)
        self.attn = MultiHeadSelfAttention(dim, num_heads, max_seq_len=max_seq_len, rope_base=rope_base)
        self.ffn_norm = RMSNorm(dim, eps=eps)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        # Pre-Norm Attention with Residual Connection
        h = x + self.attn(self.attn_norm(x), causal=causal)

        # Pre-Norm SwiGLU FFN with Residual Connection
        out = h + self.ffn(self.ffn_norm(h))

        return out

    def __repr__(self) -> str:
        return f"TransformerBlock(dim={self.dim}, num_heads={self.num_heads}, hidden_dim={self.hidden_dim})"
