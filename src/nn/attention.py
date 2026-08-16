import numpy as np  # type-only annotations
from src.backend import array_module
from typing import Optional, Tuple, Union
from src.nn.module import Module
from src.nn.linear import Linear
from src.tensor import Tensor
from src.ops.basic import concat_op, to_tensor

def softmax(x: Union[Tensor, float, np.ndarray], axis: int = -1) -> Tensor:
    """
    Standalone numerically stable Softmax operation with efficient row-wise autograd derivative:
    dx = s * (g - sum(g * s))
    """
    x_t = to_tensor(x)
    xp = array_module(x_t.data)
    max_x = xp.max(x_t.data, axis=axis, keepdims=True)
    exp_x = xp.exp(x_t.data - max_x)
    s_data = exp_x / xp.sum(exp_x, axis=axis, keepdims=True)

    out = Tensor(s_data, requires_grad=x_t.requires_grad, _parents=(x_t,), _op="Softmax")

    def _backward():
        if out.grad is None:
            return
        if x_t.requires_grad:
            sum_grad_s = xp.sum(out.grad * s_data, axis=axis, keepdims=True)
            grad_x = s_data * (out.grad - sum_grad_s)
            x_t.grad = grad_x if x_t.grad is None else x_t.grad + grad_x

    out._backward = _backward
    return out

def get_causal_mask(seq_len: int) -> Tensor:
    """
    Generates a sequence-length causal mask matrix M where M[i, j] = 0 for j <= i and -1e9 for j > i.
    """
    mask_data = np.full((seq_len, seq_len), -1e9, dtype=np.float64)
    mask_data = np.triu(mask_data, k=1)
    return Tensor(mask_data, requires_grad=False)

def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool = True,
    causal_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Computes Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax((QK^T / sqrt(d_k)) + M) V
    """
    d_k = q.shape[-1]
    xp = array_module(q.data)
    scale = xp.asarray(xp.sqrt(d_k), dtype=q.data.dtype)
    scores = (q @ k.transpose(-1, -2)) / scale
    
    if causal:
        seq_len = q.shape[-2]
        mask = causal_mask if causal_mask is not None else get_causal_mask(seq_len)
        if mask.device != q.device or mask.dtype != q.data.dtype:
            mask.to(q.device, dtype=q.data.dtype)
        scores = scores + mask
        
    attn_weights = softmax(scores, axis=-1)
    output = attn_weights @ v
    return output

class MultiHeadSelfAttention(Module):
    """
    Multi-Head Self-Attention layer with integrated Rotary Position Embeddings (RoPE).
    Input shape: (B, T, C)
    Output shape: (B, T, C)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        bias: bool = False,
        fused_qkv: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"MultiHeadSelfAttention expected dim {dim} to be divisible by num_heads {num_heads}.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.fused_qkv = fused_qkv

        self.q_proj = Linear(dim, dim, bias=bias)
        self.k_proj = Linear(dim, dim, bias=bias)
        self.v_proj = Linear(dim, dim, bias=bias)
        self.out_proj = Linear(dim, dim, bias=bias)
        self._causal_mask_cache = {}

        from src.nn.rope import RotaryEmbedding
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, base=rope_base)

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        B, T, C = x.shape
        if C != self.dim:
            raise ValueError(f"Input channel dim {C} does not match model dim {self.dim}")

        # Project inputs. Concatenating the existing parameter tensors keeps
        # checkpoint ordering/schema intact while reducing three GEMMs to one.
        if self.fused_qkv:
            qkv_weight = concat_op((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), axis=1)
            qkv = x @ qkv_weight
            Q, K, V = qkv[:, :, :self.dim], qkv[:, :, self.dim:2 * self.dim], qkv[:, :, 2 * self.dim:]
        else:
            Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE to Q and K
        Q, K = self.rope(Q, K)

        # Cache the backend-resident mask by sequence length/device/dtype.
        causal_mask = None
        if causal:
            key = (T, Q.device, str(Q.dtype))
            causal_mask = self._causal_mask_cache.get(key)
            if causal_mask is None:
                causal_mask = get_causal_mask(T)
                causal_mask.to(Q.device, dtype=Q.dtype)
                self._causal_mask_cache[key] = causal_mask

        # Scaled dot product attention
        attn_out = scaled_dot_product_attention(Q, K, V, causal=causal, causal_mask=causal_mask)

        # Merge heads back to (B, T, C)
        attn_merged = attn_out.transpose(0, 2, 1, 3).reshape(B, T, C)

        return self.out_proj(attn_merged)
