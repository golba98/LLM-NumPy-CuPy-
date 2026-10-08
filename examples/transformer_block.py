import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from llm_numpy.tensor import Tensor

from llm_numpy.nn.transformer import TransformerBlock
from llm_numpy.nn.module import count_parameters
from llm_numpy.utils.seed import set_seed

def run_transformer_block_inspection():
    set_seed(42)
    print("=" * 65)
    print("NumPy Transformer Runtime — Inspection Benchmark")
    print("=" * 65)

    B, T, C = 2, 16, 64
    heads = 4
    head_dim = C // heads
    ffn_dim = 176

    block = TransformerBlock(
        dim=C,
        num_heads=heads,
        hidden_dim=ffn_dim,
        max_seq_len=128
    )

    x = Tensor(np.random.randn(B, T, C), requires_grad=True)

    # 1. Forward Pass
    y = block(x, causal=True)
    forward_pass = (y.shape == (B, T, C)) and np.all(np.isfinite(y.data))

    # 2. Backward Pass
    loss = y.sum()
    loss.backward()
    
    input_grad_pass = (x.grad is not None) and np.all(np.isfinite(x.grad))
    param_grads_pass = all((p.grad is not None and np.all(np.isfinite(p.grad))) for p in block.parameters())

    # 3. Causal Prefix-Invariance Test
    # Changing future tokens in sequence must NOT affect predictions for earlier tokens
    seq_A = np.random.randn(1, 8, C)
    seq_B = seq_A.copy()
    seq_B[0, 6:, :] = np.random.randn(1, 2, C) # Change tokens at t=6 and t=7

    x_A = Tensor(seq_A, requires_grad=False)
    x_B = Tensor(seq_B, requires_grad=False)

    out_A = block(x_A, causal=True)
    out_B = block(x_B, causal=True)

    # Positions t=0..5 must be identical between out_A and out_B
    causal_diff = np.max(np.abs(out_A.data[0, :6, :] - out_B.data[0, :6, :]))
    causal_pass = causal_diff < 1e-12

    total_params = count_parameters(block)

    print(f"Batch size:       {B}")
    print(f"Sequence length:  {T}")
    print(f"Model dimension:  {C}")
    print(f"Attention heads:  {heads}")
    print(f"Head dimension:   {head_dim}")
    print(f"FFN dimension:    {ffn_dim}")
    print("-" * 65)
    print(f"Input shape:      {x.shape}")
    print(f"Output shape:     {y.shape}")
    print(f"Parameters:       {total_params:,}")
    print(f"Output finite:    {'yes' if np.all(np.isfinite(y.data)) else 'no'}")
    print(f"Input grad finite: {'yes' if input_grad_pass else 'no'}")
    print(f"Parameter grads:   {'yes' if param_grads_pass else 'no'}")
    print("-" * 65)
    print(f"Forward:          {'PASS' if forward_pass else 'FAIL'}")
    print(f"Backward:         {'PASS' if input_grad_pass and param_grads_pass else 'FAIL'}")
    print(f"Causal test:      {'PASS' if causal_pass else 'FAIL'}")
    print("=" * 65 + "\n")

    return total_params

if __name__ == "__main__":
    run_transformer_block_inspection()
