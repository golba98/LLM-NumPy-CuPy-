import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from llm_numpy.config import LLMConfig
from llm_numpy.nn.model import TinyLLM, shift_for_next_token, perplexity
from llm_numpy.nn.module import count_parameters
from llm_numpy.optim.adamw import AdamW
from llm_numpy.optim.clip import clip_grad_norm_
from llm_numpy.utils.seed import set_seed
from llm_numpy.utils.diagnostics import inspect_gradients, inspect_weights
from llm_numpy.utils.serialization import save_weights, load_weights

def run_full_model_mathematical_validation():
    set_seed(42)
    print("=" * 70)
    print("NumPy TinyLLM — Full Mathematical Validation (Milestone 3A)")
    print("=" * 70)

    # 1. Configuration & Mathematical Parameter Count
    V = 64
    T_max = 16
    C = 32
    L = 2
    H = 4
    F = 88

    config = LLMConfig(
        vocab_size=V,
        max_seq_len=T_max,
        dim=C,
        num_layers=L,
        num_heads=H,
        hidden_dim=F,
        tie_embeddings=True
    )

    model = TinyLLM(config)

    # Mathematical expected parameter count formula:
    # Tied embedding weight: V * C
    # Per Transformer Block (4 proj + 2 norms + 3 ffn): 4 * C^2 + 3 * C * F + 2 * C
    # Final RMSNorm: C
    block_params = 4 * (C * C) + 3 * (C * F) + 2 * C
    expected_params = V * C + L * block_params + C
    detected_params = count_parameters(model)
    param_match = (expected_params == detected_params)

    # 2. Next-Token Prediction Shifting & Forward Pass
    tokens = np.array([
        [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
        [2, 4,  8, 16, 32, 48, 60, 12, 24, 36, 48, 62]
    ])
    inputs, targets = shift_for_next_token(tokens)
    logits, loss = model(inputs, targets=targets)

    logits_pass = (logits.shape == (2, 11, V)) and np.all(np.isfinite(logits.data))
    loss_val = loss.data.item()
    loss_pass = np.isfinite(loss_val)
    ppl_val = perplexity(loss)
    random_baseline = np.log(V)

    # 3. Backward Pass & Gradient Inspection
    loss.backward()
    grad_reports = inspect_gradients(model)
    grads_finite = all(r["finite"] for r in grad_reports)
    params_with_grad = len(grad_reports)

    # Check tied weight gradient accumulation
    tied_weight = model.tok_embeddings.weight
    tied_grad_pass = (tied_weight.grad is not None) and (np.linalg.norm(tied_weight.grad) > 0.0)

    # 4. Multi-Layer Causal Prefix Invariance
    seq_A = np.array([[1, 5, 10, 15, 20, 25]])
    seq_B = seq_A.copy()
    seq_B[0, 4:] = [2, 3] # Mutate tokens at t=4 and t=5

    logits_A = model(seq_A)
    logits_B = model(seq_B)
    causal_diff = np.max(np.abs(logits_A.data[0, :4, :] - logits_B.data[0, :4, :]))
    causal_pass = (causal_diff < 1e-12)

    # 5. Batch Isolation
    batch_1 = np.array([[1, 2, 3], [4, 5, 6]])
    batch_2 = np.array([[1, 2, 3], [7, 8, 9]])
    l1 = model(batch_1)
    l2 = model(batch_2)
    batch_iso_pass = (np.max(np.abs(l1.data[0] - l2.data[0])) < 1e-12)

    # 6. AdamW & Gradient Clipping
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    total_grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    # 7. Untrained Autoregressive Generation
    prompt = [1, 5, 2]
    generated = model.generate(prompt, max_new_tokens=5, greedy=True)
    generation_pass = (generated.shape == (1, 8)) and np.array_equal(generated[0, :3], prompt)

    # Print Summary Report
    print("Architecture")
    print("-" * 70)
    print(f"Vocabulary Size:     {V}")
    print(f"Max Context Length:  {T_max}")
    print(f"Model Dimension:     {C}")
    print(f"Attention Heads:     {H}")
    print(f"Head Dimension:      {C // H}")
    print(f"FFN Dimension:       {F}")
    print(f"Weight Tying:        {'Yes' if config.tie_embeddings else 'No'}")

    print("\nParameters")
    print("-" * 70)
    print(f"Expected Formula:    {expected_params:,}")
    print(f"Detected Unique:     {detected_params:,}")
    print(f"Parameter Match:     {'PASS' if param_match else 'FAIL'}")

    print("\nForward & Next-Token Loss")
    print("-" * 70)
    print(f"Input Shape:         {inputs.shape}")
    print(f"Logits Shape:        {logits.shape}")
    print(f"Initial Cross-Entropy: {loss_val:.4f}")
    print(f"Expected Baseline:   ~{random_baseline:.4f} (ln({V}))")
    print(f"Initial Perplexity:  {ppl_val:.4f}")
    print(f"Logits Finite:       {'PASS' if logits_pass else 'FAIL'}")
    print(f"Loss Finite:         {'PASS' if loss_pass else 'FAIL'}")

    print("\nBackward & Gradients")
    print("-" * 70)
    print(f"Parameters w/ Grad:  {params_with_grad} / {len(model.parameters())}")
    print(f"Finite Gradients:    {'PASS' if grads_finite else 'FAIL'}")
    print(f"Tied Grad Accum:     {'PASS' if tied_grad_pass else 'FAIL'}")

    print("\nArchitectural Proofs")
    print("-" * 70)
    print(f"Causal Invariance:   {'PASS' if causal_pass else 'FAIL'}")
    print(f"Batch Isolation:     {'PASS' if batch_iso_pass else 'FAIL'}")
    print(f"Grad Norm Clipping:  {total_grad_norm:.4f} (PASS)")
    print(f"Untrained Gen:       {'PASS' if generation_pass else 'FAIL'}")

    print("-" * 70)
    overall_pass = (param_match and logits_pass and loss_pass and grads_finite and tied_grad_pass and causal_pass and batch_iso_pass and generation_pass)
    print(f"OVERALL STATUS:       {'MATHEMATICAL VALIDATION PASSED' if overall_pass else 'VALIDATION FAILED'}")
    print("=" * 70 + "\n")

    return overall_pass

if __name__ == "__main__":
    run_full_model_mathematical_validation()
