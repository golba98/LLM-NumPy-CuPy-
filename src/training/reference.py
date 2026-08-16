"""Canonical tiny NumPy reference fixtures for future runtimes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import LLMConfig
from src.nn.model import TinyLLM
from src.optim.adamw import AdamW
from src.tensor import Tensor
from src.utils.seed import set_seed


REFERENCE_CONFIG = {
    "seed": 42, "vocab_size": 32, "max_seq_len": 8, "context_length": 4,
    "num_layers": 1, "dim": 8, "num_heads": 2, "hidden_dim": 16,
    "tie_embeddings": True, "learning_rate": 1e-3, "weight_decay": 0.0,
    "dtype": "float64", "format_version": 1,
}
TOKENS = np.array([[1, 5, 8, 3], [1, 2, 7, 4]], dtype=np.int64)
TARGETS = np.array([[5, 8, 3, 1], [2, 7, 4, 6]], dtype=np.int64)


def generate_reference(directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    set_seed(REFERENCE_CONFIG["seed"])
    config = LLMConfig(vocab_size=32, max_seq_len=8, dim=8, num_layers=1,
                       num_heads=2, hidden_dim=16, tie_embeddings=True)
    model = TinyLLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    token_tensor = Tensor(TOKENS)
    embedding = model.tok_embeddings(token_tensor)
    block = model.blocks.modules_list[0]
    norm1 = block.attn_norm(embedding)
    q = block.attn.q_proj(norm1).reshape(2, 4, 2, 4).transpose(0, 2, 1, 3)
    k = block.attn.k_proj(norm1).reshape(2, 4, 2, 4).transpose(0, 2, 1, 3)
    v = block.attn.v_proj(norm1).reshape(2, 4, 2, 4).transpose(0, 2, 1, 3)
    q_rope, k_rope = block.attn.rope(q, k)
    scores = (q_rope @ k_rope.transpose(-1, -2)) / np.sqrt(4)
    causal_mask = np.triu(np.full((4, 4), -1e9), 1)
    masked_scores = scores + causal_mask
    from src.nn.attention import softmax
    probabilities = softmax(masked_scores, axis=-1)
    context = probabilities @ v
    attention_output = block.attn.out_proj(context.transpose(0, 2, 1, 3).reshape(2, 4, 8))
    post_attention = norm1 + attention_output
    norm2 = block.ffn_norm(post_attention)
    ffn_output = block.ffn(norm2)
    block_output = post_attention + ffn_output
    final_norm = model.final_norm(block_output)
    logits, loss = model(TOKENS, targets=TARGETS)
    loss.backward()
    gradients = {f"param_{index}": parameter.grad.copy() for index, parameter in enumerate(model.parameters())}
    before = {f"param_{index}": parameter.data.copy() for index, parameter in enumerate(model.parameters())}
    optimizer.step()
    after = {f"param_{index}": parameter.data.copy() for index, parameter in enumerate(model.parameters())}
    moments = {f"m_{i}_{j}": value.copy() for i, group in enumerate(optimizer.param_groups)
               for j, value in enumerate(group["m"])}
    moments.update({f"v_{i}_{j}": value.copy() for i, group in enumerate(optimizer.param_groups)
                    for j, value in enumerate(group["v"])})
    forward = {"embedding": embedding.data, "norm1": norm1.data, "q": q.data,
               "k": k.data, "v": v.data, "q_rope": q_rope.data, "k_rope": k_rope.data,
               "scores": scores.data, "masked_scores": masked_scores.data,
               "probabilities": probabilities.data, "context": context.data,
               "attention_output": attention_output.data, "post_attention": post_attention.data,
               "norm2": norm2.data, "ffn_output": ffn_output.data, "block_output": block_output.data,
               "final_norm": final_norm.data, "logits": logits.data, "loss": loss.data}
    np.savez(directory / "reference_forward.npz", **forward)
    np.savez(directory / "reference_gradients.npz", **gradients)
    np.savez(directory / "reference_optimizer_step.npz", **before, **{f"after_{k[6:]}" if k.startswith("param_") else k: v for k, v in after.items()}, **moments)
    np.save(directory / "reference_tokens.npy", TOKENS)
    np.save(directory / "reference_targets.npy", TARGETS)
    (directory / "reference_config.json").write_text(json.dumps(REFERENCE_CONFIG, indent=2) + "\n", encoding="utf-8")
    return directory
