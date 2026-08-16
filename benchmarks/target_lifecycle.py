"""Bounded lifecycle validation for the exact 240.9M target model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.backend import describe, memory_stats, synchronize, to_cpu, to_device
from src.config import LLMConfig
from src.nn.model import TinyLLM
from src.optim.adamw import AdamW
from src.optim.clip import clip_grad_norm_
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.config import TrainingConfig
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--loss-scale", type=float, default=None)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("runs/benchmarks/target_lifecycle.npz"))
    args = parser.parse_args()
    dtype_name = args.dtype
    dtype = np.float16 if dtype_name == "float16" else np.float32
    loss_scale = args.loss_scale if args.loss_scale is not None else (128.0 if dtype_name == "float16" else 1.0)
    config = LLMConfig(vocab_size=256, max_seq_len=args.context, dim=768,
                       num_layers=34, num_heads=12, hidden_dim=2048)
    training_config = TrainingConfig(device="cuda", dtype=dtype_name,
                                     max_steps=args.steps, checkpoint_dir=str(args.checkpoint.parent),
                                     loss_scale=loss_scale)
    set_seed(42)
    model = TinyLLM(config).to("cuda", dtype=dtype)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.0).to("cuda", dtype=dtype)
    inputs_cpu = np.arange(args.batch * args.context, dtype=np.int64).reshape(args.batch, args.context) % 256
    targets_cpu = np.roll(inputs_cpu, -1, axis=1)
    inputs = to_device(inputs_cpu, "cuda")
    targets = to_device(targets_cpu, "cuda")
    losses = []
    memories = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        _, loss = model(inputs, targets=targets)
        (loss * loss_scale).backward()
        if loss_scale != 1.0:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad /= loss_scale
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        synchronize("cuda")
        losses.append(float(to_cpu(loss.data)))
        memories.append({"step": step, **memory_stats("cuda")})
    elapsed = time.perf_counter() - started
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(args.checkpoint), model, optimizer, args.steps, 0,
                    args.steps * args.batch * args.context, config, training_config)

    del model, optimizer
    synchronize("cuda")
    restored = TinyLLM(config).to("cuda", dtype=dtype)
    restored_optimizer = AdamW(restored.parameters(), lr=3e-4, weight_decay=0.0).to("cuda", dtype=dtype)
    resume = load_checkpoint(str(args.checkpoint), restored, restored_optimizer)
    _, resumed_loss = restored(to_device(inputs_cpu, "cuda"), targets=to_device(targets_cpu, "cuda"))
    (resumed_loss * loss_scale).backward()
    if loss_scale != 1.0:
        for parameter in restored.parameters():
            if parameter.grad is not None:
                parameter.grad /= loss_scale
    clip_grad_norm_(restored.parameters(), 1.0)
    restored_optimizer.step()
    synchronize("cuda")
    generated = restored.generate(np.arange(8, dtype=np.int64), max_new_tokens=2)
    result = {
        "backend": describe("cuda"), "dtype": dtype_name, "loss_scale": loss_scale,
        "parameters": int(sum(p.data.size for p in restored.parameters())),
        "batch": args.batch, "context": args.context, "steps": args.steps,
        "loss_first": losses[0], "loss_last": losses[-1],
        "losses_finite": bool(np.isfinite(losses).all()),
        "elapsed_seconds": elapsed, "tokens_per_second": args.steps * args.batch * args.context / elapsed,
        "checkpoint": str(args.checkpoint), "checkpoint_bytes": args.checkpoint.stat().st_size,
        "resume_tuple": resume, "resumed_loss": float(to_cpu(resumed_loss.data)),
        "generation_shape": list(generated.shape), "generation_finite": bool(np.isfinite(generated).all()),
        "memory_samples": memories, "post_reload_memory": memory_stats("cuda"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
