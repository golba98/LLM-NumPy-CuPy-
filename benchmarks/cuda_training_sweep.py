"""Synchronized end-to-end CUDA throughput sweep for the current Transformer."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
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
from src.utils.seed import set_seed


def target_config(context: int) -> LLMConfig:
    return LLMConfig(
        vocab_size=256,
        max_seq_len=context,
        dim=768,
        num_layers=34,
        num_heads=12,
        hidden_dim=2048,
    )


def parse_config(value: str) -> tuple[int, int]:
    try:
        batch, context = (int(part) for part in value.split(":", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("configuration must be BATCH:CONTEXT") from exc
    if batch < 1 or context < 2:
        raise argparse.ArgumentTypeError("batch must be positive and context must be >= 2")
    return batch, context


def scalar(value) -> float:
    return float(to_cpu(value).item())


def run_configuration(batch: int, context: int, dtype_name: str, warmup: int, steps: int,
                      fused: bool = False, flatten_gemm: bool = True,
                      loss_scale: float = 1.0) -> dict:
    dtype = {"float16": np.float16, "float32": np.float32, "float64": np.float64}[dtype_name]
    set_seed(42)
    config = target_config(context)
    model = TinyLLM(config).to("cuda", dtype=dtype)
    for block in model.blocks.modules_list:
        block.attn.q_proj.flatten_gemm = flatten_gemm
        block.attn.k_proj.flatten_gemm = flatten_gemm
        block.attn.v_proj.flatten_gemm = flatten_gemm
        block.attn.out_proj.flatten_gemm = flatten_gemm
        block.ffn.w_gate.flatten_gemm = flatten_gemm
        block.ffn.w_value.flatten_gemm = flatten_gemm
        block.ffn.w_down.flatten_gemm = flatten_gemm
    model.lm_head.flatten_gemm = flatten_gemm
    if not fused:
        for block in model.blocks.modules_list:
            block.attn.fused_qkv = False
            block.ffn.fused_input = False
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.0).to("cuda", dtype=dtype)
    inputs_cpu = np.arange(batch * context, dtype=np.int64).reshape(batch, context) % config.vocab_size
    targets_cpu = np.roll(inputs_cpu, -1, axis=1)
    inputs = to_device(inputs_cpu, "cuda")
    targets = to_device(targets_cpu, "cuda")
    peak_memory = {"used_bytes": 0, "free_bytes": 0, "pool_used_bytes": 0, "pool_total_bytes": 0}

    def observe_memory() -> None:
        current = memory_stats("cuda")
        for key in peak_memory:
            if key == "free_bytes":
                peak_memory[key] = current[key] if peak_memory[key] == 0 else min(peak_memory[key], current[key])
            else:
                peak_memory[key] = max(peak_memory[key], current[key])

    def step(measure: bool = False) -> dict[str, float]:
        synchronize("cuda")
        started = time.perf_counter()
        optimizer.zero_grad()
        forward_started = time.perf_counter()
        logits, loss = model(inputs, targets=targets)
        synchronize("cuda")
        forward_ms = (time.perf_counter() - forward_started) * 1000.0
        backward_started = time.perf_counter()
        (loss * loss_scale).backward()
        if loss_scale != 1.0:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad /= loss_scale
        synchronize("cuda")
        backward_ms = (time.perf_counter() - backward_started) * 1000.0
        optimizer_started = time.perf_counter()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        synchronize("cuda")
        observe_memory()
        optimizer_ms = (time.perf_counter() - optimizer_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        return {
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "optimizer_ms": optimizer_ms,
            "total_ms": total_ms,
            "loss": scalar(loss.data),
        }

    for _ in range(warmup):
        step()
    synchronize("cuda")
    samples = [step(True) for _ in range(steps)]
    synchronize("cuda")
    memory = memory_stats("cuda")
    result = {
        "status": "ok",
        "batch": batch,
        "context": context,
        "tokens_per_step": batch * context,
        "parameters": int(sum(parameter.data.size for parameter in model.parameters())),
        "dtype": dtype_name,
        "fused_projections": fused,
        "flatten_gemm": flatten_gemm,
        "loss_scale": loss_scale,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "loss_first": samples[0]["loss"],
        "loss_last": samples[-1]["loss"],
        "timing_ms": {},
        "tokens_per_second": {},
        "memory": memory,
        "peak_memory": peak_memory,
    }
    for key in ("forward_ms", "backward_ms", "optimizer_ms", "total_ms"):
        values = [sample[key] for sample in samples]
        result["timing_ms"][key] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stddev": statistics.pstdev(values),
        }
    total_values = [sample["total_ms"] for sample in samples]
    result["tokens_per_second"] = {
        "mean_step_time": statistics.fmean(total_values),
        "mean": batch * context * 1000.0 / statistics.fmean(total_values),
        "median": batch * context * 1000.0 / statistics.median(total_values),
        "best": batch * context * 1000.0 / min(total_values),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("target",), default="target")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--loss-scale", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--fuse", action="store_true",
                        help="benchmark the opt-in fused QKV and SwiGLU input GEMMs")
    parser.add_argument("--no-flatten-gemm", action="store_true",
                        help="benchmark the original batched GEMM layout")
    parser.add_argument("--config", action="append", type=parse_config,
                        help="repeatable BATCH:CONTEXT configuration")
    parser.add_argument("--output", type=Path, default=Path("runs/benchmarks/cuda_target_sweep.json"))
    args = parser.parse_args()
    configurations = args.config or [(1, 16), (1, 32), (1, 64), (1, 128), (2, 32), (2, 64), (4, 32)]
    results = []
    print(json.dumps({"backend": describe("cuda"), "dtype": args.dtype,
                      "python": sys.version.split()[0], "numpy": np.__version__,
                      "platform": platform.platform(), "configurations": configurations}, indent=2))
    for batch, context in configurations:
        print(f"\nCONFIG batch={batch} context={context}", flush=True)
        try:
            scale = args.loss_scale if args.loss_scale is not None else (128.0 if args.dtype == "float16" else 1.0)
            result = run_configuration(batch, context, args.dtype, args.warmup, args.steps,
                                       fused=args.fuse, flatten_gemm=not args.no_flatten_gemm,
                                       loss_scale=scale)
        except (MemoryError, RuntimeError) as exc:
            message = str(exc)
            if "out of memory" not in message.lower() and "memory" not in message.lower():
                raise
            result = {"status": "OOM", "batch": batch, "context": context,
                      "tokens_per_step": batch * context, "error": message}
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    payload = {"backend": describe("cuda"), "dtype": args.dtype, "tier": args.tier,
               "fused_projections": args.fuse,
               "flatten_gemm": not args.no_flatten_gemm,
               "loss_scale": args.loss_scale if args.loss_scale is not None else (128.0 if args.dtype == "float16" else 1.0),
               "warmup_steps": args.warmup, "measured_steps": args.steps,
               "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
