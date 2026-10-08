"""Deterministic CPU timing baseline for the NumPy TinyLLM."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import platform
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import argparse
import json

from llm_numpy.config import LLMConfig
from llm_numpy.backend import describe, synchronize, to_device
from llm_numpy.nn.model import TinyLLM
from llm_numpy.optim.adamw import AdamW
from llm_numpy.optim.clip import clip_grad_norm_
from llm_numpy.training.profiler import StepProfiler
from llm_numpy.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--tier", choices=("tiny", "small", "target"), default="tiny")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("runs/cuda_backend_baseline.json"))
    from llm_numpy.assets import normalize_arguments
    args = normalize_arguments(parser.parse_args())
    batch, context, vocab = args.batch, args.context, 256
    if args.tier == "tiny":
        config = LLMConfig(vocab_size=vocab, max_seq_len=context, dim=64, num_layers=2,
                           num_heads=4, hidden_dim=176)
    elif args.tier == "small":
        config = LLMConfig(vocab_size=vocab, max_seq_len=context, dim=320, num_layers=8,
                           num_heads=8, hidden_dim=896)
    else:
        config = LLMConfig(vocab_size=vocab, max_seq_len=context, dim=768, num_layers=34,
                           num_heads=12, hidden_dim=2048)
    set_seed(42)
    model = TinyLLM(config).to(args.device, dtype=np.float32 if args.dtype == "float32" else np.float64)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    inputs = np.arange(batch * context, dtype=np.int64).reshape(batch, context) % vocab
    targets = np.roll(inputs, -1, axis=1)
    profiler = StepProfiler()
    for _ in range(3):
        optimizer.zero_grad(); _, loss = model(to_device(inputs, model.device), targets=to_device(targets, model.device)); loss.backward()
        clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    for _ in range(10):
        started = time.perf_counter()
        synchronize(model.device)
        optimizer.zero_grad()
        with profiler.stage("forward"):
            logits, loss = model(to_device(inputs, model.device), targets=to_device(targets, model.device))
        with profiler.stage("backward"):
            loss.backward()
        with profiler.stage("gradient_clip"):
            clip_grad_norm_(model.parameters(), 1.0)
        with profiler.stage("adamw"):
            optimizer.step()
        synchronize(model.device)
        profiler.samples["total_step"].append((time.perf_counter() - started) * 1000.0)
    summary = profiler.summary()
    total = np.mean([x for x in profiler.samples["total_step"]])
    print("From-scratch TinyLLM Performance Baseline")
    print(f"Python={sys.version.split()[0]} NumPy={np.__version__} platform={platform.platform()}")
    print(f"parameters={sum(p.data.size for p in model.parameters())} batch={batch} context={context} tokens/step={batch*context}")
    for name in ("forward", "backward", "gradient_clip", "adamw", "total_step"):
        print(f"{name:16s} {summary[name]['mean_ms']:.3f} ms ({summary[name]['mean_ms']/total*100:.1f}%)")
    print(f"steps/sec={1000/total:.3f} tokens/sec={1000*batch*context/total:.2f}")
    environment = StepProfiler.environment()
    result = {"device": args.device, "dtype": args.dtype, "backend": describe(model.device),
              "tier": args.tier, "parameters": int(sum(p.data.size for p in model.parameters())), "batch": batch,
              "context": context, "tokens_per_step": batch * context,
              "stages_ms": {name: summary[name]["mean_ms"] for name in ("forward", "backward", "gradient_clip", "adamw", "total_step")},
              "tokens_per_second": 1000 * batch * context / total, "environment": environment}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print("environment:", environment)
    print("saved:", args.output)


if __name__ == "__main__":
    main()
