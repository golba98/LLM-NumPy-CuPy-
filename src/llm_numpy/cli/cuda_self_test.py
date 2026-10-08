"""Fail-fast CUDA smoke test without importing CuPy on CPU-only paths."""
from __future__ import annotations
from llm_numpy.assets import workspace_root, normalize_arguments, resolve_input_path, resolve_output_path
import sys
from pathlib import Path

def main() -> int:
    print("CUDA SELF TEST")
    try:
        import cupy as cp
        from llm_numpy.config import LLMConfig
        from llm_numpy.nn.model import TinyLLM
        from llm_numpy.optim.adamw import AdamW
        from llm_numpy.backend import synchronize
        import numpy as np
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device found")
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", b"unknown")
        name = name.decode() if isinstance(name, bytes) else str(name)
        total = int(props.get("totalGlobalMem", 0))
        print(f"GPU                {name}")
        print(f"CuPy               {cp.__version__}")
        print(f"CUDA Runtime       {cp.cuda.runtime.runtimeGetVersion()}")
        print(f"VRAM               {total / 2**30:.1f} GB")
        x = cp.ones((8, 8), dtype=cp.float32)
        _ = x @ x
        synchronize("cuda")
        print("Array allocation   PASS")
        print("Matrix multiply    PASS")
        config = LLMConfig(vocab_size=64, max_seq_len=8, dim=16, num_layers=1, num_heads=4, hidden_dim=32)
        model = TinyLLM(config)
        model.to("cuda", dtype=np.float32)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        optimizer.to("cuda", dtype=np.float32)
        tokens = np.arange(16, dtype=np.int64).reshape(2, 8) % 64
        optimizer.zero_grad()
        _, loss = model(tokens, targets=tokens)
        loss.backward()
        print("Forward            PASS")
        print("Backward           PASS")
        optimizer.step()
        synchronize("cuda")
        print("AdamW              PASS")
        used = cp.get_default_memory_pool().used_bytes()
        print(f"Memory used        {used / 2**20:.1f} MiB")
        print("\nCUDA READY")
        return 0
    except Exception as exc:
        print(f"\nCUDA SELF TEST FAILED: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
