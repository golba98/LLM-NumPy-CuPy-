"""Coarse synchronized forward component profile for the target model.

Each wrapper synchronizes around one component, so this is diagnostic rather
than a throughput benchmark. It identifies where the Python/autograd path is
spending time without changing the production model.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import llm_numpy.nn.attention as attention_module
from llm_numpy.backend import describe, synchronize, to_device
from llm_numpy.config import LLMConfig
from llm_numpy.nn.model import TinyLLM
from llm_numpy.tensor import no_grad
from llm_numpy.utils.seed import set_seed


def main() -> None:
    set_seed(42)
    config = LLMConfig(vocab_size=256, max_seq_len=128, dim=768,
                       num_layers=34, num_heads=12, hidden_dim=2048)
    model = TinyLLM(config).to("cuda", dtype=np.float16)
    for block in model.blocks.modules_list:
        for module, name in ((block.attn_norm, "rmsnorm"), (block.attn.q_proj, "q_projection"),
                             (block.attn.k_proj, "k_projection"), (block.attn.v_proj, "v_projection"),
                             (block.attn.out_proj, "attention_output"), (block.ffn_norm, "rmsnorm"),
                             (block.ffn.w_gate, "swiglu_gate"), (block.ffn.w_value, "swiglu_value"),
                             (block.ffn.w_down, "swiglu_down")):
            original = module.forward
            def timed(*args, _original=original, _name=name, **kwargs):
                synchronize("cuda")
                started = time.perf_counter()
                result = _original(*args, **kwargs)
                synchronize("cuda")
                timings[_name].append((time.perf_counter() - started) * 1000.0)
                return result
            module.forward = timed

    original_attention = attention_module.scaled_dot_product_attention
    original_softmax = attention_module.softmax
    def timed_attention(*args, **kwargs):
        synchronize("cuda")
        started = time.perf_counter()
        result = original_attention(*args, **kwargs)
        synchronize("cuda")
        timings["attention_core"].append((time.perf_counter() - started) * 1000.0)
        return result
    def timed_softmax(*args, **kwargs):
        synchronize("cuda")
        started = time.perf_counter()
        result = original_softmax(*args, **kwargs)
        synchronize("cuda")
        timings["softmax"].append((time.perf_counter() - started) * 1000.0)
        return result

    attention_module.scaled_dot_product_attention = timed_attention
    attention_module.softmax = timed_softmax
    timings = defaultdict(list)
    inputs = to_device(np.arange(4 * 128, dtype=np.int64).reshape(4, 128) % 256, "cuda")
    with no_grad():
        synchronize("cuda")
        started = time.perf_counter()
        model(inputs)
        synchronize("cuda")
        total_ms = (time.perf_counter() - started) * 1000.0
    summary = {name: {"mean_ms": float(np.mean(values)), "calls": len(values)}
               for name, values in timings.items()}
    result = {"backend": describe("cuda"), "dtype": "float16", "batch": 4, "context": 128,
              "warning": "synchronized diagnostic profile; component sums include instrumentation overhead",
              "forward_total_ms": total_ms, "components": summary}
    output = Path("runs/benchmarks/cuda_component_profile_fp16.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
