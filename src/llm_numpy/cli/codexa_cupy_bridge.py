"""JSON-lines bridge used by Codexa-dev for the NumPy/CuPy checkpoint."""

from __future__ import annotations
from llm_numpy.assets import workspace_root, normalize_arguments, resolve_input_path, resolve_output_path

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = workspace_root()

from llm_numpy.model_presets import architecture
from llm_numpy.backend import array_module
from llm_numpy.nn.model import TinyLLM
from llm_numpy.tokenization import ByteLevelBPETokenizer
from llm_numpy.utils.seed import set_seed


def load_model_weights(checkpoint: Path, model: TinyLLM) -> None:
    """Load inference weights without allocating unused AdamW state."""
    with np.load(checkpoint, allow_pickle=False) as archive:
        for index, parameter in enumerate(model.parameters()):
            xp = array_module(parameter.data)
            parameter.data[...] = xp.asarray(archive[f"model_p_{index}"], dtype=parameter.data.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = normalize_arguments(parser.parse_args())

    tokenizer = ByteLevelBPETokenizer.from_file(args.tokenizer)
    model = TinyLLM(architecture("target", len(tokenizer.vocab), 128)).to(args.device, dtype=np.float32)
    load_model_weights(args.checkpoint, model)
    model.eval()
    print(json.dumps({"type": "ready", "device": model.device, "context_length": 128}), flush=True)

    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            prompt = str(request.get("prompt", ""))
            prompt_ids = np.asarray(tokenizer.encode(prompt), dtype=int)
            prompt_ids = prompt_ids[-127:]
            output = model.generate(
                prompt_ids,
                max_new_tokens=min(int(request.get("max_new_tokens", 48)), 48),
                greedy=False,
                temperature=float(request.get("temperature", 0.85)),
                top_k=int(request.get("top_k", 40)),
                seed=int(request.get("seed", 42)),
            )[0]
            text = tokenizer.decode(output[len(prompt_ids):]).strip()
            print(json.dumps({"type": "response", "id": request.get("id"), "text": text}), flush=True)
        except Exception as error:  # bridge must report errors without dying silently
            print(json.dumps({"type": "error", "id": request.get("id") if isinstance(request, dict) else None, "message": str(error)}), flush=True)


if __name__ == "__main__":
    set_seed(42)
    main()
