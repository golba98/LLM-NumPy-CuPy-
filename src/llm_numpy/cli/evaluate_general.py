"""Run the fixed general-language generation suite before and after training."""

from __future__ import annotations
from llm_numpy.assets import workspace_root, normalize_arguments, resolve_input_path, resolve_output_path

import argparse
import json
import sys
from pathlib import Path

import numpy as np


from llm_numpy.model_presets import architecture
from llm_numpy.optim.adamw import AdamW
from llm_numpy.nn.model import TinyLLM
from llm_numpy.tokenization import ByteLevelBPETokenizer
from llm_numpy.training.checkpoint import load_checkpoint
from llm_numpy.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tier", choices=("tiny", "small", "medium", "large"), default="small")
    parser.add_argument("--prompts", type=Path, default=Path("data/evaluation/general_prompts.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = normalize_arguments(parser.parse_args())
    manifest = json.loads(args.manifest.read_text())
    suite = json.loads(args.prompts.read_text())
    tokenizer = ByteLevelBPETokenizer.from_file(resolve_input_path(manifest["tokenizer"]["path"]))
    config = architecture(args.tier, int(manifest["tokenizer"]["vocab_size"]), 64)
    set_seed(int(suite["seed"]))
    before = TinyLLM(config)
    after = TinyLLM(config)
    optimizer = AdamW(after.parameters(), lr=3e-4)
    load_checkpoint(str(args.checkpoint), after, optimizer)
    rows = []
    for prompt in suite["prompts"]:
        ids = np.asarray(tokenizer.encode(prompt), dtype=int)
        before_ids = before.generate(ids, max_new_tokens=int(suite["max_new_tokens"]), greedy=bool(suite["greedy"]))[0]
        after_ids = after.generate(ids, max_new_tokens=int(suite["max_new_tokens"]), greedy=bool(suite["greedy"]))[0]
        rows.append({
            "prompt": prompt,
            "before_training": tokenizer.decode(before_ids),
            "after_general_pretraining": tokenizer.decode(after_ids),
        })
    result = {"suite": str(args.prompts), "checkpoint": str(args.checkpoint), "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "prompts": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
