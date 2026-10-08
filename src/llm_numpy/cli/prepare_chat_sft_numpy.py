"""Convert cleaned JSONL conversations into fixed NumPy SFT shards."""

from __future__ import annotations
from llm_numpy.assets import workspace_root, normalize_arguments, resolve_input_path, resolve_output_path

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import sys

from llm_numpy.data.sft import conversation_to_example
from llm_numpy.tokenization import ByteLevelBPETokenizer


def fitting_example(messages, tokenizer: ByteLevelBPETokenizer, context: int):
    """Keep a complete conversation when possible, otherwise a fitting final turn."""
    # Long multi-turn records are common in UltraChat. Encode the final turn
    # first so oversized histories do not spend time tokenizing megabytes that
    # will be discarded for this model's 128-token context.
    for index in range(len(messages) - 1, 0, -1):
        if str(messages[index - 1].get("role", "")).lower() != "user":
            continue
        if str(messages[index].get("role", "")).lower() != "assistant":
            continue
        user_content = str(messages[index - 1].get("content", ""))
        assistant_content = str(messages[index].get("content", ""))
        # A byte-level tokenizer cannot fit very large raw turns in 128
        # positions. Reject obvious misses before the expensive BPE pass;
        # accepted examples remain complete and are never truncated.
        if len(user_content.encode("utf-8")) + len(assistant_content.encode("utf-8")) > 1024:
            continue
        try:
            example = conversation_to_example(messages[index - 1:index + 1], tokenizer)
            if example.input_ids.size <= context:
                return example
        except (TypeError, ValueError):
            continue
    if len(messages) <= 2:
        try:
            example = conversation_to_example(messages, tokenizer)
            if example.input_ids.size <= context:
                return example
        except (TypeError, ValueError):
            pass
    return None


def iter_records(paths: list[Path], limit: int | None):
    seen = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if limit is not None and seen >= limit:
                    return
                row = json.loads(line)
                messages = row.get("messages")
                if isinstance(messages, list):
                    seen += 1
                    yield row


def convert_split(paths: list[Path], output_dir: Path, split: str, tokenizer: ByteLevelBPETokenizer,
                  context: int, limit: int | None) -> dict[str, int | str]:
    accepted = 0
    rejected = 0
    supervised_tokens = 0
    for row in iter_records(paths, limit):
        try:
            example = fitting_example(row["messages"], tokenizer, context)
            if example is None:
                raise ValueError("no fitting complete user-assistant turn")
            accepted += 1
            supervised_tokens += int(np.sum(example.loss_mask > 0))
        except (KeyError, TypeError, ValueError):
            rejected += 1

    inputs_path = output_dir / f"{split}_inputs.bin"
    targets_path = output_dir / f"{split}_targets.bin"
    masks_path = output_dir / f"{split}_masks.bin"
    inputs = np.memmap(inputs_path, mode="w+", dtype=np.uint16, shape=(accepted, context))
    targets = np.memmap(targets_path, mode="w+", dtype=np.uint16, shape=(accepted, context))
    masks = np.memmap(masks_path, mode="w+", dtype=np.uint8, shape=(accepted, context))
    inputs[:] = tokenizer.PAD_ID
    targets[:] = tokenizer.PAD_ID
    masks[:] = 0
    row_index = 0
    for row in iter_records(paths, limit):
        try:
            example = fitting_example(row["messages"], tokenizer, context)
            if example is None:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        size = example.input_ids.size
        inputs[row_index, :size] = example.input_ids
        targets[row_index, :size] = example.target_ids
        masks[row_index, :size] = (example.loss_mask > 0).astype(np.uint8)
        row_index += 1
    inputs.flush(); targets.flush(); masks.flush()
    return {
        "records": accepted,
        "rejected": rejected,
        "supervised_tokens": supervised_tokens,
        "inputs": inputs_path.name,
        "targets": targets_path.name,
        "masks": masks_path.name,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--validation-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--limit-per-split", type=int)
    args = normalize_arguments(parser.parse_args())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer.from_file(args.tokenizer)
    train = convert_split(args.train_jsonl, args.output_dir, "train", tokenizer, args.context, args.limit_per_split)
    validation = convert_split(args.validation_jsonl, args.output_dir, "validation", tokenizer, args.context, args.limit_per_split)
    manifest = {
        "format": "codexa-numpy-chat-sft-v1",
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": sha256(args.tokenizer),
        "context_length": args.context,
        "train": train,
        "validation": validation,
        "source_train": [str(path) for path in args.train_jsonl],
        "source_validation": [str(path) for path in args.validation_jsonl],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
