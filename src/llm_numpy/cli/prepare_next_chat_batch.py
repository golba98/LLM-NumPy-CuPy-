"""Prepare a deterministic, multi-turn NumPy chat-SFT batch from cleaned JSONL."""

from __future__ import annotations
from llm_numpy.assets import workspace_root, normalize_arguments, resolve_input_path, resolve_output_path

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


from llm_numpy.data.sft import conversation_to_example
from llm_numpy.tokenization import ByteLevelBPETokenizer


def longest_fitting_suffix(messages, tokenizer, context: int):
    """Return the longest complete user->assistant suffix that fits."""
    if not isinstance(messages, list):
        return None
    candidates = []
    for end in range(1, len(messages)):
        if str(messages[end].get("role", "")).lower() != "assistant":
            continue
        if str(messages[end - 1].get("role", "")).lower() != "user":
            continue
        for start in range(end - 1, -1, -1):
            if str(messages[start].get("role", "")).lower() != "user":
                continue
            part = messages[start:end + 1]
            # Avoid feeding obviously oversized raw turns into the pure-Python
            # BPE implementation. UTF-8 bytes are a conservative upper bound
            # for the 128-token context used by this checkpoint.
            if sum(len(str(item.get("content", "")).encode("utf-8")) for item in part) > context * 8:
                continue
            try:
                example = conversation_to_example(part, tokenizer)
            except (TypeError, ValueError):
                continue
            if example.input_ids.size <= context:
                candidates.append((len(part), end, example))
                break
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def stable_key(row: dict, source: str) -> str:
    payload = json.dumps(row.get("messages", []), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256((source + "\0" + payload).encode("utf-8")).hexdigest()


def collect(path: Path, source: str, tokenizer, context: int, limit: int):
    source = str(source)
    candidates = []
    rejected = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            example = longest_fitting_suffix(row.get("messages"), tokenizer, context)
            if example is None:
                rejected += 1
                continue
            candidates.append((stable_key(row, source), source, example))
    candidates.sort(key=lambda item: item[0])
    return candidates[:limit], rejected, len(candidates)


def write_shard(rows, output: Path, split: str, tokenizer, context: int):
    inputs_path = output / f"{split}_inputs.bin"
    targets_path = output / f"{split}_targets.bin"
    masks_path = output / f"{split}_masks.bin"
    inputs = np.memmap(inputs_path, mode="w+", dtype=np.uint16, shape=(len(rows), context))
    targets = np.memmap(targets_path, mode="w+", dtype=np.uint16, shape=(len(rows), context))
    masks = np.memmap(masks_path, mode="w+", dtype=np.uint8, shape=(len(rows), context))
    inputs[:] = tokenizer.PAD_ID
    targets[:] = tokenizer.PAD_ID
    masks[:] = 0
    supervised_tokens = 0
    turn_counts = {}
    source_counts = {}
    for index, (_, source, example) in enumerate(rows):
        size = example.input_ids.size
        inputs[index, :size] = example.input_ids
        targets[index, :size] = example.target_ids
        masks[index, :size] = (example.loss_mask > 0).astype(np.uint8)
        supervised_tokens += int(np.sum(example.loss_mask > 0))
        turn_counts[str((size, int(np.sum(example.loss_mask > 0))))] = turn_counts.get(str((size, int(np.sum(example.loss_mask > 0)))), 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
    inputs.flush(); targets.flush(); masks.flush()
    return {
        "records": len(rows),
        "supervised_tokens": supervised_tokens,
        "source_counts": source_counts,
        "inputs": inputs_path.name,
        "targets": targets_path.name,
        "masks": masks_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train", type=Path, action="append", nargs=2, metavar=("SOURCE", "JSONL"), required=True)
    parser.add_argument("--validation", type=Path, action="append", nargs=2, metavar=("SOURCE", "JSONL"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=5000)
    parser.add_argument("--validation-per-source", type=int, default=500)
    parser.add_argument("--context", type=int, default=128)
    args = normalize_arguments(parser.parse_args())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer.from_file(args.tokenizer)

    manifest = {
        "format": "codexa-numpy-chat-sft-v2",
        "selection": "stable-sha256-per-source; longest complete fitting suffix",
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "context_length": args.context,
        "train": {},
        "validation": {},
        "rejected": {},
    }
    for split, specs, limit in (("train", args.train, args.per_source), ("validation", args.validation, args.validation_per_source)):
        rows = []
        for source, path in specs:
            source = str(source)
            selected, rejected, fitting = collect(path, source, tokenizer, args.context, limit)
            rows.extend(selected)
            manifest["rejected"][f"{split}:{source}"] = rejected
            manifest.setdefault("fitting_candidates", {})[f"{split}:{source}"] = fitting
        rows.sort(key=lambda item: item[0])
        manifest[split] = write_shard(rows, args.output_dir, split, tokenizer, args.context)
    manifest["source_train"] = [[str(source), str(path)] for source, path in args.train]
    manifest["source_validation"] = [[str(source), str(path)] for source, path in args.validation]
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
