"""Train the target NumPy model with assistant-only conversational loss."""

from __future__ import annotations
from llm_numpy.assets import workspace_root, normalize_arguments, resolve_input_path, resolve_output_path

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


from llm_numpy.model_presets import architecture
from llm_numpy.backend import array_module, describe, to_cpu, to_device
from llm_numpy.losses.masked_cross_entropy import masked_cross_entropy_loss
from llm_numpy.nn.model import TinyLLM, perplexity
from llm_numpy.optim.adamw import AdamW
from llm_numpy.optim.clip import clip_grad_norm_
from llm_numpy.tokenization import ByteLevelBPETokenizer
from llm_numpy.training.checkpoint import save_checkpoint
from llm_numpy.utils.seed import set_seed


from llm_numpy.training.checkpoint import load_model_weights as load_weights


def load_shard(manifest: dict, root: Path, split: str):
    info = manifest[split]
    count = int(info["records"])
    context = int(manifest["context_length"])
    return (
        np.memmap(root / info["inputs"], mode="r", dtype=np.uint16, shape=(count, context)),
        np.memmap(root / info["targets"], mode="r", dtype=np.uint16, shape=(count, context)),
        np.memmap(root / info["masks"], mode="r", dtype=np.uint8, shape=(count, context)),
    )


def batch(shards, indices: np.ndarray):
    inputs, targets, masks = shards
    return np.asarray(inputs[indices]), np.asarray(targets[indices]), np.asarray(masks[indices], dtype=np.float32)


def evaluate(model: TinyLLM, shards, batch_size: int, batches: int) -> float:
    total = 0.0; count = 0
    with __import__("llm_numpy.tensor", fromlist=["no_grad"]).no_grad():
        for start in range(0, min(len(shards[0]), batch_size * batches), batch_size):
            indices = np.arange(start, min(start + batch_size, len(shards[0])))
            inputs, targets, masks = batch(shards, indices)
            logits = model(to_device(inputs, model.device))
            flat_logits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2])
            loss = masked_cross_entropy_loss(flat_logits, to_cpu(to_device(targets, model.device)), to_cpu(to_device(masks, model.device)))
            selected = int(masks.sum())
            total += float(loss.data.item()) * selected
            count += selected
    return total / max(count, 1)


def generate_samples(model: TinyLLM, tokenizer: ByteLevelBPETokenizer) -> list[dict[str, str]]:
    prompts = ["Hello!", "What is 2 + 2?", "Explain RAM simply.", "What is the difference between a CPU and GPU?"]
    rows = []
    for text in prompts:
        prompt = f"User: {text}\nAssistant:"
        ids = np.asarray(tokenizer.encode(prompt), dtype=np.int64)
        output = model.generate(ids, max_new_tokens=24, greedy=True)[0]
        rows.append({"prompt": prompt, "output": tokenizer.decode(output[len(ids):])})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batches", type=int, default=16)
    args = normalize_arguments(parser.parse_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    manifest = json.loads(args.manifest.read_text())
    tokenizer = ByteLevelBPETokenizer.from_file(args.tokenizer)
    model_config = architecture("target", len(tokenizer.vocab), int(manifest["context_length"]))
    model = TinyLLM(model_config).to(args.device, dtype=np.float32 if args.dtype == "float32" else np.float64)
    load_weights(args.checkpoint, model)
    model.eval()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    train_shards = load_shard(manifest, args.manifest.parent, "train")
    validation_shards = load_shard(manifest, args.manifest.parent, "validation")
    rng = np.random.default_rng(args.seed)
    validation_before = evaluate(model, validation_shards, args.batch_size, args.eval_batches)
    print(json.dumps({"backend": describe(args.device), "train_records": len(train_shards[0]), "validation_records": len(validation_shards[0]), "validation_loss_before": validation_before}), flush=True)

    started = time.perf_counter()
    losses = []
    for step in range(1, args.steps + 1):
        indices = rng.integers(0, len(train_shards[0]), size=args.batch_size)
        inputs, targets, masks = batch(train_shards, indices)
        device_targets = to_device(targets, model.device)
        device_masks = to_device(masks, model.device)
        logits = model(to_device(inputs, model.device))
        flat_logits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2])
        loss = masked_cross_entropy_loss(flat_logits, device_targets, device_masks)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.data.item())
        losses.append(value)
        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            print(json.dumps({"step": step, "steps": args.steps, "loss": value, "perplexity": perplexity(value), "grad_norm": grad_norm, "elapsed_seconds": time.perf_counter() - started}), flush=True)

    model.eval()
    validation_after = evaluate(model, validation_shards, args.batch_size, args.eval_batches)
    checkpoint = args.output_dir / "sft_final.npz"
    save_checkpoint(str(checkpoint), model, optimizer, args.steps, 0, args.steps * args.batch_size * int(manifest["context_length"]), model_config, tokenizer_path=str(args.tokenizer))
    report = {"checkpoint": str(checkpoint), "steps": args.steps, "elapsed_seconds": time.perf_counter() - started, "initial_loss": losses[0], "final_train_loss": losses[-1], "validation_loss_before": validation_before, "validation_loss_after": validation_after, "validation_perplexity_after": perplexity(validation_after), "samples": generate_samples(model, tokenizer)}
    (args.output_dir / "sft_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
