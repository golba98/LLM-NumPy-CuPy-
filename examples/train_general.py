"""Train a bounded NumPy language model from a persisted general corpus."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LLMConfig
from src.backend import describe
from src.data import DataLoader, LanguageModelDataset
from src.data.pipeline import open_token_shard
from src.nn.model import TinyLLM
from src.optim.adamw import AdamW
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.config import TrainingConfig
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def architecture(tier: str, vocab_size: int, context: int) -> LLMConfig:
    if tier == "tiny":
        return LLMConfig(vocab_size=vocab_size, max_seq_len=context, dim=128, num_layers=2, num_heads=4, hidden_dim=352)
    if tier == "small":
        return LLMConfig(vocab_size=vocab_size, max_seq_len=context, dim=320, num_layers=8, num_heads=8, hidden_dim=896)
    if tier == "medium":
        return LLMConfig(vocab_size=vocab_size, max_seq_len=context, dim=512, num_layers=14, num_heads=8, hidden_dim=1408)
    if tier == "large":
        return LLMConfig(vocab_size=vocab_size, max_seq_len=context, dim=640, num_layers=18, num_heads=10, hidden_dim=1760)
    if tier == "target":
        return LLMConfig(vocab_size=vocab_size, max_seq_len=context, dim=768, num_layers=34, num_heads=12, hidden_dim=2048)
    raise ValueError(f"unknown tier: {tier}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/tokenized/general-small/manifest.json"))
    parser.add_argument("--tier", choices=("tiny", "small", "medium", "large", "target"), default="tiny")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument(
        "--final-eval-batches", type=int,
        help="Validation batches for the final report; defaults to --eval-batches. Use 0 for a full pass.",
    )
    parser.add_argument(
        "--skip-final-checkpoint", action="store_true",
        help="Use the latest periodic checkpoint as the final artifact; useful for multi-GiB models.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--dtype", choices=("float16", "float32", "float64"), default="float64")
    parser.add_argument("--loss-scale", type=float, default=None,
                        help="static FP16 loss scale; defaults to 128 for float16 and 1 otherwise")
    parser.add_argument("--min-gradient-norm", type=float, default=0.0,
                        help="minimum acceptable gradient norm; zero disables the guard")
    parser.add_argument("--max-consecutive-small-gradient-steps", type=int, default=0,
                        help="abort after this many gradients at or below --min-gradient-norm; zero disables")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/general"))
    args = parser.parse_args()
    print(json.dumps({"backend": describe(args.device), "dtype": args.dtype}, indent=2), flush=True)
    set_seed(args.seed)
    manifest = json.loads(args.manifest.read_text())
    train_tokens = open_token_shard(manifest, "train")
    val_tokens = open_token_shard(manifest, "validation")
    train_loader = DataLoader(LanguageModelDataset(train_tokens, args.context), args.batch_size, seed=args.seed)
    val_loader = DataLoader(LanguageModelDataset(val_tokens, args.context), args.batch_size, shuffle=False)
    model_config = architecture(args.tier, int(manifest["tokenizer"]["vocab_size"]), args.context)
    model = TinyLLM(model_config)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_config = TrainingConfig(
        batch_size=args.batch_size, context_length=args.context, learning_rate=args.learning_rate,
        max_steps=args.steps, eval_interval=max(1, args.steps // 5), checkpoint_interval=max(1, args.steps // 2),
        seed=args.seed, log_csv=str(args.output_dir / f"{args.tier}.csv"), checkpoint_dir=str(args.output_dir),
        tokenizer_path=str(Path(manifest["tokenizer"]["path"])), dataset_manifest_path=str(args.manifest),
        dataset_manifest_sha256=manifest["document_manifest_sha256"],
        eval_max_batches=args.eval_batches,
        device=args.device, dtype=args.dtype,
        loss_scale=(args.loss_scale if args.loss_scale is not None else (128.0 if args.dtype == "float16" else 1.0)),
        min_gradient_norm=args.min_gradient_norm,
        max_consecutive_small_gradient_steps=args.max_consecutive_small_gradient_steps,
    )
    resumed = None
    if args.resume:
        resumed = load_checkpoint(str(args.resume), model, optimizer)
    trainer = Trainer(model, optimizer, train_loader, val_loader, training_config)
    if resumed:
        trainer.step, trainer.epoch, trainer.tokens_processed = resumed
    initial_val = trainer.evaluate()
    prompt = "The capital city of France is"
    before = model.generate(np.array([__import__("src.tokenization", fromlist=["ByteLevelBPETokenizer"]).ByteLevelBPETokenizer.from_file(manifest["tokenizer"]["path"]).encode(prompt)]), max_new_tokens=12, greedy=True)[0]
    started = time.perf_counter()
    history = trainer.train()
    elapsed = time.perf_counter() - started
    trainer.config.eval_max_batches = (
        args.eval_batches if args.final_eval_batches is None else args.final_eval_batches
    )
    final_val = trainer.evaluate()
    trainer.best_val_loss = min(trainer.best_val_loss, float(final_val["loss"]))
    if args.skip_final_checkpoint:
        final_path = args.output_dir / f"checkpoint_step_{trainer.step:06d}.npz"
        if not final_path.exists():
            raise FileNotFoundError(
                f"periodic checkpoint missing for --skip-final-checkpoint: {final_path}"
            )
    else:
        final_path = args.output_dir / f"{args.tier}_final.npz"
        save_checkpoint(str(final_path), model, optimizer, trainer.step, trainer.epoch, trainer.tokens_processed,
                        model_config, training_config, tokenizer_path=str(manifest["tokenizer"]["path"]),
                        best_val_loss=trainer.best_val_loss, dataset_manifest_sha256=manifest["document_manifest_sha256"])
    print(json.dumps({
        "tier": args.tier, "parameters": int(sum(p.data.size for p in model.parameters())),
        "layers": model_config.num_layers, "hidden_dimension": model_config.dim, "heads": model_config.num_heads,
        "context": args.context, "steps": trainer.step, "tokens_trained": trainer.tokens_processed,
        "initial_train_loss": history["train_loss"][0] if history["train_loss"] else None,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "initial_validation_loss": initial_val["loss"], "best_validation_loss": trainer.best_val_loss,
        "final_validation_loss": final_val["loss"], "perplexity": final_val["perplexity"],
        "tokens_per_second": float(np.mean(history["tokens_per_second"])) if history["tokens_per_second"] else 0.0,
        "elapsed_seconds": elapsed, "checkpoint": str(final_path), "prompt": prompt,
    }, indent=2))


if __name__ == "__main__":
    main()
