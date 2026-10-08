"""Run at most three exact-path optimizer steps; never a pretraining job."""
from __future__ import annotations
from llm_numpy.assets import workspace_root, normalize_arguments, resolve_input_path, resolve_output_path
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from llm_numpy.backend import describe
from llm_numpy.config import LLMConfig
from llm_numpy.data import DataLoader, LanguageModelDataset
from llm_numpy.data.pipeline import open_token_shard
from llm_numpy.nn.model import TinyLLM
from llm_numpy.optim.adamw import AdamW
from llm_numpy.tokenization.bpe import ByteLevelBPETokenizer
from llm_numpy.training.checkpoint import load_checkpoint, save_checkpoint
from llm_numpy.training.config import TrainingConfig
from llm_numpy.training.trainer import Trainer
from llm_numpy.utils.seed import set_seed

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--allow-uncleared-data", action="store_true")
    args = normalize_arguments(parser.parse_args())
    if not 1 <= args.steps <= 3:
        raise SystemExit("production dry run permits only 1-3 optimizer steps")
    root = workspace_root()
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    if cfg.get("production_data_status") != "cleared" and not args.allow_uncleared_data:
        raise SystemExit("production data is not cleared; use --allow-uncleared-data only for this bounded dry run")
    manifest_path = resolve_input_path(cfg["dataset_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))
    model_cfg = LLMConfig(**cfg["model"])
    dtype = np.float16 if cfg["dtype"] == "float16" else np.float32
    model = TinyLLM(model_cfg).to(cfg["device"], dtype=dtype)
    opt_cfg = cfg["optimizer"]
    optimizer = AdamW(model.parameters(), lr=opt_cfg["learning_rate"], weight_decay=opt_cfg["weight_decay"],
                      betas=(opt_cfg["beta1"], opt_cfg["beta2"]), eps=opt_cfg["epsilon"]).to(cfg["device"], dtype=dtype)
    batch, context = int(cfg["micro_batch_size"]), int(cfg["context_length"])
    train_loader = DataLoader(LanguageModelDataset(open_token_shard(manifest, "train"), context), batch, seed=int(cfg["seed"]))
    val_loader = DataLoader(LanguageModelDataset(open_token_shard(manifest, "validation"), context), batch, shuffle=False)
    run_dir = resolve_output_path("runs/dry_runs/production_250m"); run_dir.mkdir(parents=True, exist_ok=True)
    training = TrainingConfig(batch_size=batch, micro_batch_size=batch, context_length=context,
                              learning_rate=opt_cfg["learning_rate"], weight_decay=opt_cfg["weight_decay"],
                              beta1=opt_cfg["beta1"], beta2=opt_cfg["beta2"], adam_eps=opt_cfg["epsilon"],
                              max_grad_norm=opt_cfg["gradient_clipping"], max_steps=args.steps,
                              eval_interval=args.steps, checkpoint_interval=0, seed=cfg["seed"],
                              log_csv=str(run_dir / "metrics.csv"), checkpoint_dir=str(run_dir),
                              tokenizer_path=str(resolve_input_path(cfg["tokenizer_path"])), dataset_manifest_path=str(manifest_path),
                              dataset_manifest_sha256=manifest.get("document_manifest_sha256", ""), eval_max_batches=2,
                              device=cfg["device"], dtype=cfg["dtype"], loss_scale=cfg["loss_scale"],
                              target_tokens=batch * context * args.steps, progress=True,
                              run_name="DRY RUN - NOT TRAINED MODEL")
    trainer = Trainer(model, optimizer, train_loader, val_loader, training)
    started = time.perf_counter(); initial = trainer.evaluate(); history = trainer.train(); elapsed = time.perf_counter() - started
    checkpoint = run_dir / "dry_run_checkpoint.npz"
    save_checkpoint(str(checkpoint), model, optimizer, trainer.step, trainer.epoch, trainer.tokens_processed,
                    model_cfg, training, tokenizer_path=str(resolve_input_path(cfg["tokenizer_path"])),
                    best_val_loss=trainer.best_val_loss, dataset_manifest_sha256=training.dataset_manifest_sha256)
    restored = TinyLLM(model_cfg).to(cfg["device"], dtype=dtype)
    restored_opt = AdamW(restored.parameters(), lr=opt_cfg["learning_rate"], weight_decay=opt_cfg["weight_decay"],
                         betas=(opt_cfg["beta1"], opt_cfg["beta2"]), eps=opt_cfg["epsilon"]).to(cfg["device"], dtype=dtype)
    resume = load_checkpoint(str(checkpoint), restored, restored_opt, expected_model_config=model_cfg,
                             expected_tokenizer_path=str(resolve_input_path(cfg["tokenizer_path"])),
                             expected_dataset_manifest_sha256=training.dataset_manifest_sha256)
    tokenizer = ByteLevelBPETokenizer.from_file(resolve_input_path(cfg["tokenizer_path"]))
    generated = restored.generate(np.asarray([tokenizer.encode("A computer is a machine that")]), max_new_tokens=4, greedy=True)
    result = {"label": "DRY RUN - NOT TRAINED MODEL", "backend": describe(cfg["device"]),
              "parameters": sum(p.data.size for p in restored.parameters()), "steps": trainer.step,
              "initial_loss": initial["loss"], "final_loss": history["train_loss"][-1],
              "checkpoint": str(checkpoint), "reload": resume, "generation_shape": list(generated.shape), "elapsed_seconds": elapsed}
    (run_dir / "dry_run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
