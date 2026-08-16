"""Run a small assistant-loss-masked SFT pilot from a pretrained checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.train_general import architecture
from src.config import LLMConfig
from src.data.sft import conversation_to_example
from src.losses.masked_cross_entropy import masked_cross_entropy_loss
from src.nn.model import TinyLLM
from src.optim.adamw import AdamW
from src.optim.clip import clip_grad_norm_
from src.tokenization.bpe import ByteLevelBPETokenizer
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.config import TrainingConfig
from src.utils.seed import set_seed


CONVERSATIONS = (
    ({"role": "user", "content": "Hello!"},
     {"role": "assistant", "content": "Hello! How can I help you today?"}),
    ({"role": "user", "content": "What is 2 + 2?"},
     {"role": "assistant", "content": "2 + 2 equals 4."}),
    ({"role": "user", "content": "Explain RAM simply."},
     {"role": "assistant", "content": "RAM is short-term memory that a computer uses while programs run."}),
    ({"role": "user", "content": "What is the difference between a CPU and GPU?"},
     {"role": "assistant", "content": "A CPU handles varied sequential work, while a GPU handles many parallel operations."}),
    ({"role": "user", "content": "Write a short story about a robot."},
     {"role": "assistant", "content": "A small robot repaired a lighthouse and learned to greet every ship."}),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tier", choices=("tiny", "small", "medium"), default="small")
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sft-pilot"))
    args = parser.parse_args()
    set_seed(42)
    manifest = json.loads(args.manifest.read_text())
    tokenizer = ByteLevelBPETokenizer.from_file(manifest["tokenizer"]["path"])
    config: LLMConfig = architecture(args.tier, int(manifest["tokenizer"]["vocab_size"]), args.context)
    model = TinyLLM(config)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    load_checkpoint(str(args.checkpoint), model, optimizer)
    before_ids = np.asarray([tokenizer.encode("User: Hello!\nAssistant:")], dtype=np.int64)
    before = tokenizer.decode(model.generate(before_ids, max_new_tokens=12, greedy=True)[0].tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_config = TrainingConfig(
        batch_size=1, context_length=args.context, learning_rate=args.learning_rate,
        max_steps=args.steps, eval_interval=max(1, args.steps // 2),
        checkpoint_interval=max(1, args.steps // 2), seed=42,
        checkpoint_dir=str(args.output_dir), tokenizer_path=str(manifest["tokenizer"]["path"]),
        dataset_manifest_path=str(args.manifest),
    )
    losses = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        example = conversation_to_example(CONVERSATIONS[(step - 1) % len(CONVERSATIONS)], tokenizer)
        logits = model(example.input_ids[None, :])
        flat_logits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2])
        loss = masked_cross_entropy_loss(flat_logits, example.target_ids, example.loss_mask)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.data))
        if step % training_config.checkpoint_interval == 0:
            save_checkpoint(
                str(args.output_dir / f"sft_step_{step:06d}.npz"), model, optimizer,
                step, 0, step * int(example.target_ids.size), config, training_config,
                tokenizer_path=str(manifest["tokenizer"]["path"]),
                dataset_manifest_sha256=manifest["document_manifest_sha256"],
            )

    final_path = args.output_dir / "sft_final.npz"
    save_checkpoint(
        str(final_path), model, optimizer, args.steps, 0,
        args.steps * int(example.target_ids.size), config, training_config,
        tokenizer_path=str(manifest["tokenizer"]["path"]),
        dataset_manifest_sha256=manifest["document_manifest_sha256"],
    )
    results = []
    for user_message, _ in CONVERSATIONS:
        prompt = f"User: {user_message['content']}\nAssistant:"
        prompt_ids = np.asarray([tokenizer.encode(prompt)], dtype=np.int64)
        results.append({
            "prompt": prompt,
            "after_instruction_tuning": tokenizer.decode(
                model.generate(prompt_ids, max_new_tokens=12, greedy=True)[0].tolist()
            ),
        })
    after = results[0]["after_instruction_tuning"]
    comparison = {
        "suite": "conversation_sft_pilot",
        "checkpoint": str(final_path),
        "steps": args.steps,
        "parameters": int(sum(p.data.size for p in model.parameters())),
        "initial_loss": losses[0], "final_loss": losses[-1],
        "elapsed_seconds": time.perf_counter() - started,
        "before_general_pretraining": before,
        "after_instruction_tuning": after,
        "conversations": len(CONVERSATIONS),
        "results": results,
    }
    (args.output_dir / "conversation_generation_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
