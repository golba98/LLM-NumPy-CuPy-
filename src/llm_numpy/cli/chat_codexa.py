"""Interactive Codexa terminal chat for a trained NumPy checkpoint."""

from __future__ import annotations
from llm_numpy.assets import workspace_root, normalize_arguments, resolve_input_path, resolve_output_path

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = workspace_root()

from llm_numpy.model_presets import architecture
from llm_numpy.backend import describe
from llm_numpy.nn.model import TinyLLM
from llm_numpy.optim.adamw import AdamW
from llm_numpy.tokenization import ByteLevelBPETokenizer
from llm_numpy.training.checkpoint import load_model_weights
from llm_numpy.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the NumPy Codexa checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return normalize_arguments(parser.parse_args())


def load_model(args: argparse.Namespace) -> tuple[TinyLLM, ByteLevelBPETokenizer]:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"tokenizer not found: {args.tokenizer}")

    tokenizer = ByteLevelBPETokenizer.from_file(args.tokenizer)
    config = architecture("target", len(tokenizer.vocab), 128)
    model = TinyLLM(config).to(args.device, dtype=np.float32)
    load_model_weights(args.checkpoint, model, expected_model_config=config, expected_tokenizer_path=args.tokenizer)
    model.eval()
    return model, tokenizer


def make_prompt(history: list[tuple[str, str]], user_text: str) -> str:
    # This is a base language model checkpoint, not an SFT checkpoint. Keep the
    # wrapper explicit so the terminal session is usable without pretending it
    # has learned a native assistant chat format.
    lines = ["Codexa conversation:"]
    for role, text in history:
        lines.append(f"{role}: {text}")
    lines.append(f"User: {user_text}")
    lines.append("Codexa:")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    set_seed(args.seed)
    print(f"Loading NumPy Codexa checkpoint on {describe(args.device)}...", flush=True)
    model, tokenizer = load_model(args)
    print("Loaded. Type /quit to exit, /clear to reset context.", flush=True)

    history: list[tuple[str, str]] = []
    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text in {"/quit", "/exit"}:
            break
        if user_text == "/clear":
            history.clear()
            print("Context cleared.")
            continue

        prompt = make_prompt(history, user_text)
        prompt_ids = np.asarray(tokenizer.encode(prompt), dtype=int)
        # The checkpoint has a 128-token context window. Preserve the newest
        # turn when a long interactive history no longer fits.
        if len(prompt_ids) >= model.config.max_seq_len:
            prompt_ids = prompt_ids[-(model.config.max_seq_len - 1):]
        generated = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            greedy=False,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed + len(history),
        )[0]
        prompt_length = len(prompt_ids)
        response = tokenizer.decode(generated[prompt_length:]).strip()
        if not response:
            response = "(The checkpoint produced no text.)"
        print(f"Codexa> {response}")
        history.append(("User", user_text))
        history.append(("Codexa", response))


if __name__ == "__main__":
    main()
