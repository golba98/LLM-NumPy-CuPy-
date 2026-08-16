"""Train the NumPy model on a deliberately microscopic text corpus."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import LLMConfig
from src.data import DataLoader, LanguageModelDataset
from src.nn.model import TinyLLM, perplexity
from src.optim.adamw import AdamW
from src.optim.clip import clip_grad_norm_
from src.tokenization import ByteLevelBPETokenizer
from src.training.metrics import token_accuracy
from src.utils.seed import set_seed


def main() -> None:
    set_seed(7)
    corpus = "The cat sat. The dog sat. The cat slept. The dog slept. " * 4
    tokenizer = ByteLevelBPETokenizer()
    # Keep the byte fallback for this deliberately tiny demo; merging the
    # entire repeated corpus into one token would leave no training windows.
    tokenizer.train(corpus, vocab_size=260)
    token_ids = tokenizer.encode(corpus)
    dataset = LanguageModelDataset(token_ids, context_length=8, stride=1)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, seed=7)
    config = LLMConfig(vocab_size=len(tokenizer.vocab), max_seq_len=8, dim=32,
                       num_layers=2, num_heads=4, hidden_dim=64)
    model = TinyLLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    print(f"Tokenizer vocabulary: {len(tokenizer.vocab)}; samples: {len(dataset)}")
    for _ in range(80):
        for inputs, targets in loader:
            optimizer.zero_grad()
            logits, loss = model(inputs, targets=targets)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if optimizer.t >= 80:
            break
    inputs, targets = next(iter(loader))
    logits, loss = model(inputs, targets=targets)
    print(f"Final loss={loss.data.item():.6f} ppl={perplexity(loss):.4f} "
          f"accuracy={token_accuracy(logits, targets):.2%}")


if __name__ == "__main__":
    main()
