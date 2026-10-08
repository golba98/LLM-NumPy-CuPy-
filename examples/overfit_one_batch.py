"""Demonstrate that the complete custom model can memorize one fixed batch."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from llm_numpy.config import LLMConfig
from llm_numpy.nn.model import TinyLLM, perplexity
from llm_numpy.optim.adamw import AdamW
from llm_numpy.optim.clip import clip_grad_norm_
from llm_numpy.training.metrics import token_accuracy
from llm_numpy.utils.seed import set_seed


def main() -> None:
    set_seed(42)
    config = LLMConfig(vocab_size=16, max_seq_len=16, dim=32, num_layers=2,
                       num_heads=4, hidden_dim=64)
    model = TinyLLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    inputs = np.array([[1, 5, 2, 8, 3, 10], [4, 9, 0, 7, 11, 2]])
    targets = np.array([[5, 2, 8, 3, 10, 14], [9, 0, 7, 11, 2, 6]])

    def report() -> tuple[float, float, float]:
        logits, loss = model(inputs, targets=targets)
        return loss.data.item(), perplexity(loss), token_accuracy(logits, targets)

    initial = report()
    print("NumPy TinyLLM - One-Batch Overfit Test")
    print(f"Initial: loss={initial[0]:.6f} ppl={initial[1]:.4f} accuracy={initial[2]:.2%}")
    for step in range(1, 121):
        optimizer.zero_grad()
        _, loss = model(inputs, targets=targets)
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in (1, 25, 50, 100, 120):
            current = report()
            print(f"step={step:3d} loss={current[0]:.6f} ppl={current[1]:.4f} "
                  f"accuracy={current[2]:.2%} grad_norm={grad_norm:.4f}")
    final = report()
    print(f"Final: loss={final[0]:.6f} ppl={final[1]:.4f} accuracy={final[2]:.2%}")
    print("ONE-BATCH OVERFIT: PASS" if final[0] < initial[0] and final[2] > initial[2]
          else "ONE-BATCH OVERFIT: FAIL")


if __name__ == "__main__":
    main()
