import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.tensor import Tensor

from src.nn.module import Module
from src.nn.linear import Linear
from src.nn.sequential import Sequential
from src.losses.mse import mse_loss
from src.optim.adam import Adam
from src.utils.seed import set_seed

class XORNet(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(2, 8)
        self.l2 = Linear(8, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.l1(x).tanh()
        out = self.l2(h).sigmoid()
        return out

def run_xor():
    set_seed(42)
    print("=" * 60)
    print("NumPy Neural Network Runtime — Example 2: XOR Problem")
    print("=" * 60)

    x_data = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ], dtype=np.float64)

    y_data = np.array([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ], dtype=np.float64)

    X = Tensor(x_data)
    Y = Tensor(y_data)

    model = XORNet()
    optimizer = Adam(model.parameters(), lr=0.05)

    num_params = sum(p.data.size for p in model.parameters())
    print(f"Parameters: {num_params}")
    print(f"Optimizer: Adam (lr=0.05)")
    print("-" * 60)
    print(f"{'step':<10} {'loss':<15}")

    for step in range(1501):
        optimizer.zero_grad()
        preds = model(X)
        loss = mse_loss(preds, Y)
        loss.backward()
        optimizer.step()

        if step % 300 == 0 or step == 1500:
            print(f"{step:<10} {loss.data.item():<15.6f}")

    final_preds = model(X).data
    print("-" * 60)
    print("Predictions:")
    for i in range(4):
        print(f"[{x_data[i,0]:.0f}, {x_data[i,1]:.0f}] -> {final_preds[i,0]:.4f} (Target: {y_data[i,0]:.0f})")
    print("=" * 60 + "\n")

    return final_preds

if __name__ == "__main__":
    run_xor()
