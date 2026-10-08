import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from llm_numpy.tensor import Tensor

from llm_numpy.nn.linear import Linear
from llm_numpy.losses.mse import mse_loss
from llm_numpy.optim.sgd import SGD
from llm_numpy.utils.seed import set_seed

def run_linear_regression():
    set_seed(42)
    print("=" * 60)
    print("NumPy Neural Network Runtime — Example 1: Linear Regression")
    print("Target Function: y = 3x + 2")
    print("=" * 60)

    # 1. Generate synthetic dataset: y = 3.0 * x + 2.0
    x_train = np.random.uniform(-5.0, 5.0, size=(100, 1)).astype(np.float64)
    y_train = 3.0 * x_train + 2.0 + np.random.normal(0, 0.01, size=(100, 1))

    X = Tensor(x_train)
    Y = Tensor(y_train)

    # 2. Instantiate model, loss, optimizer
    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.02)

    num_params = sum(p.data.size for p in model.parameters())
    print(f"Parameters: {num_params}")
    print(f"Optimizer: SGD (lr=0.02)")
    print("-" * 60)
    print(f"{'step':<10} {'loss':<15}")

    # 3. Training Loop
    for step in range(1001):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = mse_loss(y_pred, Y)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1000:
            print(f"{step:<10} {loss.data.item():<15.6f}")

    weight_val = model.weight.data.item()
    bias_val = model.bias.data.item()

    print("-" * 60)
    print(f"Learned Weight: {weight_val:.4f} (Expected ≈ 3.0000)")
    print(f"Learned Bias:   {bias_val:.4f} (Expected ≈ 2.0000)")
    print("=" * 60 + "\n")

    return weight_val, bias_val

if __name__ == "__main__":
    run_linear_regression()
