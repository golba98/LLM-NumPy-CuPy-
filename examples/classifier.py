import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.tensor import Tensor

from src.nn.module import Module
from src.nn.linear import Linear
from src.losses.cross_entropy import cross_entropy_loss
from src.optim.adam import Adam
from src.utils.seed import set_seed

def generate_spiral_dataset(N=100, K=3):
    """Generate 3-class spiral dataset without external ML libraries."""
    X = np.zeros((N * K, 2), dtype=np.float64)
    y = np.zeros(N * K, dtype=int)
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N) # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2 # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y

class ClassifierMLP(Module):
    def __init__(self, in_dim=2, hidden_dim=16, num_classes=3):
        super().__init__()
        self.fc1 = Linear(in_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc1(x).relu()
        logits = self.fc2(h)
        return logits

def run_classifier():
    set_seed(42)
    print("=" * 60)
    print("NumPy Neural Network Runtime — Example 3: Multiclass Classifier")
    print("Dataset: 3-Class Synthetic Spiral (300 samples)")
    print("=" * 60)

    X_data, y_data = generate_spiral_dataset(N=100, K=3)
    X = Tensor(X_data)

    model = ClassifierMLP(2, 32, 3)
    optimizer = Adam(model.parameters(), lr=0.02)

    num_params = sum(p.data.size for p in model.parameters())
    print(f"Architecture: 2 -> 32 (ReLU) -> 3 (Logits)")
    print(f"Parameters: {num_params}")
    print(f"Optimizer: Adam (lr=0.02)")
    print("-" * 60)
    print(f"{'step':<10} {'loss':<15} {'accuracy':<15}")

    for step in range(501):
        optimizer.zero_grad()
        logits = model(X)
        loss = cross_entropy_loss(logits, y_data)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 500:
            preds = np.argmax(logits.data, axis=1)
            acc = np.mean(preds == y_data) * 100.0
            print(f"{step:<10} {loss.data.item():<15.6f} {acc:<15.2f}%")

    logits_final = model(X).data
    preds_final = np.argmax(logits_final, axis=1)
    final_acc = np.mean(preds_final == y_data) * 100.0

    print("-" * 60)
    print(f"Final Classification Accuracy: {final_acc:.2f}%")
    print("=" * 60 + "\n")

    return final_acc

if __name__ == "__main__":
    run_classifier()
