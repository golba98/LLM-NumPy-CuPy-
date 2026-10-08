import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import sys
from examples.linear_regression import run_linear_regression
from examples.xor import run_xor
from examples.classifier import run_classifier
from examples.transformer_block import run_transformer_block_inspection
from examples.full_model_math import run_full_model_mathematical_validation
from examples.train_tiny_llm import run_training_demonstration

def main():
    w, b = run_linear_regression()

    xor_preds = run_xor()

    acc = run_classifier()

    params = run_transformer_block_inspection()

    model_pass = run_full_model_mathematical_validation()

    train_pass = run_training_demonstration()


if __name__ == "__main__":
    main()
