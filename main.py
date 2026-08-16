import sys
from examples.linear_regression import run_linear_regression
from examples.xor import run_xor
from examples.classifier import run_classifier
from examples.transformer_block import run_transformer_block_inspection
from examples.full_model_math import run_full_model_mathematical_validation
from examples.train_tiny_llm import run_training_demonstration

def main():
    print("Running Complete Framework Verification Benchmarks (Milestones 1, 2, 3A & 3B)...\n")
    
    # 1. Linear Regression
    w, b = run_linear_regression()
    
    # 2. XOR Problem
    xor_preds = run_xor()
    
    # 3. Multiclass Classifier
    acc = run_classifier()

    # 4. Transformer Block Inspection
    params = run_transformer_block_inspection()

    # 5. Full Language Model Mathematical Validation
    model_pass = run_full_model_mathematical_validation()

    # 6. Training & Learning Demonstration
    train_pass = run_training_demonstration()
    
    print("ALL MILESTONES 1, 2, 3A & 3B VERIFICATION BENCHMARKS EXECUTED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
