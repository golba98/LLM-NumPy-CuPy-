# Codexa CPU port

This directory is the beginning of Milestone 4. It has no third-party
dependencies and currently implements the explicit row-major `float64`
storage/stride layer plus same-shape elementwise arithmetic and full-tensor
reduction.

Build and run its test from the repository root:

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
```

Broadcasting, matrix multiplication, and reverse-mode autograd are subsequent
port steps. The NumPy reference fixtures remain the behavioral target.
