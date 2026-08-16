# Repository Guidelines

## Project Structure & Module Organization

This repository is a from-scratch neural-network and tiny language-model framework using Python and NumPy. Core code lives in `src/`: `tensor.py` and `ops/` implement reverse-mode autograd and math operations; `nn/` contains model layers; `losses/`, `optim/`, `data/`, `tokenization/`, `generation/`, and `training/` provide supporting functionality. Put runnable demonstrations in `examples/` and automated checks in `tests/`. `main.py` is the benchmark entry point.

## Build, Test, and Development Commands

Install the small dependency set with:

```bash
python3 -m pip install -r requirements.txt
```

Run the complete test suite with verbose output:

```bash
python3 -m pytest -v tests/
```

Run the mathematical model validation or project benchmarks with:

```bash
python3 examples/full_model_math.py
python3 main.py
```

There is no separate build system; execute scripts from the repository root so imports resolve consistently.

## Coding Style & Naming Conventions

Use clear, idiomatic Python with four-space indentation and type-aware, descriptive names. Use `snake_case` for functions, methods, and modules; `PascalCase` for classes; and uppercase names for constants. Keep numerical operations explicit and preserve NumPy array shapes and gradient semantics. Avoid adding deep-learning dependencies or hiding mathematical behavior behind opaque helpers.

## Testing Guidelines

Tests use `pytest` and are organized by component, such as `test_attention.py`, `test_adamw.py`, and `test_checkpoint_resume.py`. Add focused regression tests beside the relevant existing test module, using `test_<behavior>.py` naming. For autograd or optimizer changes, cover both forward values and gradients; run the full suite before submitting.

## Commit & Pull Request Guidelines

Git metadata is not included in this checkout, so no existing commit convention can be verified. Use concise imperative subjects with a Conventional Commit prefix, for example `fix: correct causal masking` or `test: cover weight tying`. Pull requests should explain the mathematical or behavioral change, list validation commands and results, identify compatibility risks, and include example output or screenshots when user-visible behavior changes.

## Numerical and Reproducibility Notes

Use the project’s seed utilities when adding demos or tests that rely on randomness. Preserve checkpoint and serialization compatibility, and verify shape, dtype, and gradient behavior when changing model components.
