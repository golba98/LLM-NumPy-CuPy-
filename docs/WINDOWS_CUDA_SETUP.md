# Windows CUDA setup

This project owns the tensor/autograd, Transformer, backward pass, optimizer,
tokenizer, and training loop. CuPy is used only for CUDA array operations.

## Supported baseline

- Windows 10/11, 64-bit
- Python 3.11+ (use the same minor version for the environment and scripts)
- NumPy 2.x and pytest 8.x (see `requirements.txt`)
- CuPy CUDA 12.x (`cupy-cuda12x`, see `requirements-cuda.txt`)
- NVIDIA driver compatible with the installed CUDA 12 runtime
- Tested target GPU: GeForce RTX 4080, 16 GB VRAM

Install from PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-cuda.txt
```

Verify the environment:

```powershell
python scripts/environment_report.py --output environment.txt
python scripts/cuda_self_test.py
python -m pytest -q tests
```

No Linux `CUDA_PATH=/home/...` assignment is required on Windows. If the
NVIDIA driver or a system CUDA installation is unusual, follow CuPy's package
compatibility guidance and keep the chosen package version recorded in the
environment report.
