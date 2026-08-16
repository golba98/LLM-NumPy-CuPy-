# CUDA backend

The model, reverse-mode autograd, optimizer, checkpointing, and generation
remain project-owned. The backend only selects the array implementation:

- `cpu` uses NumPy.
- `cuda` uses CuPy and keeps model state, gradients, and AdamW state on the GPU.
- `auto` selects CUDA when CuPy and a CUDA device are available, otherwise CPU.

The CPU dependency set intentionally does not install CuPy. On a CUDA machine,
install the optional dependency with:

```bash
python3 -m pip install -r requirements-cuda.txt
```

If those runtime wheels are not used because CUDA is installed system-wide,
set `CUDA_PATH` to the toolkit root when CuPy needs to compile its first JIT
kernel.

CUDA is explicit: `--device cuda` reports an actionable error when CuPy, the
CUDA runtime, or a device is unavailable; it does not silently fall back to
NumPy. Training examples accept:

```bash
python3 examples/train_general.py --device cpu --dtype float32
python3 examples/train_general.py --device cuda --dtype float32
```

Checkpoints are serialized through CPU arrays, so a checkpoint written on CUDA
can be loaded on CPU and vice versa. Only scalar metrics and checkpoint data
are copied back during normal operation; the forward, backward, and AdamW hot
path stays on the selected backend.
