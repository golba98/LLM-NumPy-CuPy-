import numpy as np
from typing import Callable, List, Tuple
from src.tensor import Tensor

def gradcheck(
    func: Callable[..., Tensor],
    inputs: List[Tensor],
    eps: float = 1e-5,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    verbose: bool = False
) -> bool:
    """
    Numerically checks analytical autograd gradients against finite differences approximation.
    
    Args:
        func: Callable taking `inputs` and returning a single Tensor.
        inputs: List of Tensors to pass to `func`.
        eps: Small perturbation for finite difference.
        atol: Absolute tolerance for maximum gradient error.
        rtol: Relative tolerance for gradient error.
        verbose: Print detailed comparison if True.
        
    Returns:
        True if all gradient comparisons pass tolerance checks, False otherwise.
    """
    # 1. Compute analytical gradients using forward + backward pass
    # First clear any existing gradients
    for inp in inputs:
        inp.zero_grad()

    # Forward pass
    out = func(*inputs)
    
    # Target loss: scalar sum of output elements
    loss = out.sum() if out.shape != () else out
    loss.backward()

    all_passed = True

    # 2. Compute numerical gradients via symmetric finite differences
    for i, inp in enumerate(inputs):
        if not inp.requires_grad:
            continue

        analytical_grad = inp.grad.copy() if inp.grad is not None else np.zeros_like(inp.data)
        numerical_grad = np.zeros_like(inp.data)

        # Iterate over every scalar element in the input tensor array
        it = np.nditer(inp.data, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            orig_val = inp.data[idx]

            # Positive step x + eps
            inp.data[idx] = orig_val + eps
            out_pos = func(*inputs)
            loss_pos = out_pos.sum().data if out_pos.shape != () else out_pos.data

            # Negative step x - eps
            inp.data[idx] = orig_val - eps
            out_neg = func(*inputs)
            loss_neg = out_neg.sum().data if out_neg.shape != () else out_neg.data

            # Reset original data value
            inp.data[idx] = orig_val

            # Symmetric finite difference formula
            num_g = (loss_pos - loss_neg) / (2.0 * eps)
            numerical_grad[idx] = num_g

            ana_g = analytical_grad[idx]
            abs_err = abs(ana_g - num_g)
            rel_err = abs_err / (max(abs(ana_g), abs(num_g), 1e-8))

            if abs_err > atol and rel_err > rtol:
                all_passed = False
                if verbose:
                    print(f"[GRADCHECK FAILED] Input {i} at {idx}: "
                          f"Analytical={ana_g:.8f}, Numerical={num_g:.8f}, "
                          f"AbsErr={abs_err:.2e}, RelErr={rel_err:.2e}")

            it.iternext()

    return all_passed
