import numpy as np
from typing import Dict
from llm_numpy.nn.module import Module

def save_weights(module: Module, filepath: str) -> None:
    """
    Saves model parameters to a NumPy .npz file format.
    Tied parameters are saved under unique keys.
    """
    unique_params = module.parameters()
    weights_dict = {f"param_{i}": p.data for i, p in enumerate(unique_params)}
    np.savez(filepath, **weights_dict)

def load_weights(module: Module, filepath: str) -> None:
    """
    Loads model parameters from a NumPy .npz file format into module.
    Preserves tied weight parameter memory sharing.
    """
    data = np.load(filepath)
    unique_params = module.parameters()
    for i, p in enumerate(unique_params):
        key = f"param_{i}"
        if key in data:
            p.data[...] = data[key]
