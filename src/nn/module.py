from typing import List, Dict
from src.backend import array_module, get_backend, is_cuda_array
from src.parameter import Parameter
from src.tensor import Tensor

class Module:
    """
    Base class for all neural network modules.
    Tracks parameters, submodules, and provides helper methods for optimization.
    """
    def __init__(self):
        self.training: bool = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses of Module must implement forward()")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self) -> List[Parameter]:
        """Returns a list of unique Parameter instances contained in this module and submodules."""
        params: List[Parameter] = []
        visited = set()
        
        def _collect(obj):
            if isinstance(obj, Parameter):
                if id(obj) not in visited:
                    visited.add(id(obj))
                    params.append(obj)
            elif isinstance(obj, Module):
                for value in obj.__dict__.values():
                    _collect(value)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _collect(item)
                    
        _collect(self)
        return params


    def zero_grad(self) -> None:
        """Clears gradients for all parameters in the module."""
        for p in self.parameters():
            p.zero_grad()

    @property
    def device(self) -> str:
        params = self.parameters()
        return params[0].device if params else "cpu"

    def to(self, device: str, dtype=None):
        """Move parameters and persistent array buffers to ``cpu`` or ``cuda``."""
        # Device selection is explicit on the module; do not mutate a process
        # global default, since a CUDA parity test must not change later CPU
        # tests or unrelated tensors.
        backend = get_backend(device)
        visited = set()

        def move(value):
            if isinstance(value, Parameter):
                if id(value) not in visited:
                    visited.add(id(value))
                    value.to(backend.device, dtype=dtype)
                return value
            if isinstance(value, Tensor):
                value.to(backend.device, dtype=dtype)
                return value
            if isinstance(value, Module):
                for name, child in value.__dict__.items():
                    if name != "training":
                        setattr(value, name, move(child))
                return value
            if isinstance(value, (list, tuple)):
                converted = [move(item) for item in value]
                return type(value)(converted)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                converted = backend.xp.asarray(value)
                return converted.astype(dtype) if dtype is not None else converted
            return value

        move(self)
        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def train(self, mode: bool = True) -> None:
        """Sets training mode for this module and all submodules."""
        self.training = mode
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.train(mode)

    def eval(self) -> None:
        """Sets module to evaluation mode."""
        self.train(False)

def count_parameters(module: Module) -> int:
    """Returns total number of trainable parameter elements in module."""
    return sum(p.data.size for p in module.parameters() if p.requires_grad)
