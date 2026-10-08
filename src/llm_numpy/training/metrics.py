import numpy as np
from llm_numpy.backend import array_module, to_cpu
from typing import Dict


def token_accuracy(logits, targets) -> float:
    values = logits.data if hasattr(logits, "data") else np.asarray(logits)
    expected = targets.data if hasattr(targets, "data") else np.asarray(targets)
    xp = array_module(values)
    return float(to_cpu(xp.mean(xp.argmax(values, axis=-1) == expected)))


def finite_or_raise(name: str, value: float, step: int) -> None:
    if not np.isfinite(value):
        raise FloatingPointError(f"{name} is not finite at step {step}: {value}")


def parameter_finiteness_or_raise(parameters, step: int) -> None:
    for index, parameter in enumerate(parameters):
        xp = array_module(parameter.data)
        if not bool(to_cpu(xp.all(xp.isfinite(parameter.data)))):
            raise FloatingPointError(f"parameter {index} is not finite at step {step}")
