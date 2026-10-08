from llm_numpy.tensor import Tensor
import numpy as np
from typing import Union

class Parameter(Tensor):
    """
    Subclass of Tensor representing a trainable model parameter.
    Defaults requires_grad to True.
    """
    def __init__(self, data: Union[int, float, list, np.ndarray], requires_grad: bool = True):
        super().__init__(data, requires_grad=requires_grad, _op="Parameter")
