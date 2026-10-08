import numpy as np
from llm_numpy.nn.module import Module
from llm_numpy.parameter import Parameter
from llm_numpy.tensor import Tensor

class Linear(Module):
    """
    Fully connected (Dense / Linear) neural network layer: Y = XW + b

    Initialization strategy:
    Uses Xavier (Glorot) uniform initialization where weights are sampled from
    Uniform(-limit, +limit) with limit = sqrt(6 / (fan_in + fan_out)).
    This preserves activation and gradient variance across layers during forward/backward passes.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 flatten_gemm: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.flatten_gemm = flatten_gemm

        # Xavier / Glorot uniform initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        weight_data = np.random.uniform(-limit, limit, size=(in_features, out_features)).astype(np.float64)
        self.weight = Parameter(weight_data)

        if bias:
            bias_data = np.zeros((out_features,), dtype=np.float64)
            self.bias = Parameter(bias_data)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if self.flatten_gemm and x.ndim > 2:
            leading_shape = x.shape[:-1]
            out = (x.reshape(-1, self.in_features) @ self.weight).reshape(
                *leading_shape, self.out_features
            )
        else:
            out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
