from llm_numpy.nn.module import Module
from llm_numpy.nn.linear import Linear
from llm_numpy.tensor import Tensor
from llm_numpy.ops.basic import concat_op

class SwiGLU(Module):
    """
    SwiGLU Feed-Forward Network layer:
    SwiGLU(x) = (SiLU(x W_gate) * x W_value) W_down
    """
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False, fused_input: bool = False):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.fused_input = fused_input

        self.w_gate = Linear(dim, hidden_dim, bias=bias)
        self.w_value = Linear(dim, hidden_dim, bias=bias)
        self.w_down = Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.fused_input:
            gate_value_weight = concat_op((self.w_gate.weight, self.w_value.weight), axis=1)
            gate_value = x @ gate_value_weight
            gate_value = gate_value[:, :, :self.hidden_dim], gate_value[:, :, self.hidden_dim:]
            gate, value = gate_value
        else:
            gate, value = self.w_gate(x), self.w_value(x)
        gate = gate.silu()
        hidden = gate * value
        return self.w_down(hidden)

    def __repr__(self) -> str:
        return f"SwiGLU(dim={self.dim}, hidden_dim={self.hidden_dim})"
