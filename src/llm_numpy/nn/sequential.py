from typing import List
from llm_numpy.nn.module import Module
from llm_numpy.tensor import Tensor

class Sequential(Module):
    """
    Sequential container that executes submodules sequentially in order.
    """
    def __init__(self, *modules: Module):
        super().__init__()
        self.modules_list: List[Module] = list(modules)

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules_list:
            x = module(x)
        return x

    def __getitem__(self, idx: int) -> Module:
        return self.modules_list[idx]

    def __len__(self) -> int:
        return len(self.modules_list)
