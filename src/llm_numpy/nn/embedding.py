import numpy as np
from llm_numpy.backend import array_module
from typing import Union
from llm_numpy.nn.module import Module
from llm_numpy.parameter import Parameter
from llm_numpy.tensor import Tensor

class Embedding(Module):
    """
    Token Embedding Layer.
    Maps discrete token IDs of shape (B, T) to dense vectors of shape (B, T, C).
    Trainable parameter `weight` has shape (num_embeddings, embedding_dim).
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Uniform initialization in range [-0.1, 0.1]
        weight_data = np.random.uniform(-0.1, 0.1, size=(num_embeddings, embedding_dim)).astype(np.float64)
        self.weight = Parameter(weight_data)

    def forward(self, token_ids: Union[Tensor, list, np.ndarray]) -> Tensor:
        if isinstance(token_ids, Tensor):
            indices = token_ids.data.astype(int)
        else:
            indices = array_module(self.weight.data).asarray(token_ids, dtype=int)

        # Perform differentiable indexing via Tensor __getitem__
        return self.weight[indices]

    def __repr__(self) -> str:
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"
