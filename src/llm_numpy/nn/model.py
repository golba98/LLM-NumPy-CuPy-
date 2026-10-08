import numpy as np
from llm_numpy.backend import array_module, to_cpu, to_device
from typing import Union, Tuple, Optional
from llm_numpy.config import LLMConfig
from llm_numpy.nn.module import Module
from llm_numpy.parameter import Parameter
from llm_numpy.tensor import Tensor, no_grad
from llm_numpy.nn.embedding import Embedding
from llm_numpy.nn.norm import RMSNorm
from llm_numpy.nn.sequential import Sequential
from llm_numpy.nn.transformer import TransformerBlock
from llm_numpy.losses.cross_entropy import cross_entropy_loss
from llm_numpy.generation.sampler import top_k_sampling, top_p_sampling

def shift_for_next_token(tokens: Union[Tensor, list, np.ndarray]) -> Tuple[Union[Tensor, np.ndarray], Union[Tensor, np.ndarray]]:
    """
    Shifts token sequence for next-token prediction alignment:
    inputs: tokens[:, :-1]
    targets: tokens[:, 1:]
    """
    if isinstance(tokens, Tensor):
        data = tokens.data
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] <= 1:
            raise ValueError("Sequence length must be > 1 to perform next-token shifting")
        return tokens[:, :-1], tokens[:, 1:]
    else:
        arr = np.array(tokens)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] <= 1:
            raise ValueError("Sequence length must be > 1 to perform next-token shifting")
        return arr[:, :-1], arr[:, 1:]

def language_model_loss(logits: Tensor, targets: Union[Tensor, np.ndarray, list]) -> Tensor:
    """
    Computes Mean Negative Log-Likelihood loss for next-token prediction:
    Reshapes logits from (B, T, V) to (B * T, V) and targets from (B, T) to (B * T,).
    """
    if logits.ndim == 3:
        B, T, V = logits.shape
        logits_2d = logits.reshape(B * T, V)
    else:
        logits_2d = logits

    if isinstance(targets, Tensor):
        targets_1d = targets.data.ravel()
    else:
        targets_1d = array_module(logits_2d.data).asarray(targets).ravel()

    return cross_entropy_loss(logits_2d, targets_1d)

def perplexity(loss: Union[Tensor, float]) -> float:
    """Computes perplexity PPL = exp(loss)."""
    loss_val = loss.data.item() if isinstance(loss, Tensor) else float(loss)
    return float(np.exp(loss_val))

class LMHead(Module):
    """
    Language Model Output Projection Head: H @ W_vocab^T
    Can use a separate weight matrix or share (tie) the embedding weight matrix.
    """
    def __init__(self, dim: int, vocab_size: int, weight: Optional[Parameter] = None, initializer_range: float = 0.02):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        if weight is not None:
            # Tied weight parameter sharing
            self.weight = weight
        else:
            weight_data = np.random.normal(0, initializer_range, size=(vocab_size, dim)).astype(np.float64)
            self.weight = Parameter(weight_data)

    def forward(self, x: Tensor) -> Tensor:
        # x is (B, T, C), self.weight is (V, C) -> x @ self.weight.T = (B, T, V)
        if x.ndim > 2:
            leading_shape = x.shape[:-1]
            return (x.reshape(-1, self.dim) @ self.weight.T).reshape(
                *leading_shape, self.vocab_size
            )
        return x @ self.weight.T

    def __repr__(self) -> str:
        return f"LMHead(dim={self.dim}, vocab_size={self.vocab_size}, tied={self.weight is not None})"

class TinyLLM(Module):
    """
    Complete Decoder-Only Language Model Stack:
    Token IDs -> Embedding -> N x TransformerBlocks -> Final RMSNorm -> LM Head -> Logits (B, T, V)
    """
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        # Token Embedding
        emb_data = np.random.normal(0, config.initializer_range, size=(config.vocab_size, config.dim)).astype(np.float64)
        self.tok_embeddings = Embedding(config.vocab_size, config.dim)
        self.tok_embeddings.weight = Parameter(emb_data)

        # Transformer Stack
        self.blocks = Sequential(*[
            TransformerBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                hidden_dim=config.hidden_dim,
                max_seq_len=config.max_seq_len,
                rope_base=config.rope_base,
                eps=config.norm_eps
            )
            for _ in range(config.num_layers)
        ])

        # Final RMSNorm
        self.final_norm = RMSNorm(config.dim, eps=config.norm_eps)

        # LM Head (optional weight tying with token embedding weight)
        self.lm_head = LMHead(
            dim=config.dim,
            vocab_size=config.vocab_size,
            weight=self.tok_embeddings.weight if config.tie_embeddings else None,
            initializer_range=config.initializer_range
        )

    def forward(self, tokens: Union[Tensor, list, np.ndarray], targets: Optional[Union[Tensor, list, np.ndarray]] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(tokens, Tensor):
            token_data = to_device(tokens.data, self.device).astype(int)
        else:
            token_data = to_device(tokens, self.device).astype(int)
        if token_data.ndim == 1:
            token_data = token_data.reshape(1, -1)
            tokens = Tensor(token_data)
        elif not isinstance(tokens, Tensor):
            tokens = Tensor(token_data)

        xp = array_module(token_data)
        if bool(xp.any(token_data < 0)) or bool(xp.any(token_data >= self.config.vocab_size)):
            raise ValueError(f"Token IDs must be in range [0, {self.config.vocab_size - 1}], but got min={to_cpu(token_data).min()}, max={to_cpu(token_data).max()}")

        B, T = token_data.shape
        if T > self.config.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}")

        # Forward pass
        x = self.tok_embeddings(tokens)
        h = self.blocks(x)
        h = self.final_norm(h)
        logits = self.lm_head(h)

        if targets is not None:
            loss = language_model_loss(logits, targets)
            return logits, loss

        return logits

    def generate(
        self,
        prompt_tokens: Union[list, np.ndarray, Tensor],
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        greedy: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Greedy / Temperature-scaled autoregressive generation for untrained validation.
        """
        if isinstance(prompt_tokens, Tensor):
            curr_tokens = prompt_tokens.data.astype(int)
        else:
            curr_tokens = np.array(prompt_tokens, dtype=int)

        if curr_tokens.ndim == 1:
            curr_tokens = curr_tokens[None, :]

        rng = np.random.default_rng(seed) if seed is not None else None
        for _ in range(max_new_tokens):
            if curr_tokens.shape[1] >= self.config.max_seq_len:
                break

            with no_grad():
                logits = self.forward(curr_tokens)
                next_logits = to_cpu(logits.data[:, -1, :]) # sampling/logging is CPU-side

                if greedy:
                    next_id = np.argmax(next_logits, axis=-1, keepdims=True)
                elif top_k is not None:
                    next_id = np.array([[top_k_sampling(next_logits[b], top_k, temperature, rng)]
                                        for b in range(curr_tokens.shape[0])])
                elif top_p is not None:
                    next_id = np.array([[top_p_sampling(next_logits[b], top_p, temperature, rng)]
                                        for b in range(curr_tokens.shape[0])])
                else:
                    if temperature <= 0:
                        raise ValueError("temperature must be > 0")
                    scaled_logits = next_logits / temperature
                    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
                    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                    next_id = np.array([[np.random.choice(self.config.vocab_size, p=probs[b])] for b in range(curr_tokens.shape[0])])

                curr_tokens = np.concatenate([curr_tokens, next_id], axis=1)

        return curr_tokens

    def __repr__(self) -> str:
        return f"TinyLLM(vocab_size={self.config.vocab_size}, dim={self.config.dim}, layers={self.config.num_layers}, heads={self.config.num_heads}, tied={self.config.tie_embeddings})"
