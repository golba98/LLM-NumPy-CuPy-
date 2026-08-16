"""Conversation formatting and assistant-only causal-language-model labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.tokenization.bpe import ByteLevelBPETokenizer


@dataclass(frozen=True)
class ConversationExample:
    """Aligned causal-LM arrays for one conversation."""

    input_ids: np.ndarray
    target_ids: np.ndarray
    loss_mask: np.ndarray


def conversation_to_example(
    messages: Sequence[Mapping[str, str]],
    tokenizer: ByteLevelBPETokenizer,
) -> ConversationExample:
    """Format a conversation and mask all non-assistant target tokens.

    Role markers remain ordinary text because the current tokenizer reserves
    only the four control IDs.  EOS is trained as part of the assistant turn,
    so the model learns to terminate its response.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    full_ids = [tokenizer.BOS_ID]
    full_mask = [0]
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", ""))
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported conversation role: {role!r}")
        prefix = f"{role.capitalize()}: "
        prefix_ids = tokenizer.encode(prefix)
        full_ids.extend(prefix_ids)
        full_mask.extend([0] * len(prefix_ids))
        content_ids = tokenizer.encode(content)
        full_ids.extend(content_ids)
        full_mask.extend([1 if role == "assistant" else 0] * len(content_ids))
        newline_ids = tokenizer.encode("\n")
        full_ids.extend(newline_ids)
        full_mask.extend([1 if role == "assistant" else 0] * len(newline_ids))
    full_ids.append(tokenizer.EOS_ID)
    full_mask.append(1 if str(messages[-1].get("role", "")).lower() == "assistant" else 0)
    ids = np.asarray(full_ids, dtype=np.int64)
    mask = np.asarray(full_mask, dtype=np.float64)
    return ConversationExample(ids[:-1], ids[1:], mask[1:])


def batch_conversations(
    conversations: Iterable[Sequence[Mapping[str, str]]],
    tokenizer: ByteLevelBPETokenizer,
    pad_to: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-pad formatted examples and return inputs, targets, and masks."""
    examples = [conversation_to_example(item, tokenizer) for item in conversations]
    if not examples:
        raise ValueError("conversations must not be empty")
    length = max(example.input_ids.size for example in examples)
    if pad_to is not None:
        if pad_to < length:
            raise ValueError("pad_to is shorter than a conversation")
        length = pad_to
    inputs = np.full((len(examples), length), tokenizer.PAD_ID, dtype=np.int64)
    targets = np.full((len(examples), length), tokenizer.PAD_ID, dtype=np.int64)
    masks = np.zeros((len(examples), length), dtype=np.float64)
    for row, example in enumerate(examples):
        size = example.input_ids.size
        inputs[row, :size] = example.input_ids
        targets[row, :size] = example.target_ids
        masks[row, :size] = example.loss_mask
    return inputs, targets, masks
