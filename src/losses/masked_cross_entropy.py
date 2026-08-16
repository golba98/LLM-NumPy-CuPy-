"""Cross-entropy helpers for assistant-only supervised fine-tuning."""

from __future__ import annotations

import numpy as np

from src.tensor import Tensor
from src.losses.cross_entropy import cross_entropy_loss


def masked_cross_entropy_loss(
    logits: Tensor,
    targets: np.ndarray,
    loss_mask: np.ndarray,
) -> Tensor:
    """Average next-token loss over positions where ``loss_mask`` is nonzero."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (tokens, vocabulary)")
    target_ids = np.asarray(targets, dtype=np.int64).reshape(-1)
    mask = np.asarray(loss_mask, dtype=np.float64).reshape(-1)
    if target_ids.size != logits.shape[0] or mask.size != target_ids.size:
        raise ValueError("targets and loss_mask must align with logits")
    selected = mask > 0
    count = int(np.sum(selected))
    if count == 0:
        raise ValueError("loss_mask must select at least one token")
    return cross_entropy_loss(logits, target_ids, sample_weights=mask)
