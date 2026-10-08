import math


def learning_rate(step: int, maximum: float, warmup_steps: int = 0,
                  decay_steps: int = 0, minimum: float = 0.0) -> float:
    if step < 0 or maximum <= 0 or minimum < 0 or minimum > maximum:
        raise ValueError("invalid learning-rate schedule arguments")
    if warmup_steps and step < warmup_steps:
        return maximum * step / warmup_steps
    if not decay_steps:
        return maximum
    progress = min(max((step - warmup_steps) / max(decay_steps, 1), 0.0), 1.0)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))
