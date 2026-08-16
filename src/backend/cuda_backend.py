"""Lazy CuPy compatibility export."""

from src.backend import get_backend


def get_xp():
    return get_backend("cuda").xp


__all__ = ["get_xp"]
