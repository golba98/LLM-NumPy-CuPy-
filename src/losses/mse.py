"""Legacy import adapter preserving the canonical module object."""
import importlib
import sys
_implementation = importlib.import_module('llm_numpy.losses.mse')
sys.modules[__name__] = _implementation
