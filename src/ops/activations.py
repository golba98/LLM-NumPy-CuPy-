"""Legacy import adapter preserving the canonical module object."""
import importlib
import sys
_implementation = importlib.import_module('llm_numpy.ops.activations')
sys.modules[__name__] = _implementation
