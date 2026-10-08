"""Legacy import adapter preserving the canonical module object."""
import importlib
import sys
_implementation = importlib.import_module('llm_numpy.utils.serialization')
sys.modules[__name__] = _implementation
