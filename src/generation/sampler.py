"""Legacy import adapter preserving the canonical module object."""
import importlib
import sys
_implementation = importlib.import_module('llm_numpy.generation.sampler')
sys.modules[__name__] = _implementation
