"""Explicit historical-input resolution and canonical output locations."""
import argparse
import json
import os
from pathlib import Path


def workspace_root():
    explicit = os.environ.get("NUMPY_LLM_WORKSPACE_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    for candidate in [*Path(__file__).resolve().parents, Path.cwd(), *Path.cwd().parents]:
        spec = candidate / "pyproject.toml"
        if spec.is_file() and 'name = "codexa-numpy"' in spec.read_text():
            return candidate
    return Path.cwd().resolve()


def settings():
    path = workspace_root() / "artifacts.local.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def asset_root():
    value = os.environ.get("NUMPY_LLM_ASSET_ROOT") or settings().get("asset_root")
    return Path(value).expanduser().resolve() if value else workspace_root()


def resolve_input_path(value, *, root=None):
    path = Path(value)
    if path.is_absolute():
        for old, new in settings().get("input_aliases", {}).items():
            try:
                return Path(new) / path.relative_to(old)
            except ValueError:
                continue
        return path
    if root is not None:
        return Path(root) / path
    primary = asset_root() / path
    if primary.exists():
        return primary
    for historical in settings().get("historical_roots", []):
        candidate = Path(historical) / path
        if candidate.exists():
            return candidate
    return primary


def resolve_output_path(value):
    config = settings()
    root = Path(os.environ.get("NUMPY_LLM_OUTPUT_ROOT", config.get("output_root", "outputs")))
    if not root.is_absolute():
        root = workspace_root() / root
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    protected = [asset_root(), *[Path(p).resolve() for p in config.get("historical_roots", [])]]
    if config.get("asset_root") or os.environ.get("NUMPY_LLM_ASSET_ROOT"):
        if any(path == p or p in path.parents for p in protected):
            raise ValueError("New outputs must not overwrite historical assets")
    return path


def normalize_arguments(args, *, output_keys=()):
    inputs = {"manifest", "base_manifest", "tokenizer", "checkpoint", "prompts", "text_file", "train_jsonl", "validation_jsonl", "documents", "source", "input"}
    outputs = {"output", "output_dir", "output_root", *output_keys}
    for name, value in vars(args).items():
        if value is None or not isinstance(value, (Path, str)):
            continue
        if name in outputs:
            setattr(args, name, resolve_output_path(value))
        elif name in inputs:
            setattr(args, name, resolve_input_path(value))
    return args
