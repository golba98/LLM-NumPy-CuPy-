"""Validate production inputs before allocating a large model."""
from __future__ import annotations
from llm_numpy.assets import workspace_root, normalize_arguments, resolve_input_path, resolve_output_path
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allow-uncleared-data", action="store_true",
                        help="only for a bounded dry run; never use for production")
    args = normalize_arguments(parser.parse_args())
    root = workspace_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = resolve_input_path(config["dataset_manifest"])
    tokenizer_path = resolve_input_path(config["tokenizer_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = []
    def check(label, ok, detail=""):
        checks.append((label, bool(ok), detail))
    check("Environment", True, sys.platform)
    check("Dataset", manifest_path.exists() and "shards" in manifest, str(manifest_path))
    check("Tokenizer", tokenizer_path.exists(), str(tokenizer_path))
    vocab = int(manifest.get("tokenizer", {}).get("vocab_size", -1))
    check("Vocabulary", vocab == int(config["model"]["vocab_size"]), f"{vocab}")
    for split in ("train", "validation"):
        shard = manifest.get("shards", {}).get(split, {})
        path = resolve_input_path(shard.get("path", ""))
        ok = path.exists() and (not shard.get("sha256") or sha256(path) == shard["sha256"])
        check(f"{split} shard", ok, str(path))
    status_path = resolve_input_path("data/manifests/pretraining_sources.json")
    status = json.loads(status_path.read_text(encoding="utf-8")).get("status") if status_path.exists() else "unknown"
    check("Dataset provenance", args.allow_uncleared_data or status == "cleared", status)
    model = config["model"]
    if model.get("tie_embeddings", True):
        params = vocab * model["dim"] + model["num_layers"] * (4 * model["dim"]**2 + 3 * model["dim"] * model["hidden_dim"] + 2 * model["dim"]) + model["dim"]
    else:
        params = vocab * model["dim"] * 2 + model["num_layers"] * (4 * model["dim"]**2 + 3 * model["dim"] * model["hidden_dim"] + 2 * model["dim"]) + model["dim"]
    expected = int(config.get("expected_parameter_count", params))
    check("Model config", params == expected, f"{params:,} parameters")
    output = resolve_output_path(config.get("output_root", "runs/pretraining"))
    check("Output directory", output.parent.exists() or not output.exists(), str(output))
    free = shutil.disk_usage(output.parent if output.parent.exists() else root).free
    check("Disk", free > 6 * 2**30, f"{free / 2**30:.1f} GB free")
    print("PRETRAINING PREFLIGHT")
    for label, ok, detail in checks:
        print(f"{label:<20} {'PASS' if ok else 'FAIL':<6} {detail}")
    print(f"\n{'READY FOR DRY RUN' if all(item[1] for item in checks) else 'NOT READY'}")
    return 0 if all(item[1] for item in checks) else 1

if __name__ == "__main__":
    raise SystemExit(main())
