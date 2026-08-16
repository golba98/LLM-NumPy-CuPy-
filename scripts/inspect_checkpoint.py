"""Inspect checkpoint metadata and checksum without constructing a model."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    expected = None
    sidecar = args.checkpoint.with_suffix(args.checkpoint.suffix + ".sha256")
    if sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").split()[0]
    with np.load(args.checkpoint, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item())) if "metadata" in archive else {}
        print(json.dumps({"path": str(args.checkpoint), "bytes": args.checkpoint.stat().st_size,
                          "checksum": digest.hexdigest(), "checksum_status":
                          "PASS" if expected is None or expected == digest.hexdigest() else "FAIL",
                          "metadata": metadata}, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
