"""Print a portable NumPy/CuPy environment report."""
from __future__ import annotations
from llm_numpy.assets import workspace_root, normalize_arguments, resolve_input_path, resolve_output_path
import argparse, json, os, platform, sys
from pathlib import Path

def report() -> dict:
    result = {"os": platform.platform(), "python": sys.version.split()[0],
              "architecture": platform.machine(), "executable": sys.executable}
    try:
        import numpy as np
        result["numpy"] = np.__version__
    except Exception as exc:
        result["numpy_error"] = str(exc)
    try:
        import cupy as cp
        result["cupy"] = cp.__version__
        result["cuda_runtime"] = int(cp.cuda.runtime.runtimeGetVersion())
        count = int(cp.cuda.runtime.getDeviceCount())
        result["cuda_devices"] = []
        for index in range(count):
            props = cp.cuda.runtime.getDeviceProperties(index)
            name = props.get("name", b"unknown")
            result["cuda_devices"].append({"index": index, "name": name.decode() if isinstance(name, bytes) else str(name),
                                            "vram_bytes": int(props.get("totalGlobalMem", 0))})
    except Exception as exc:
        result["cuda_error"] = str(exc)
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = normalize_arguments(parser.parse_args())
    value = report()
    text = json.dumps(value, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
