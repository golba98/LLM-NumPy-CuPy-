"""Compatibility launcher; the implementation is packaged as llm_numpy."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_numpy.cli.production_dry_run import *
if __name__ == "__main__":
    raise SystemExit(main())
