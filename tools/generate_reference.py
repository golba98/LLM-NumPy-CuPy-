import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_numpy.training.reference import generate_reference


if __name__ == "__main__":
    generate_reference(Path(__file__).resolve().parents[1] / "reference")
    print("Generated deterministic NumPy reference fixtures in reference/")
