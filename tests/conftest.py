import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import sys
from pathlib import Path

# Add project root directory to sys.path for pytest module resolution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

@pytest.fixture(autouse=True)
def isolate_generated_test_outputs(tmp_path, monkeypatch):
    """Default trainer logs must never overwrite an operator experiment."""
    monkeypatch.chdir(tmp_path)
