"""Small dependency-free terminal dashboard with redirected-output fallback."""
from __future__ import annotations
import os, shutil, sys, time

class LiveProgress:
    def __init__(self, name: str, target_tokens: int = 0, stream=None):
        self.name, self.target_tokens = name, target_tokens
        self.stream = stream or sys.stdout
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)()) and os.environ.get("TERM") != "dumb"
        self.last_draw = 0.0

    def update(self, row: dict) -> None:
        now = time.monotonic()
        if now - self.last_draw < 0.5 and row.get("step", 0) != 1:
            return
        self.last_draw = now
        tokens = int(row.get("tokens_processed", 0))
        percent = min(100.0, 100.0 * tokens / self.target_tokens) if self.target_tokens else 0.0
        if self.interactive:
            width = max(12, min(40, shutil.get_terminal_size((100, 24)).columns - 58))
            filled = int(width * percent / 100.0)
            line = (f"\r{self.name} | [{'#' * filled}{'-' * (width-filled)}] {percent:6.2f}% "
                    f"tokens={tokens:,} step={row.get('step', 0):,} "
                    f"loss={row.get('train_loss', 0.0):.4f} val={row.get('val_loss', 0.0):.4f} "
                    f"tok/s={row.get('tokens_per_second', 0.0):,.0f}")
            cols = shutil.get_terminal_size((100, 24)).columns
            self.stream.write(line[:cols - 1].ljust(cols - 1)); self.stream.flush()
        else:
            self.stream.write(f"step={row.get('step', 0)} tokens={tokens} train_loss={row.get('train_loss', 0.0):.4f} "
                              f"val_loss={row.get('val_loss', 0.0):.4f} tok/s={row.get('tokens_per_second', 0.0):.0f}\n")
            self.stream.flush()

    def close(self) -> None:
        if self.interactive:
            self.stream.write("\n"); self.stream.flush()
