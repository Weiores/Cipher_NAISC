from __future__ import annotations

import sys
from pathlib import Path


def ensure_reasoning_path() -> None:
    """Make the shared reasoning-layer schema importable without repackaging the repo."""
    repo_root = Path(__file__).resolve().parents[2]
    reasoning_dir = repo_root / "reasoning-layer"
    reasoning_path = str(reasoning_dir)
    if reasoning_path not in sys.path:
        sys.path.insert(0, reasoning_path)


ensure_reasoning_path()
