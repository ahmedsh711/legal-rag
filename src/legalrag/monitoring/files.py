"""Small file helpers for batch jobs whose output is read while they write it."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_atomic(path: Path, text: str) -> None:
    """Write a synced temp file and rename it over ``path``, so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)
