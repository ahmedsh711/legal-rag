"""Small file helpers for batch jobs whose output is read while they write it."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_atomic(path: Path, text: str) -> None:
    """node-exporter, Grafana or a person may read the file at any moment: write a uniquely
    named temp file next to it, flush it to disk, then rename it over the old one (one step)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as tmp:  # fmt: skip
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)
