"""Ship a built index as Qdrant snapshots, and restore it on a machine that has never built one.

uv run python -m legalrag.index.snapshot export --out snapshots/
uv run python -m legalrag.index.snapshot restore --source https://github.com/.../download/v1.0.0

Restore does nothing when the alias already exists, so it is safe to run on every start.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx

from legalrag.index.store import META_COLLECTION, alias_target, swap_alias
from legalrag.logging_conf import configure_logging, get_logger

log = get_logger(__name__)


def restore(client: Any, source: str, manifest: dict[str, Any]) -> bool:
    """Recover ``index_meta`` and the indexed collection from ``<source>/<name>.snapshot``."""
    alias, collection = manifest["alias"], manifest["collection"]
    if alias_target(client, alias):
        return False
    base = source.rstrip("/")
    for name in (META_COLLECTION, collection):
        client.recover_snapshot(name, location=f"{base}/{name}.snapshot", wait=True)
    swap_alias(client, alias, collection)
    log.info("index_restored", alias=alias, collection=collection, source=base)
    return True


def export(client: Any, qdrant_url: str, manifest: dict[str, Any], out: Path) -> list[Path]:
    """Snapshot both collections and download them as ``<out>/<name>.snapshot``."""
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for name in (META_COLLECTION, manifest["collection"]):
        snap = client.create_snapshot(name, wait=True)
        target = out / f"{name}.snapshot"
        url = f"{qdrant_url.rstrip('/')}/collections/{name}/snapshots/{snap.name}"
        with httpx.stream("GET", url, timeout=300) as r, target.open("wb") as f:
            r.raise_for_status()
            for chunk in r.iter_bytes():
                f.write(chunk)
        files.append(target)
    return files


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="export or restore the index as Qdrant snapshots")
    p.add_argument("--manifest", default="data/processed/index_manifest.json")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("export").add_argument("--out", default="snapshots")
    sub.add_parser("restore").add_argument("--source", required=True)
    args = p.parse_args(argv)

    from qdrant_client import QdrantClient

    from legalrag.settings import get_settings

    settings = get_settings()
    configure_logging(settings.log_level)
    client = QdrantClient(url=settings.qdrant_url, timeout=300)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.command == "export":
        for path in export(client, settings.qdrant_url, manifest, Path(args.out)):
            log.info("snapshot_written", path=str(path), mb=round(path.stat().st_size / 2**20, 1))
    elif not restore(client, args.source, manifest):
        log.info("index_present", alias=manifest["alias"])


if __name__ == "__main__":
    main()
