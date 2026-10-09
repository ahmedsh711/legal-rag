"""Build the Qdrant index from articles.json (a DVC stage).

Steps: read articles -> texts per language -> bge-m3 dense + sparse -> new collection
``articles_<hash>`` -> metadata -> move the alias ``articles`` to it -> write a manifest.

The collection name is a hash of everything that changes the vectors (articles.json, embedding
model and revision, max_length, normalization version), so the same inputs always give the same collection
and a changed input never overwrites what is being served.

    uv run python -m legalrag.index.build            # uses params.yaml -> index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from legalrag.index.store import (
    TEXT_FORMAT_VERSION,
    alias_target,
    article_texts,
    create_collection,
    read_metadata,
    swap_alias,
    upsert_points,
    write_metadata,
)
from legalrag.ingest.normalize import NORMALIZATION_VERSION
from legalrag.ingest.params import CorpusInputError
from legalrag.ingest.schema import Article
from legalrag.ingest.validate import load_articles
from legalrag.logging_conf import configure_logging, get_logger

log = get_logger(__name__)


class IndexParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    embedding_model: str = "BAAI/bge-m3"
    embedding_revision: str = "9a0624b896d81da7492a910ffa53731274b6cf3d"  # pragma: allowlist secret
    max_length: int = 1024
    batch_size: int = 16
    alias: str = "articles"


def load_index_params(path: str | Path) -> IndexParams:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return IndexParams(**data.get("index", {}))


def collection_name(alias: str, articles_md5: str, params: IndexParams) -> str:
    key = "|".join(
        [articles_md5, params.embedding_model, params.embedding_revision,
         str(params.max_length), NORMALIZATION_VERSION, TEXT_FORMAT_VERSION]
    )  # fmt: skip
    return f"{alias}_{hashlib.sha1(key.encode()).hexdigest()[:10]}"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_index(
    articles: list[Article], embedder: Any, client: Any, params: IndexParams, articles_md5: str
) -> dict[str, Any]:
    items = [(a, lang) for a in articles for lang, _ in article_texts(a)]
    if not items:
        raise ValueError("no articles to index; refusing to publish an empty index")
    name = collection_name(params.alias, articles_md5, params)
    if done := finished_build(client, name):
        # Same inputs as a finished build: reuse it. Deleting and refilling it would take the
        # serving index down for the whole embedding run. (embedder is not needed here.)
        previous = swap_alias(client, params.alias, name)
        log.info("index_reused", collection=name, previous=previous)
        return {**done, "collection": name, "alias": params.alias, "previous_collection": previous,
                "embed_seconds": 0.0, "reused": True}  # fmt: skip
    t0 = time.perf_counter()
    embeddings = embedder.encode([text for _, text in _texts(articles)], params.batch_size)
    embed_s = time.perf_counter() - t0
    create_collection(client, name, embedder.dim)  # a leftover without metadata is replaced
    points = upsert_points(client, name, items, embeddings)
    if (stored := client.count(name, exact=True).count) != points:  # same id twice = overwrite
        raise RuntimeError(f"{points} points sent, {stored} stored: duplicate article numbers?")
    metadata = {
        "embedding_model": embedder.model_name,
        "embedding_revision": embedder.revision,
        "dim": embedder.dim,
        "max_length": params.max_length,
        "normalization_version": NORMALIZATION_VERSION,
        "text_format_version": TEXT_FORMAT_VERSION,
        "articles_md5": articles_md5,
        "articles": len(articles),
        "points": points,
        "git_sha": git_sha(),
    }
    write_metadata(client, name, metadata)
    previous = swap_alias(client, params.alias, name)
    log.info(
        "index_built",
        collection=name,
        previous=previous,
        points=points,
        embed_seconds=round(embed_s, 1),
    )
    return {**metadata, "collection": name, "alias": params.alias, "previous_collection": previous,
            "embed_seconds": round(embed_s, 1), "reused": False}  # fmt: skip


def finished_build(client: Any, name: str) -> dict[str, Any] | None:
    """Metadata of a completed build of this exact collection, else None (metadata is written
    last, so a crashed build has none)."""
    return read_metadata(client, name) if client.collection_exists(name) else None


def _texts(articles: list[Article]) -> list[tuple[str, str]]:
    return [t for a in articles for t in article_texts(a)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Embed articles.json into Qdrant")
    parser.add_argument("--articles", default=None, help="default: settings.articles_path")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--manifest", default="data/processed/index_manifest.json")
    parser.add_argument("--metrics", default="reports/index_metrics.json")
    args = parser.parse_args(argv)

    from qdrant_client import QdrantClient

    from legalrag.index.embedder import BgeM3Embedder
    from legalrag.settings import get_settings

    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        params = load_index_params(args.params)
        path = Path(args.articles or settings.articles_path)
        articles = load_articles(path)
    except (CorpusInputError, OSError, ValueError) as exc:
        log.error("index_input_error", detail=str(exc))
        sys.exit(1)

    client = QdrantClient(url=settings.qdrant_url, timeout=60)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    # DVC runs this stage every time (Qdrant is state DVC cannot see); when the index is already
    # there it is reused in seconds, so only load the 2.3 GB model when we really embed.
    needed = not finished_build(client, collection_name(params.alias, md5, params))
    embedder = (
        BgeM3Embedder(
            params.embedding_model,
            settings.embedding_device,
            params.max_length,
            revision=params.embedding_revision,
        )
        if needed
        else None
    )
    manifest = build_index(articles, embedder, client, params, md5)
    if alias_target(client, params.alias) != manifest["collection"]:  # not assert: -O removes it
        raise RuntimeError(f"alias {params.alias!r} does not point at {manifest['collection']}")

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    stable = {
        k: v
        for k, v in manifest.items()
        if k not in ("embed_seconds", "previous_collection", "git_sha", "reused")
    }
    # deterministic -> DVC out; newline="\n" so the bytes (and DVC's hash) are the same on every
    # OS, and a final newline so the end-of-file pre-commit hook never rewrites it
    Path(args.manifest).write_text(
        json.dumps(stable, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    metrics = {k: manifest[k] for k in ("points", "embed_seconds", "reused")}  # of THIS run
    Path(args.metrics).write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
