"""Move Qdrant aliases: promote a candidate build to production, or roll back.

uv run python -m legalrag.index.alias show
uv run python -m legalrag.index.alias promote --from-alias articles_candidate --to-alias articles
uv run python -m legalrag.index.alias set --alias articles --collection articles_fb16830c7f
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from legalrag.index.store import alias_target, swap_alias
from legalrag.logging_conf import configure_logging, get_logger

log = get_logger(__name__)


def _point(client: Any, alias: str, collection: str) -> None:
    if not client.collection_exists(collection):
        raise SystemExit(f"collection {collection!r} does not exist")
    previous = swap_alias(client, alias, collection)
    log.info("alias_moved", alias=alias, collection=collection, previous=previous)
    print(json.dumps({"alias": alias, "collection": collection, "previous": previous}))  # noqa: T201


def main(argv: list[str] | None = None, client: Any = None) -> None:
    p = argparse.ArgumentParser(description="promote or roll back index aliases")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("show")
    promote = sub.add_parser("promote", help="point one alias at the collection behind another")
    promote.add_argument("--from-alias", required=True)
    promote.add_argument("--to-alias", required=True)
    pin = sub.add_parser("set", help="point an alias at a collection, e.g. to roll back")
    pin.add_argument("--alias", required=True)
    pin.add_argument("--collection", required=True)
    args = p.parse_args(argv)

    if client is None:
        from qdrant_client import QdrantClient

        from legalrag.settings import get_settings

        settings = get_settings()
        configure_logging(settings.log_level)
        client = QdrantClient(url=settings.qdrant_url, timeout=30)

    if args.command == "show":
        aliases = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
        print(json.dumps(aliases, indent=2))  # noqa: T201
    elif args.command == "promote":
        target = alias_target(client, args.from_alias)
        if target is None:
            raise SystemExit(f"alias {args.from_alias!r} does not exist")
        _point(client, args.to_alias, target)
    else:
        _point(client, args.alias, args.collection)


if __name__ == "__main__":
    main()
