"""Alias operations used to promote and roll back index builds."""

import pytest
from qdrant_client import QdrantClient

from legalrag.index.alias import main
from legalrag.index.build import IndexParams, build_index
from legalrag.index.store import alias_target


@pytest.fixture
def two_builds(articles, embedder):
    client = QdrantClient(":memory:")
    old = build_index(articles, embedder, client, IndexParams(), articles_md5="v1")
    new = build_index(
        articles[:2], embedder, client, IndexParams(), "v2", target_alias="articles_candidate"
    )
    return client, old["collection"], new["collection"]


def test_promote_points_production_at_the_candidate(two_builds, capsys):
    client, old, new = two_builds
    main(["promote", "--from-alias", "articles_candidate", "--to-alias", "articles"], client=client)
    assert alias_target(client, "articles") == new
    assert old in capsys.readouterr().out  # the previous collection is printed for rollback


def test_set_rolls_back_to_a_previous_collection(two_builds):
    client, old, new = two_builds
    main(["promote", "--from-alias", "articles_candidate", "--to-alias", "articles"], client=client)
    main(["set", "--alias", "articles", "--collection", old], client=client)
    assert alias_target(client, "articles") == old


def test_unknown_alias_or_collection_fails_without_touching_production(two_builds):
    client, old, _ = two_builds
    with pytest.raises(SystemExit):
        main(["promote", "--from-alias", "missing", "--to-alias", "articles"], client=client)
    with pytest.raises(SystemExit):
        main(["set", "--alias", "articles", "--collection", "articles_nope"], client=client)
    assert alias_target(client, "articles") == old
