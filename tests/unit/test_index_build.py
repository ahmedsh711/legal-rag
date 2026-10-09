"""Index build on in-memory Qdrant with the fake embedder."""

import pytest
from qdrant_client import QdrantClient

from legalrag.index.build import IndexParams, build_index, collection_name, load_index_params
from legalrag.index.store import alias_target, read_metadata


def test_collection_name_changes_only_with_inputs():
    p = IndexParams()
    assert collection_name("articles", "abc", p) == collection_name("articles", "abc", p)
    assert collection_name("articles", "abc", p) != collection_name("articles", "abd", p)
    assert collection_name("articles", "abc", p) != collection_name(
        "articles", "abc", IndexParams(max_length=512)
    )
    assert collection_name("articles", "abc", p) != collection_name(
        "articles", "abc", IndexParams(embedding_revision="another-commit")
    )
    assert collection_name("articles", "abc", p).startswith("articles_")


def test_build_index_creates_points_metadata_and_alias(articles, embedder):
    client = QdrantClient(":memory:")
    manifest = build_index(articles, embedder, client, IndexParams(), articles_md5="deadbeef")
    # 4 live articles x (ar + en) + 1 repealed note point
    assert manifest["points"] == 9
    assert alias_target(client, "articles") == manifest["collection"]
    meta = read_metadata(client, manifest["collection"])
    assert meta["embedding_model"] == "fake/bag-of-words" and meta["articles_md5"] == "deadbeef"
    assert meta["embedding_revision"] == embedder.revision
    assert manifest["previous_collection"] is None


def test_rebuild_with_new_data_moves_alias_and_keeps_old_collection(articles, embedder):
    client = QdrantClient(":memory:")
    first = build_index(articles, embedder, client, IndexParams(), articles_md5="v1")
    second = build_index(articles[:2], embedder, client, IndexParams(), articles_md5="v2")
    assert second["previous_collection"] == first["collection"]
    assert client.collection_exists(first["collection"])  # rollback = move the alias back


def test_rebuild_with_same_inputs_reuses_the_serving_collection(articles, embedder):
    # `dvc repro -f` must never delete and refill the collection that is answering users
    client = QdrantClient(":memory:")
    first = build_index(articles, embedder, client, IndexParams(), articles_md5="v1")
    again = build_index(articles, embedder, client, IndexParams(), articles_md5="v1")
    assert again["collection"] == first["collection"] and again["reused"]
    assert alias_target(client, "articles") == first["collection"]
    assert client.count(first["collection"]).count == first["points"]


def test_empty_corpus_never_goes_live(embedder):
    client = QdrantClient(":memory:")
    with pytest.raises(ValueError, match="no articles"):
        build_index([], embedder, client, IndexParams(), articles_md5="empty")
    assert alias_target(client, "articles") is None


def test_duplicate_article_numbers_fail_before_the_alias_moves(articles, embedder):
    # same article twice -> same point ids -> Qdrant silently overwrites -> fewer points than texts
    client = QdrantClient(":memory:")
    with pytest.raises(RuntimeError, match="points"):
        build_index([*articles, articles[0]], embedder, client, IndexParams(), articles_md5="dup")
    assert alias_target(client, "articles") is None


def test_load_index_params(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text("index:\n  max_length: 512\n", encoding="utf-8")
    assert load_index_params(p).max_length == 512
    assert load_index_params(p).embedding_model == "BAAI/bge-m3"
