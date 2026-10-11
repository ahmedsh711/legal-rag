"""Shipping a built index as Qdrant snapshots and restoring it on a fresh machine."""

from types import SimpleNamespace

from legalrag.index.snapshot import restore

MANIFEST = {"collection": "articles_abc", "alias": "articles", "points": 9}


class FakeQdrant:
    def __init__(self, aliases=None):
        self.aliases = dict(aliases or {})
        self.recovered = []

    def get_aliases(self):
        names = [SimpleNamespace(alias_name=a, collection_name=c) for a, c in self.aliases.items()]
        return SimpleNamespace(aliases=names)

    def recover_snapshot(self, collection_name, location, wait=True):
        self.recovered.append((collection_name, location))

    def collection_exists(self, name):
        return any(name == c for c, _ in self.recovered)

    def update_collection_aliases(self, change_aliases_operations):
        for op in change_aliases_operations:
            if getattr(op, "create_alias", None):
                self.aliases[op.create_alias.alias_name] = op.create_alias.collection_name


def test_a_fresh_qdrant_gets_both_collections_and_the_alias():
    qdrant = FakeQdrant()
    restored = restore(qdrant, "https://example.org/index", MANIFEST)
    assert restored is True
    assert qdrant.recovered == [
        ("index_meta", "https://example.org/index/index_meta.snapshot"),
        ("articles_abc", "https://example.org/index/articles_abc.snapshot"),
    ]
    assert qdrant.aliases == {"articles": "articles_abc"}


def test_an_existing_index_is_left_alone():
    qdrant = FakeQdrant(aliases={"articles": "articles_older"})
    assert restore(qdrant, "https://example.org/index", MANIFEST) is False
    assert qdrant.recovered == [] and qdrant.aliases == {"articles": "articles_older"}
