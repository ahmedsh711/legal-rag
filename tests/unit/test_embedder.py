"""bge-m3 post-processing, tested without downloading the model."""

import math

import pytest
import torch

from legalrag.index.embedder import (
    Embedding,
    l2_normalize,
    length_sorted_batches,
    sparse_from_token_weights,
)


def test_l2_normalize_gives_unit_vectors():
    v = l2_normalize(torch.tensor([[3.0, 4.0], [0.0, 2.0]]))
    assert torch.allclose(v.norm(dim=1), torch.ones(2))
    assert v[0].tolist() == pytest.approx([0.6, 0.8])


def test_sparse_keeps_max_weight_per_token_and_drops_special_tokens():
    # token ids: 0 = <s>, 2 = </s>, 1 = <pad>; 500 appears twice with different weights
    ids = [0, 500, 77, 500, 2, 1, 1]
    weights = [0.9, 0.2, 0.0, 0.7, 0.8, 0.5, 0.5]
    sparse = sparse_from_token_weights(ids, weights, unused_ids={0, 1, 2, 3})
    assert sparse == {500: pytest.approx(0.7)}  # 77 has weight 0 -> dropped


def test_embedding_exposes_qdrant_friendly_sparse():
    e = Embedding(dense=[0.6, 0.8], sparse={9: 0.5, 3: 0.25})
    indices, values = e.sparse_indices_values()
    assert sorted(zip(indices, values, strict=True)) == [(3, 0.25), (9, 0.5)]
    assert math.isclose(sum(x * x for x in e.dense), 1.0)


def test_length_sorted_batches_group_similar_lengths_and_cover_every_text_once():
    texts = ["a" * 5, "a" * 50, "a" * 1, "a" * 40, "a" * 3]
    batches = length_sorted_batches(texts, batch_size=2)
    assert batches == [[1, 3], [0, 4], [2]]  # longest first: less padding per batch
    assert sorted(i for b in batches for i in b) == list(range(len(texts)))
