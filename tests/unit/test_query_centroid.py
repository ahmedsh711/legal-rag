"""Query-embedding drift: the reference centroid and each question's cosine to it."""

from __future__ import annotations

import json
import math

import pytest

from legalrag.monitoring.query_centroid import QueryCentroid, load_centroid, reference_centroid

HALF = 1 / math.sqrt(2)


def test_the_centroid_is_the_mean_direction_of_the_normalised_questions():
    centroid, _ = reference_centroid([[2.0, 0.0], [0.0, 3.0]], "bge", "rev")
    assert centroid.vector == pytest.approx((HALF, HALF))
    assert centroid.cosine([1.0, 0.0]) == pytest.approx(HALF)
    assert centroid.cosine([5.0, 5.0]) == pytest.approx(1.0)  # any length, same direction


def test_reference_cosines_leave_each_question_out_of_its_own_centroid():
    # in-sample cosines would sit higher than those of new questions from the same traffic
    _, cosines = reference_centroid([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], "bge", "rev")
    assert cosines == pytest.approx([HALF, HALF, 0.0])


def test_one_question_is_not_a_reference():
    with pytest.raises(ValueError, match="two"):
        reference_centroid([[1.0, 0.0]], "bge", "rev")


def test_a_saved_centroid_loads_back_only_for_the_model_that_built_it(tmp_path):
    centroid, _ = reference_centroid([[2.0, 1.0], [0.5, 3.0]], "bge", "rev")
    path = tmp_path / "query_centroid.json"
    centroid.save(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["model"] == "bge" and saved["revision"] == "rev" and saved["dim"] == 2
    assert load_centroid(path, "bge", "rev") == centroid
    # a cosine against another model's vectors means nothing
    assert load_centroid(path, "other", "rev") is None
    assert load_centroid(path, "bge", "other-rev") is None
    assert load_centroid(tmp_path / "missing.json", "bge", "rev") is None
    assert load_centroid("", "bge", "rev") is None


def test_a_centroid_file_with_a_wrong_dim_is_rejected(tmp_path):
    path = tmp_path / "query_centroid.json"
    path.write_text(
        json.dumps({"model": "bge", "revision": None, "dim": 3, "vector": [1.0, 0.0]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dim"):
        load_centroid(path, "bge", None)


def test_the_cosine_of_a_zero_vector_is_zero():
    assert QueryCentroid("bge", None, (1.0, 0.0)).cosine([0.0, 0.0]) == 0.0
