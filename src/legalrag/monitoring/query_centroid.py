"""Query-embedding drift: each question's cosine to the centroid of the reference questions.

The centroid is the mean of the L2-normalised bge-m3 dense vectors of the reference questions,
saved as JSON with the model and revision that produced it. The API records the cosine of every
question it embeds; the drift job compares those cosines with the reference ones.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from legalrag.logging_conf import get_logger
from legalrag.monitoring.files import write_atomic

log = get_logger(__name__)


@dataclass(frozen=True)
class QueryCentroid:
    model: str
    revision: str | None
    vector: tuple[float, ...]  # unit length

    def cosine(self, dense: Sequence[float]) -> float:
        norm = math.hypot(*dense)
        return math.sumprod(dense, self.vector) / norm if norm else 0.0

    def save(self, path: Path) -> None:
        body = {
            "model": self.model,
            "revision": self.revision,
            "dim": len(self.vector),
            "vector": list(self.vector),
        }
        write_atomic(path, json.dumps(body) + "\n")


def reference_centroid(
    vectors: Sequence[Sequence[float]], model: str, revision: str | None
) -> tuple[QueryCentroid, list[float]]:
    """The centroid of the reference vectors, and each one's cosine to the centroid of the others.

    Leaving a question out of its own centroid keeps the reference cosines comparable with those
    of new questions, which were never part of it."""
    if len(vectors) < 2:
        raise ValueError("a reference centroid needs at least two questions")
    x = np.asarray(vectors, float)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    total = x.sum(0)
    others = total - x
    cosines = np.einsum("ij,ij->i", x, others) / np.linalg.norm(others, axis=1)
    centroid = total / np.linalg.norm(total)
    return QueryCentroid(model, revision, tuple(centroid.tolist())), cosines.tolist()


def load_centroid(path: str | Path, model: str, revision: str | None) -> QueryCentroid | None:
    """The centroid at ``path``, or None when there is none or another model built it."""
    if not path or not Path(path).is_file():
        log.info("query_centroid_off", path=str(path))
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(raw["vector"]) != raw["dim"]:
        raise ValueError(f"{path}: dim {raw['dim']} but {len(raw['vector'])} values")
    if (raw["model"], raw["revision"]) != (model, revision):
        # a cosine against another model's vectors means nothing
        log.warning(
            "query_centroid_mismatch",
            path=str(path),
            built_with=f"{raw['model']}@{raw['revision']}",
            serving=f"{model}@{revision}",
        )
        return None
    return QueryCentroid(raw["model"], raw["revision"], tuple(raw["vector"]))
