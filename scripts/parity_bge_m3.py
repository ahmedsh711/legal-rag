"""Parity check: our BgeM3Embedder vs the official FlagEmbedding implementation.

Same idea as the course's pickle-vs-ONNX parity test: two implementations that should agree,
checked on real inputs before we trust the lighter one.

    uv run --with FlagEmbedding python scripts/parity_bge_m3.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import legalrag  # noqa: F401  (OS certificate store for downloads)
from legalrag.index.build import load_index_params
from legalrag.index.embedder import BgeM3Embedder
from legalrag.index.store import article_texts
from legalrag.ingest.schema import Article

DENSE_TOL = 1e-4  # cosine distance between the two dense vectors
SPARSE_TOL = 1e-4  # max absolute difference between sparse weights


def main() -> int:
    from FlagEmbedding import BGEM3FlagModel
    from huggingface_hub import snapshot_download

    params = load_index_params("params.yaml")
    # Give FlagEmbedding a local folder at the pinned commit; otherwise it downloads the whole
    # repo (pytorch_model.bin + a 2.2 GB ONNX copy) on its own.
    local = snapshot_download(
        params.embedding_model,
        revision=params.embedding_revision,
        allow_patterns=["*.json", "*.model", "model.safetensors", "sparse_linear.pt"],
    )
    articles = [
        Article(**r)
        for r in json.loads(Path("data/processed/articles.json").read_text(encoding="utf-8"))
    ]
    live = [a for a in articles if not a.is_repealed]
    texts = [t for a in random.Random(7).sample(live, 6) for _, t in article_texts(a)]
    texts += ["ما هي مدة تقادم الالتزام؟", "Is a contract binding on the parties?"]

    ours = BgeM3Embedder(
        params.embedding_model, "cpu", params.max_length, revision=params.embedding_revision
    ).encode(texts, batch_size=4)
    ref = BGEM3FlagModel(local, use_fp16=False, devices=["cpu"]).encode(
        texts, batch_size=4, max_length=params.max_length, return_dense=True, return_sparse=True
    )
    worst_dense, worst_sparse, key_mismatch = 0.0, 0.0, 0
    for i, emb in enumerate(ours):
        ref_dense = ref["dense_vecs"][i]
        cos = sum(a * b for a, b in zip(emb.dense, ref_dense, strict=True))
        worst_dense = max(worst_dense, 1 - cos)
        ref_sparse = {int(k): float(v) for k, v in ref["lexical_weights"][i].items()}
        key_mismatch += len(set(ref_sparse) ^ set(emb.sparse))
        for k, v in ref_sparse.items():
            worst_sparse = max(worst_sparse, abs(v - emb.sparse.get(k, 0.0)))
    print(f"texts={len(texts)} worst_dense_cosine_distance={worst_dense:.2e} "
          f"worst_sparse_abs_diff={worst_sparse:.2e} sparse_key_mismatches={key_mismatch}")  # fmt: skip
    ok = worst_dense < DENSE_TOL and worst_sparse < SPARSE_TOL and key_mismatch == 0
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
