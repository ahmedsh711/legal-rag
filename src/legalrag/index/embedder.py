"""bge-m3 dense + sparse embeddings with plain ``transformers`` (no FlagEmbedding dependency).

bge-m3 gives two signals from one forward pass over the text:
- **dense**: the hidden state of the first token ([CLS]), L2-normalized -> meaning;
- **sparse**: ``relu(sparse_linear(hidden_state))`` for every token -> how much each word
  matters; we keep the highest weight per token id and drop special tokens -> keywords.

This is the same computation as FlagEmbedding's ``BGEM3FlagModel``; ``scripts/parity_bge_m3.py``
checks the two agree. Using it directly avoids six heavy libraries in the API image.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from legalrag.logging_conf import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Embedding:
    dense: list[float]
    sparse: dict[int, float]  # token id -> weight

    def sparse_indices_values(self) -> tuple[list[int], list[float]]:
        """Qdrant's SparseVector wants two parallel lists."""
        return list(self.sparse), list(self.sparse.values())


def l2_normalize(vectors: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vectors, p=2, dim=-1)


def sparse_from_token_weights(
    token_ids: Iterable[int], weights: Iterable[float], unused_ids: set[int]
) -> dict[int, float]:
    """Max weight per token id, ignoring special tokens and zero weights (bge-m3's rule)."""
    out: dict[int, float] = {}
    for tid, w in zip(token_ids, weights, strict=True):
        if tid in unused_ids or w <= 0:
            continue
        if w > out.get(tid, 0.0):
            out[tid] = float(w)
    return out


def length_sorted_batches(texts: Sequence[str], batch_size: int) -> list[list[int]]:
    """Text indices grouped longest-first, so each batch pads to a similar length.

    A batch is padded to its longest text; mixing a 600-token article with 40-token ones wastes
    most of the compute on padding. FlagEmbedding sorts the same way."""
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


class BgeM3Embedder:
    """Loads bge-m3 once and turns texts into ``Embedding`` objects (CPU by default)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        max_length: int = 1024,
        revision: str | None = None,
    ):
        from huggingface_hub import hf_hub_download  # heavy imports kept local
        from transformers import AutoModel, AutoTokenizer

        self.model_name, self.device, self.max_length = model_name, device, max_length
        self.revision = revision  # a commit sha: the same weights on every machine, every rebuild
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
        self.sparse_linear = torch.nn.Linear(self.model.config.hidden_size, 1)
        state = torch.load(
            hf_hub_download(model_name, "sparse_linear.pt", revision=revision),
            map_location=device,
            weights_only=True,
        )  # weights_only: never execute code from a downloaded pickle
        self.sparse_linear.load_state_dict(state)
        self.sparse_linear.to(device).eval()
        tok = self.tokenizer
        self.unused_ids = {
            i
            for i in (tok.cls_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id)
            if i is not None
        }
        self.dim = self.model.config.hidden_size
        # The fast tokenizer changes its own padding/truncation state on every call, so two API
        # requests in worker threads can collide ("Already borrowed"). One encode at a time; on
        # CPU the model would not run faster in parallel anyway.
        # ponytail: one global lock; a pool of embedders if query throughput ever matters
        self._lock = threading.Lock()
        log.info(
            "embedder_loaded", model=model_name, revision=revision, device=device, dim=self.dim
        )

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> list[Embedding]:
        with self._lock:
            return self._encode(texts, batch_size)

    @torch.inference_mode()
    def _encode(self, texts: Sequence[str], batch_size: int) -> list[Embedding]:
        out: dict[int, Embedding] = {}
        for idx in length_sorted_batches(texts, batch_size):
            enc = self.tokenizer(
                [texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state  # [batch, tokens, hidden]
            dense = l2_normalize(hidden[:, 0]).cpu()
            weights = torch.relu(self.sparse_linear(hidden)).squeeze(-1).cpu()  # [batch, tokens]
            for row, i in enumerate(idx):
                out[i] = Embedding(
                    dense=dense[row].tolist(),
                    sparse=sparse_from_token_weights(
                        enc["input_ids"][row].tolist(), weights[row].tolist(), self.unused_ids
                    ),
                )
        return [out[i] for i in range(len(texts))]  # back in the caller's order
