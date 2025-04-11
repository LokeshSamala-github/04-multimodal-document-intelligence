"""Embedding backends, behind one common interface.

Honest fallback, exercised for real (not just described in prose): the
preferred backend is `sentence-transformers` (a real neural embedding
model, `all-MiniLM-L6-v2`). Building it actually attempts to download that
model's weights from huggingface.co on first use. In this project's build
environment that download is blocked by the network sandbox (see README),
so the `try/except` below genuinely falls through to the classical
alternative: scikit-learn's `TfidfVectorizer`, which needs no network at
all and is a perfectly legitimate "classical embeddings" baseline for a
corpus this size. `sentence-transformers` is intentionally NOT in
requirements.txt (it pulls in torch, and this project's default, tested
path is TF-IDF) -- it's an opt-in extra; install it yourself and this
module will pick it up automatically if the model download succeeds in
your environment.

Both backends implement `fit_transform(texts) -> matrix` and
`transform(texts) -> matrix`, returning L2-normalized row vectors so cosine
similarity is a plain dot product (see index.py).
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    name: str

    def fit_transform(self, texts: list[str]) -> np.ndarray: ...

    def transform(self, texts: list[str]) -> np.ndarray: ...


class TfidfEmbeddingBackend:
    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        # sublinear_tf softens the effect of a term repeated many times in
        # one short document (e.g. "$" appearing on every line item);
        # min_df=1 is deliberate -- this corpus is small and every term is
        # potentially a distinguishing invoice/vendor/store name.
        self._vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=1, stop_words="english")

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.fit_transform(texts)
        return matrix.toarray().astype(np.float32)

    def transform(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)


class SentenceTransformerEmbeddingBackend:
    name = "sentence-transformers:all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def build_embedding_backend(preferred: str = "auto") -> EmbeddingBackend:
    """Return a ready-to-use embedding backend, preferring
    sentence-transformers if it's installed AND the model can actually be
    loaded (i.e. either already cached locally or a live download
    succeeds), falling back to TF-IDF otherwise."""
    if preferred in ("auto", "sentence-transformers"):
        try:
            backend = SentenceTransformerEmbeddingBackend()
            logger.info("embedding backend: %s", backend.name)
            return backend
        except Exception as exc:  # ImportError, or a network/HF Hub error at model load time
            if preferred == "sentence-transformers":
                raise
            logger.warning(
                "sentence-transformers unavailable (%s: %s); falling back to TF-IDF embeddings",
                type(exc).__name__, exc,
            )
    backend = TfidfEmbeddingBackend()
    logger.info("embedding backend: %s", backend.name)
    return backend
