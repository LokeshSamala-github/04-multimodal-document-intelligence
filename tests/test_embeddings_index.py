from __future__ import annotations

import numpy as np

from doc_intelligence.documents import Chunk
from doc_intelligence.embeddings import TfidfEmbeddingBackend, build_embedding_backend
from doc_intelligence.index import VectorIndex


def test_tfidf_backend_ranks_matching_document_first() -> None:
    texts = [
        "Acme Logistics invoice total 938.06 dollars",
        "Corner Market receipt coffee bagel total 8.20",
        "Quarterly Fulfillment status report by J Alvarez",
    ]
    backend = TfidfEmbeddingBackend()
    matrix = backend.fit_transform(texts)
    query_vec = backend.transform(["What is the total on the Acme Logistics invoice?"])[0]

    scores = matrix @ query_vec
    assert int(np.argmax(scores)) == 0


def test_tfidf_vectors_are_unit_normalized() -> None:
    backend = TfidfEmbeddingBackend()
    matrix = backend.fit_transform(["alpha beta gamma", "delta epsilon zeta alpha"])
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_build_embedding_backend_falls_back_to_tfidf_when_sentence_transformers_unavailable(monkeypatch) -> None:
    import doc_intelligence.embeddings as emb_mod

    def _boom(self):
        raise ImportError("simulated: sentence-transformers not installed")

    monkeypatch.setattr(emb_mod.SentenceTransformerEmbeddingBackend, "__init__", _boom)
    backend = build_embedding_backend("auto")
    assert backend.name == "tfidf"


def test_vector_index_search_returns_ranked_top_k() -> None:
    chunks = [
        Chunk(chunk_id="a", doc_id="a", doc_type="x", text="a"),
        Chunk(chunk_id="b", doc_id="b", doc_type="x", text="b"),
        Chunk(chunk_id="c", doc_id="c", doc_type="x", text="c"),
    ]
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    index = VectorIndex(chunks, matrix)

    query = np.array([1.0, 0.0], dtype=np.float32)
    results = index.search(query, top_k=2)
    assert [r.chunk.doc_id for r in results] == ["a", "c"]
    assert results[0].score > results[1].score


def test_vector_index_rejects_mismatched_lengths() -> None:
    import pytest

    chunks = [Chunk(chunk_id="a", doc_id="a", doc_type="x", text="a")]
    with pytest.raises(ValueError):
        VectorIndex(chunks, np.zeros((2, 3)))
