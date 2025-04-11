"""A plain numpy cosine-similarity vector index.

FAISS would be overkill (and one more heavy optional dependency) for a
corpus of a few dozen chunks -- at this scale a dense matrix and a matrix
multiply IS the search, and it's exact, not approximate. See the README's
"what I'd do next" section for when this would need to become an actual
ANN index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from doc_intelligence.documents import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorIndex:
    def __init__(self, chunks: list[Chunk], matrix: np.ndarray) -> None:
        if len(chunks) != matrix.shape[0]:
            raise ValueError("chunks and matrix row count must match")
        self.chunks = chunks
        self.matrix = matrix

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[ScoredChunk]:
        if self.matrix.shape[0] == 0:
            return []
        # Rows of self.matrix and query_vector are already L2-normalized by
        # the embedding backend, so a dot product IS cosine similarity.
        scores = self.matrix @ query_vector
        top_k = min(top_k, len(self.chunks))
        top_idx = np.argsort(-scores)[:top_k]
        return [ScoredChunk(chunk=self.chunks[i], score=float(scores[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self.chunks)
