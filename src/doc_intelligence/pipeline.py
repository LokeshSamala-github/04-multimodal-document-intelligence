"""Wires ingestion -> chunking -> embedding -> indexing into one `Corpus`.

`Corpus.build` is the single entry point the CLI evaluator, the test suite,
and the FastAPI app all call to go from the raw data/ directory to a
queryable, in-memory index. Rebuilding it is cheap at this corpus size (a
few dozen documents), so nothing here is persisted to disk between runs --
see api.py's lifespan handler, which builds it once at process startup.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from doc_intelligence import chunking, ingest, ocr
from doc_intelligence.config import Paths
from doc_intelligence.documents import Chunk, IngestedDocument
from doc_intelligence.embeddings import EmbeddingBackend, build_embedding_backend
from doc_intelligence.index import ScoredChunk, VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class Corpus:
    documents: list[IngestedDocument]
    chunks: list[Chunk]
    index: VectorIndex
    embedding_backend: EmbeddingBackend
    ocr_backend_name: str
    build_seconds: float = 0.0
    _doc_by_id: dict[str, IngestedDocument] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._doc_by_id = {d.doc_id: d for d in self.documents}

    def get_document(self, doc_id: str) -> IngestedDocument | None:
        return self._doc_by_id.get(doc_id)

    def query(self, text: str, top_k: int = 5) -> list[ScoredChunk]:
        vector = self.embedding_backend.transform([text])[0]
        return self.index.search(vector, top_k=top_k)

    def add_text_document(self, doc_id: str, doc_type: str, text: str, fields: dict) -> None:
        """Ingest one new plain-text document into the live, in-memory
        index -- this is what backs POST /ingest. Only plain text is
        supported here (no OCR at request time): a real system would run
        the same ocr.py + extraction.py path an uploaded image would need,
        which is out of scope for a live API call in this project.

        This re-fits the embedding backend on the whole (now one-document-
        larger) chunk set rather than just transforming the new chunk with
        the old, frozen one. For TF-IDF that matters: `transform` silently
        drops any word that wasn't in the vocabulary at fit time, so a
        transform-only add would make a genuinely new term (a vendor name,
        a project name) invisible to search. Refitting is cheap at this
        corpus size (well under a second) and keeps the guarantee that a
        just-ingested document is actually findable, which is the whole
        point of an ingestion endpoint.
        """
        if doc_id in self._doc_by_id:
            raise ValueError(f"doc_id already exists: {doc_id}")
        doc = IngestedDocument(
            doc_id=doc_id, doc_type=doc_type, modality="text", source_path="<api>",
            text=text, fields=fields,
        )
        self.documents.append(doc)
        self._doc_by_id[doc_id] = doc
        self.chunks.extend(chunking.chunk_document(doc))
        matrix = self.embedding_backend.fit_transform([c.text for c in self.chunks])
        self.index = VectorIndex(self.chunks, matrix)


def build_corpus(paths: Paths, embedding_backend_pref: str = "auto") -> Corpus:
    start = time.monotonic()
    documents = ingest.ingest_corpus(paths.raw_dir, paths.ground_truth_file, paths.ocr_cache_file)
    chunks = chunking.chunk_documents(documents)
    backend = build_embedding_backend(embedding_backend_pref)
    matrix = backend.fit_transform([c.text for c in chunks])
    index = VectorIndex(chunks, matrix)
    elapsed = time.monotonic() - start
    logger.info(
        "built corpus: %d documents, %d chunks, embedding=%s, ocr=%s, %.2fs",
        len(documents), len(chunks), backend.name, ocr.backend_name(), elapsed,
    )
    return Corpus(
        documents=documents, chunks=chunks, index=index, embedding_backend=backend,
        ocr_backend_name=ocr.backend_name(), build_seconds=elapsed,
    )
