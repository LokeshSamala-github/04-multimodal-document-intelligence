"""Turn ingested documents into retrieval chunks.

Every synthetic document here is short (a one-page invoice, receipt, memo,
or a handful of expense rows), so one chunk per document is the right
granularity -- it keeps the chunk-to-document mapping trivial, which is
what makes recall@k and MRR mean exactly what they say in evaluate.py. See
the README's "what I'd do next" section for the real-world case (long,
multi-page documents) where this would need to become an actual
overlapping-window chunker.
"""

from __future__ import annotations

from doc_intelligence.documents import Chunk, IngestedDocument


def chunk_document(doc: IngestedDocument) -> list[Chunk]:
    return [Chunk(chunk_id=f"{doc.doc_id}::0", doc_id=doc.doc_id, doc_type=doc.doc_type, text=doc.text, fields=doc.fields)]


def chunk_documents(docs: list[IngestedDocument]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks
