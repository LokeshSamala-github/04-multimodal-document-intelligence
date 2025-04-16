"""FastAPI serving layer.

Run with:
    uvicorn doc_intelligence.api:app --reload

Endpoints:
    GET  /health              liveness probe
    GET  /corpus/info         corpus size, OCR + embedding backends actually in use
    POST /query                top-k relevant chunks for a natural-language query
    GET  /documents/{doc_id}   one document's extracted text + fields
    POST /ingest                add a new plain-text document to the live index
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from doc_intelligence import __version__
from doc_intelligence.config import PATHS, RETRIEVAL_CONFIG
from doc_intelligence.pipeline import Corpus, build_corpus
from doc_intelligence.schemas import (
    CorpusInfo,
    DocumentInfo,
    IngestTextRequest,
    IngestTextResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

_state: dict[str, Corpus] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the corpus once at startup (OCR + extraction + embedding fit),
    # not on the first request -- ingesting ~50 documents takes a few
    # seconds, and the first real request shouldn't pay for it.
    _state["corpus"] = build_corpus(PATHS, embedding_backend_pref=RETRIEVAL_CONFIG.embedding_backend)
    yield
    _state.clear()


app = FastAPI(
    title="Multimodal Document Intelligence API",
    description=(
        "OCR + rule-based extraction + embedding retrieval over a mixed corpus of scanned-style "
        "invoices/receipts/memos (images), expense tables (CSV) and plain-text reports."
    ),
    version=__version__,
    lifespan=lifespan,
)


def _corpus() -> Corpus:
    corpus = _state.get("corpus")
    if corpus is None:
        raise HTTPException(status_code=503, detail="corpus not built yet")
    return corpus


@app.get("/health")
def health() -> dict:
    corpus = _state.get("corpus")
    return {"status": "ok" if corpus else "starting", "corpus_loaded": corpus is not None}


@app.get("/corpus/info", response_model=CorpusInfo)
def corpus_info() -> CorpusInfo:
    corpus = _corpus()
    by_type: dict[str, int] = {}
    for doc in corpus.documents:
        by_type[doc.doc_type] = by_type.get(doc.doc_type, 0) + 1
    return CorpusInfo(
        n_documents=len(corpus.documents),
        n_chunks=len(corpus.chunks),
        by_doc_type=dict(sorted(by_type.items())),
        ocr_backend=corpus.ocr_backend_name,
        embedding_backend=corpus.embedding_backend.name,
        build_seconds=corpus.build_seconds,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    corpus = _corpus()
    scored = corpus.query(request.query, top_k=request.top_k)
    results = [
        RetrievedChunk(
            chunk_id=s.chunk.chunk_id, doc_id=s.chunk.doc_id, doc_type=s.chunk.doc_type,
            score=round(s.score, 4), snippet=s.chunk.text[:280], fields=s.chunk.fields,
        )
        for s in scored
    ]
    return QueryResponse(query=request.query, embedding_backend=corpus.embedding_backend.name, results=results)


@app.get("/documents/{doc_id}", response_model=DocumentInfo)
def get_document(doc_id: str) -> DocumentInfo:
    corpus = _corpus()
    doc = corpus.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"unknown doc_id: {doc_id}")
    return DocumentInfo(doc_id=doc.doc_id, doc_type=doc.doc_type, modality=doc.modality, fields=doc.fields, text=doc.text)


@app.post("/ingest", response_model=IngestTextResponse)
def ingest_text(request: IngestTextRequest) -> IngestTextResponse:
    corpus = _corpus()
    try:
        corpus.add_text_document(request.doc_id, request.doc_type, request.text, request.fields)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return IngestTextResponse(doc_id=request.doc_id, n_documents=len(corpus.documents), n_chunks=len(corpus.chunks))
