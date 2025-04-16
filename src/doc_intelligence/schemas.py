"""Pydantic request/response models for the FastAPI serving layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_type: str
    score: float
    snippet: str
    fields: dict


class QueryResponse(BaseModel):
    query: str
    embedding_backend: str
    results: list[RetrievedChunk]


class DocumentInfo(BaseModel):
    doc_id: str
    doc_type: str
    modality: str
    fields: dict
    text: str


class CorpusInfo(BaseModel):
    n_documents: int
    n_chunks: int
    by_doc_type: dict[str, int]
    ocr_backend: str
    embedding_backend: str
    build_seconds: float


class IngestTextRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    doc_type: str = Field(default="text_report")
    text: str = Field(min_length=1)
    fields: dict = Field(default_factory=dict)


class IngestTextResponse(BaseModel):
    doc_id: str
    n_documents: int
    n_chunks: int
