from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from doc_intelligence.api import app


@pytest.fixture(scope="module")
def client(_raw_corpus):
    # Reuses the session-built corpus directory (conftest's autouse
    # _raw_corpus fixture); the API's own lifespan handler builds its own
    # in-memory Corpus from the same DOCINT_* paths on startup.
    with TestClient(app) as c:
        yield c


def test_health(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "corpus_loaded": True}


def test_corpus_info(client) -> None:
    resp = client.get("/corpus/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_documents"] > 0
    assert body["n_chunks"] == body["n_documents"]
    assert body["ocr_backend"] in ("tesseract", "ground_truth_fallback")
    assert body["embedding_backend"]


def test_query_returns_ranked_results(client) -> None:
    resp = client.post("/query", json={"query": "invoice total amount", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"]
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_query_rejects_empty_string(client) -> None:
    resp = client.post("/query", json={"query": "", "top_k": 3})
    assert resp.status_code == 422


def test_get_document_found_and_not_found(client) -> None:
    info_resp = client.get("/corpus/info")
    query_resp = client.post("/query", json={"query": "report", "top_k": 1})
    doc_id = query_resp.json()["results"][0]["doc_id"]

    resp = client.get(f"/documents/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["doc_id"] == doc_id
    assert info_resp.status_code == 200

    resp_404 = client.get("/documents/totally-unknown-doc")
    assert resp_404.status_code == 404


def test_ingest_text_document_then_query_finds_it(client) -> None:
    resp = client.post("/ingest", json={
        "doc_id": "api-ad-hoc-001",
        "doc_type": "text_report",
        "text": "Quarterly Falcon Analytics rollout update prepared by API Test Author.",
        "fields": {"title": "Quarterly Falcon Analytics rollout update", "author": "API Test Author"},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "api-ad-hoc-001"

    query_resp = client.post("/query", json={"query": "Falcon Analytics rollout update", "top_k": 3})
    doc_ids = [r["doc_id"] for r in query_resp.json()["results"]]
    assert "api-ad-hoc-001" in doc_ids


def test_ingest_duplicate_doc_id_conflicts(client) -> None:
    resp = client.post("/ingest", json={"doc_id": "api-ad-hoc-001", "text": "dup"})
    assert resp.status_code == 409
