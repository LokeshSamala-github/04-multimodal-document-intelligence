from __future__ import annotations

import json

from doc_intelligence.config import PATHS
from doc_intelligence.evaluate import run_evaluation


def test_corpus_builds_all_modalities(corpus) -> None:
    modalities = {d.modality for d in corpus.documents}
    assert modalities == {"image", "csv", "text"}
    assert len(corpus.documents) == len(corpus.chunks)  # one chunk per doc, by design
    assert len(corpus.index) == len(corpus.chunks)


def test_image_documents_have_ocr_accuracy_recorded(corpus) -> None:
    image_docs = [d for d in corpus.documents if d.modality == "image"]
    assert image_docs
    for d in image_docs:
        assert d.ocr_word_accuracy is not None
        assert 0.0 <= d.ocr_word_accuracy <= 1.0


def test_query_retrieves_the_matching_invoice_by_number(corpus) -> None:
    invoice = next(d for d in corpus.documents if d.doc_type == "invoice")
    inv_num = invoice.fields["invoice_number"]
    vendor = invoice.fields["vendor"]
    results = corpus.query(f"What is the total on invoice {inv_num} from {vendor}?", top_k=3)
    assert results
    assert results[0].chunk.doc_id == invoice.doc_id


def test_query_retrieves_the_matching_receipt_by_store_and_date(corpus) -> None:
    receipt = next(d for d in corpus.documents if d.doc_type == "receipt")
    results = corpus.query(
        f"Find the {receipt.fields['store']} receipt from {receipt.fields['date']}", top_k=3
    )
    assert results[0].chunk.doc_id == receipt.doc_id


def test_get_document_returns_none_for_unknown_id(corpus) -> None:
    assert corpus.get_document("does-not-exist") is None


def test_add_text_document_is_immediately_queryable_and_rejects_duplicates(corpus) -> None:
    import pytest

    n_docs_before = len(corpus.documents)
    n_chunks_before = len(corpus.chunks)

    corpus.add_text_document(
        doc_id="ad-hoc-001",
        doc_type="text_report",
        text="Special Zebra Migration Project status update by Q. Rare, filed 2026-05-01.",
        fields={"title": "Special Zebra Migration Project", "author": "Q. Rare", "date": "2026-05-01"},
    )
    assert len(corpus.documents) == n_docs_before + 1
    assert len(corpus.chunks) == n_chunks_before + 1
    assert len(corpus.index) == n_chunks_before + 1

    results = corpus.query("Who filed the Zebra Migration Project update?", top_k=3)
    assert results[0].chunk.doc_id == "ad-hoc-001"

    with pytest.raises(ValueError, match="already exists"):
        corpus.add_text_document("ad-hoc-001", "text_report", "duplicate", {})


def test_run_evaluation_produces_plausible_metrics(tmp_path) -> None:
    metrics = run_evaluation(PATHS, embedding_backend_pref="tfidf")

    assert metrics["corpus"]["n_documents"] > 0
    assert metrics["ocr"]["overall_word_accuracy"] > 0.5
    assert 0.0 <= metrics["extraction"]["overall_precision"] <= 1.0
    retrieval = metrics["retrieval"]
    assert retrieval["recall_at_5"] >= retrieval["recall_at_1"]
    assert retrieval["recall_at_5"] >= retrieval["mrr"]
    assert 0.0 <= retrieval["mrr"] <= 1.0

    # metrics.json actually gets written by main(), sanity check it round-trips as JSON
    json.dumps(metrics)
