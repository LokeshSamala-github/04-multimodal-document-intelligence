"""Builds the corpus and writes reports/metrics.json with real, measured numbers:

  - OCR word accuracy (1 - WER), overall and per image document type, against
    the known ground-truth text.
  - Structured-extraction field precision, overall and per document type,
    against the known ground-truth fields.
  - Retrieval recall@1/3/5 and MRR over data/raw/eval_queries.json, a fixed
    set of natural-language queries each with one known correct source
    document.

Run with:
    python -m doc_intelligence.evaluate
or:
    make evaluate
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from doc_intelligence.config import PATHS, RETRIEVAL_CONFIG, Paths
from doc_intelligence.pipeline import Corpus, build_corpus

FIELD_CHECKS: dict[str, list[str]] = {
    "invoice": ["vendor", "invoice_number", "date", "total"],
    "receipt": ["store", "date", "total", "payment_method"],
    "report_image": ["title", "author", "date"],
    "text_report": ["title", "author", "date"],
    "expense_csv": ["department", "month", "row_count", "total_amount", "top_category"],
}


def _field_matches(extracted, expected) -> bool:
    if extracted is None or expected is None:
        return extracted == expected
    if isinstance(expected, int | float) or isinstance(extracted, int | float):
        try:
            return abs(float(extracted) - float(expected)) <= max(0.01, abs(float(expected)) * 0.005)
        except (TypeError, ValueError):
            return False
    return str(extracted).strip().lower() == str(expected).strip().lower()


def evaluate_ocr(corpus: Corpus) -> dict:
    by_type: dict[str, list[float]] = {}
    for doc in corpus.documents:
        if doc.ocr_word_accuracy is not None:
            by_type.setdefault(doc.doc_type, []).append(doc.ocr_word_accuracy)
    all_scores = [s for scores in by_type.values() for s in scores]
    return {
        "backend": corpus.ocr_backend_name,
        "n_documents": len(all_scores),
        "overall_word_accuracy": round(sum(all_scores) / len(all_scores), 4) if all_scores else None,
        "by_doc_type": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(by_type.items())
        },
    }


def evaluate_extraction(corpus: Corpus, ground_truth: list[dict]) -> dict:
    gt_by_id = {r["doc_id"]: r for r in ground_truth}
    per_field_hits: dict[str, int] = {}
    per_field_total: dict[str, int] = {}
    per_type_hits: dict[str, int] = {}
    per_type_total: dict[str, int] = {}

    for doc in corpus.documents:
        fields_to_check = FIELD_CHECKS.get(doc.doc_type, [])
        expected_fields = gt_by_id.get(doc.doc_id, {}).get("fields", {})
        for f in fields_to_check:
            key = f"{doc.doc_type}.{f}"
            match = _field_matches(doc.fields.get(f), expected_fields.get(f))
            per_field_hits[key] = per_field_hits.get(key, 0) + int(match)
            per_field_total[key] = per_field_total.get(key, 0) + 1
            per_type_hits[doc.doc_type] = per_type_hits.get(doc.doc_type, 0) + int(match)
            per_type_total[doc.doc_type] = per_type_total.get(doc.doc_type, 0) + 1

    total_hits = sum(per_type_hits.values())
    total_checks = sum(per_type_total.values())
    return {
        "overall_precision": round(total_hits / total_checks, 4) if total_checks else None,
        "n_field_checks": total_checks,
        "by_doc_type": {
            k: round(per_type_hits[k] / per_type_total[k], 4) for k in sorted(per_type_total)
        },
        "by_field": {
            k: round(per_field_hits[k] / per_field_total[k], 4) for k in sorted(per_field_total)
        },
    }


def evaluate_retrieval(corpus: Corpus, queries: list[dict]) -> dict:
    ks = [1, 3, 5]
    recall_hits = dict.fromkeys(ks, 0)
    reciprocal_ranks = []
    per_query = []

    for q in queries:
        results = corpus.query(q["query"], top_k=len(corpus.chunks))
        ranked_doc_ids = [r.chunk.doc_id for r in results]
        rank = None
        if q["expected_doc_id"] in ranked_doc_ids:
            rank = ranked_doc_ids.index(q["expected_doc_id"]) + 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in ks:
            if rank is not None and rank <= k:
                recall_hits[k] += 1
        per_query.append({
            "query": q["query"], "expected_doc_id": q["expected_doc_id"], "rank": rank,
            "top1_doc_id": ranked_doc_ids[0] if ranked_doc_ids else None,
        })

    n = len(queries)
    return {
        "n_queries": n,
        "embedding_backend": corpus.embedding_backend.name,
        **{f"recall_at_{k}": round(recall_hits[k] / n, 4) for k in ks},
        "mrr": round(sum(reciprocal_ranks) / n, 4) if n else None,
        "per_query": per_query,
    }


def run_evaluation(paths: Paths, embedding_backend_pref: str = "auto") -> dict:
    started = time.time()
    corpus = build_corpus(paths, embedding_backend_pref=embedding_backend_pref)
    ground_truth = json.loads(paths.ground_truth_file.read_text(encoding="utf-8"))
    eval_queries = json.loads(paths.eval_queries_file.read_text(encoding="utf-8"))

    by_type: dict[str, int] = {}
    for doc in corpus.documents:
        by_type[doc.doc_type] = by_type.get(doc.doc_type, 0) + 1

    return {
        "evaluated_at_unix": int(started),
        "corpus": {
            "n_documents": len(corpus.documents),
            "n_chunks": len(corpus.chunks),
            "by_doc_type": dict(sorted(by_type.items())),
            "build_seconds": round(corpus.build_seconds, 3),
        },
        "ocr": evaluate_ocr(corpus),
        "extraction": evaluate_extraction(corpus, ground_truth),
        "retrieval": evaluate_retrieval(corpus, eval_queries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-backend", default=RETRIEVAL_CONFIG.embedding_backend,
                         choices=["auto", "tfidf", "sentence-transformers"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    metrics = run_evaluation(PATHS, embedding_backend_pref=args.embedding_backend)
    out_path = args.out or PATHS.metrics_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"OCR backend: {metrics['ocr']['backend']}  |  word accuracy: {metrics['ocr']['overall_word_accuracy']}")
    print(f"extraction overall precision: {metrics['extraction']['overall_precision']}")
    r = metrics["retrieval"]
    print(f"retrieval ({r['embedding_backend']}): recall@1={r['recall_at_1']} "
          f"recall@3={r['recall_at_3']} recall@5={r['recall_at_5']} mrr={r['mrr']}")


if __name__ == "__main__":
    main()
