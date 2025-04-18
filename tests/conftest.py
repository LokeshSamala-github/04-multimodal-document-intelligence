"""Shared pytest fixtures.

Same isolation pattern as project #1: the whole test session runs against a
throwaway directory instead of the real data/ and reports/ folders, by
setting the DOCINT_* environment variables that `doc_intelligence.config.Paths`
already supports (see config.py's `_env_path` helper) *before* anything in
doc_intelligence gets imported. conftest.py is guaranteed to run before any
test module's imports, so every `Paths()` instance constructed anywhere in
the app picks up the temp location automatically.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (str(SRC), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

_TMP = Path(tempfile.mkdtemp(prefix="doc_intelligence_tests_"))
os.environ.setdefault("DOCINT_RAW_DIR", str(_TMP / "data" / "raw"))
os.environ.setdefault("DOCINT_IMAGES_DIR", str(_TMP / "data" / "raw" / "images"))
os.environ.setdefault("DOCINT_TABLES_DIR", str(_TMP / "data" / "raw" / "tables"))
os.environ.setdefault("DOCINT_REPORTS_TXT_DIR", str(_TMP / "data" / "raw" / "reports_txt"))
os.environ.setdefault("DOCINT_GROUND_TRUTH_FILE", str(_TMP / "data" / "raw" / "ground_truth.json"))
os.environ.setdefault("DOCINT_EVAL_QUERIES_FILE", str(_TMP / "data" / "raw" / "eval_queries.json"))
os.environ.setdefault("DOCINT_OCR_CACHE_FILE", str(_TMP / "data" / "raw" / "ocr_cache.json"))
os.environ.setdefault("DOCINT_METRICS_FILE", str(_TMP / "reports" / "metrics.json"))
# Pin TF-IDF for the whole test session: sentence-transformers would try (and,
# in this sandbox, fail) to download model weights from huggingface.co on
# every corpus build, which is slow and makes tests depend on network state.
os.environ.setdefault("DOCINT_EMBEDDING_BACKEND", "tfidf")

import json  # noqa: E402

import pytest  # noqa: E402

from doc_intelligence.config import PATHS  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _raw_corpus() -> Path:
    """Generate a small synthetic corpus once per test session, via the
    real generator script -- not a hand-typed fixture -- so tests exercise
    the actual generation code path."""
    from data.generate_documents import generate

    records, queries = generate(
        n_invoices=6, n_receipts=6, n_report_images=4, n_expense_csvs=3, n_text_reports=3,
        seed=7, out_dir=PATHS.raw_dir,
    )
    PATHS.raw_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = [
        {"doc_id": r.doc_id, "doc_type": r.doc_type, "modality": r.modality,
         "source_path": r.source_path, "text": r.text, "fields": r.fields}
        for r in records
    ]
    PATHS.ground_truth_file.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    PATHS.eval_queries_file.write_text(json.dumps(queries, indent=2), encoding="utf-8")
    return PATHS.raw_dir


@pytest.fixture(scope="session")
def corpus(_raw_corpus):
    """Build the real ingestion -> chunking -> embedding -> index pipeline
    once per session (TF-IDF backend, since sentence-transformers needs a
    model download this sandbox's network does not allow -- see README)."""
    from doc_intelligence.pipeline import build_corpus

    return build_corpus(PATHS, embedding_backend_pref="tfidf")
