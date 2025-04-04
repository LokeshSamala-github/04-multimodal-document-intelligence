"""Central configuration for the document-intelligence pipeline.

Same pattern as project #1: a frozen dataclass, not a settings framework,
because the whole point is that every path used by every stage is declared
in exactly one place. Every field is overridable with a `DOCINT_*`
environment variable (see `_env_path`), which is what lets the test suite
point the whole pipeline at a throwaway directory without any monkeypatching
(see tests/conftest.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    raw_dir: Path = field(default_factory=lambda: _env_path("DOCINT_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
    images_dir: Path = field(
        default_factory=lambda: _env_path("DOCINT_IMAGES_DIR", PROJECT_ROOT / "data" / "raw" / "images")
    )
    tables_dir: Path = field(
        default_factory=lambda: _env_path("DOCINT_TABLES_DIR", PROJECT_ROOT / "data" / "raw" / "tables")
    )
    reports_txt_dir: Path = field(
        default_factory=lambda: _env_path(
            "DOCINT_REPORTS_TXT_DIR", PROJECT_ROOT / "data" / "raw" / "reports_txt"
        )
    )
    ground_truth_file: Path = field(
        default_factory=lambda: _env_path(
            "DOCINT_GROUND_TRUTH_FILE", PROJECT_ROOT / "data" / "raw" / "ground_truth.json"
        )
    )
    eval_queries_file: Path = field(
        default_factory=lambda: _env_path(
            "DOCINT_EVAL_QUERIES_FILE", PROJECT_ROOT / "data" / "raw" / "eval_queries.json"
        )
    )
    ocr_cache_file: Path = field(
        default_factory=lambda: _env_path(
            "DOCINT_OCR_CACHE_FILE", PROJECT_ROOT / "data" / "raw" / "ocr_cache.json"
        )
    )
    metrics_file: Path = field(
        default_factory=lambda: _env_path("DOCINT_METRICS_FILE", PROJECT_ROOT / "reports" / "metrics.json")
    )


@dataclass(frozen=True)
class GenerationConfig:
    """How many synthetic documents to generate, per modality."""

    n_invoices: int = 16
    n_receipts: int = 14
    n_report_images: int = 10
    n_expense_csvs: int = 6
    n_text_reports: int = 6
    seed: int = 42


@dataclass(frozen=True)
class RetrievalConfig:
    default_top_k: int = 5
    # "auto" | "tfidf" | "sentence-transformers". Env-var overridable like
    # every Paths field, so tests/CI can pin "tfidf" and skip the
    # network-dependent sentence-transformers download attempt entirely
    # instead of waiting out its retry/backoff on every corpus build.
    embedding_backend: str = field(
        default_factory=lambda: os.environ.get("DOCINT_EMBEDDING_BACKEND", "auto")
    )


PATHS = Paths()
GENERATION_CONFIG = GenerationConfig()
RETRIEVAL_CONFIG = RetrievalConfig()
