"""Ingestion: turn the raw multimodal corpus into `IngestedDocument`s.

Three modalities, three read paths, one common output shape:
    image  -> OCR (ocr.py) -> extraction.py regex extraction on the OCR text
    csv    -> parsed directly with csv.DictReader -> extraction.py aggregation
    text   -> read directly -> extraction.py regex extraction on the raw text

This is the "multimodal" half of the project: each modality is genuinely
handled differently, and only the image path goes through OCR at all.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from doc_intelligence import extraction, ocr
from doc_intelligence.documents import IngestedDocument

logger = logging.getLogger(__name__)


def load_ground_truth(ground_truth_file: Path) -> list[dict]:
    if not ground_truth_file.exists():
        raise FileNotFoundError(
            f"{ground_truth_file} not found -- run `python data/generate_documents.py` first"
        )
    return json.loads(ground_truth_file.read_text(encoding="utf-8"))


def ingest_document(record: dict, raw_dir: Path, ocr_cache_file: Path) -> IngestedDocument:
    doc_id = record["doc_id"]
    doc_type = record["doc_type"]
    modality = record["modality"]
    source_path = raw_dir / record["source_path"]
    ground_truth_text = record["text"]

    ocr_word_accuracy = None
    if modality == "image":
        ocr_text = ocr.run_ocr(source_path, ground_truth_text, ocr_cache_file)
        ocr_word_accuracy = ocr.word_accuracy(ocr_text, ground_truth_text)
        text = ocr_text
        fields = extraction.extract_fields(doc_type, text)
    elif modality == "csv":
        text = ground_truth_text  # flattened rendition; no OCR involved
        fields = extraction.extract_fields(doc_type, text, raw_path=source_path)
    elif modality == "text":
        text = source_path.read_text(encoding="utf-8") if source_path.exists() else ground_truth_text
        fields = extraction.extract_fields(doc_type, text)
    else:
        raise ValueError(f"unknown modality: {modality}")

    return IngestedDocument(
        doc_id=doc_id, doc_type=doc_type, modality=modality, source_path=str(source_path),
        text=text, fields=fields, ocr_word_accuracy=ocr_word_accuracy,
    )


def ingest_corpus(raw_dir: Path, ground_truth_file: Path, ocr_cache_file: Path) -> list[IngestedDocument]:
    records = load_ground_truth(ground_truth_file)
    docs = [ingest_document(r, raw_dir, ocr_cache_file) for r in records]
    logger.info("ingested %d documents (OCR backend: %s)", len(docs), ocr.backend_name())
    return docs
