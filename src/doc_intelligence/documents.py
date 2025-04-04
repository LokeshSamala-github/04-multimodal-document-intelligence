"""The common document/chunk representation every ingestion path converges on.

Whatever the source modality (image, CSV, plain text), ingestion produces an
`IngestedDocument`: some text (OCR'd, or read directly) plus structured
`fields` pulled out by the extraction layer. Chunking then turns each
document into one or more `Chunk`s, which is what actually gets embedded and
indexed. Keeping this as a small, format-agnostic dataclass pair is what
lets `embeddings.py` and `index.py` not care at all whether a chunk
ultimately came from a scanned invoice or a CSV row.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IngestedDocument:
    doc_id: str
    doc_type: str  # invoice | receipt | report_image | expense_csv | text_report
    modality: str  # image | csv | text
    source_path: str
    text: str  # OCR output (images) or the raw file content (csv/text)
    fields: dict = field(default_factory=dict)  # rule-based structured extraction
    ocr_word_accuracy: float | None = None  # set only for OCR'd (image) documents


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    text: str
    fields: dict = field(default_factory=dict)
