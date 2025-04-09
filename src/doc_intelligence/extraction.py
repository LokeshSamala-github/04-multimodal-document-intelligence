"""Rule-based structured-field extraction.

This is the text-to-fields half of the "VLM" substitute described in
ocr.py: a real VLM call would return structured fields directly from
pixels in one shot. Here that is split into two honest, inspectable, and
separately measurable steps -- OCR (pixels to text) then this module (text
to fields), using plain regex/line-based rules because the synthetic
documents have a small, fixed set of layouts. `evaluate.py` measures how
often each extracted field matches the known ground truth, which is the
real "extraction accuracy" reported in reports/metrics.json.

For the CSV and plain-text documents there is no OCR noise to propagate --
extraction there reads the source data (or text) directly -- so their
accuracy is expected to sit near 100%, in contrast to the OCR'd image
documents where extraction accuracy is capped by whatever OCR got wrong.
That gap, reported honestly, is itself part of the point of this project.
"""

from __future__ import annotations

import csv
import re
from io import StringIO
from pathlib import Path

_MONEY_RE = re.compile(r"\$?\s?([\d,]+\.\d{2})")
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_INVOICE_NUM_RE = re.compile(r"INV-\d+")


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _last_money_on_labeled_line(lines: list[str], label: str, exclude: str | None = None) -> float | None:
    best = None
    for line in lines:
        if re.search(label, line, re.IGNORECASE) and (exclude is None or exclude.lower() not in line.lower()):
            found = _MONEY_RE.findall(line)
            if found:
                best = float(found[-1].replace(",", ""))
    return best


def _first_line_before(lines: list[str], boilerplate: str) -> str | None:
    """The vendor/store/title header line, with a same-line boilerplate
    label (e.g. "INVOICE") stripped off -- OCR frequently merges the two
    since they're drawn on the same row of the rendered page."""
    if not lines:
        return None
    first = lines[0]
    m = re.search(rf"^(.*?)\s*{boilerplate}\b", first, re.IGNORECASE)
    return m.group(1).strip() if m and m.group(1).strip() else first


def extract_invoice_fields(text: str) -> dict:
    lines = _lines(text)
    m_num = _INVOICE_NUM_RE.search(text)
    m_date = _DATE_RE.search(text)
    return {
        "vendor": _first_line_before(lines, "invoice"),
        "invoice_number": m_num.group(0) if m_num else None,
        "date": m_date.group(1) if m_date else None,
        "total": _last_money_on_labeled_line(lines, r"\btotal\b", exclude="subtotal"),
    }


def extract_receipt_fields(text: str) -> dict:
    lines = _lines(text)
    m_date = _DATE_RE.search(text)
    m_pay = re.search(r"Payment:?\s*(Cash|Credit Card|Debit Card)", text, re.IGNORECASE)
    payment = None
    if m_pay:
        raw = m_pay.group(1).strip().title()
        payment = {"Cash": "Cash", "Credit Card": "Credit Card", "Debit Card": "Debit Card"}.get(raw, raw)
    return {
        "store": lines[0] if lines else None,
        "date": m_date.group(1) if m_date else None,
        "total": _last_money_on_labeled_line(lines, r"\btotal\b", exclude="subtotal"),
        "payment_method": payment,
    }


def extract_report_fields(text: str) -> dict:
    lines = _lines(text)
    # Non-greedy, and stop at a same-line "Date:" label -- OCR often puts
    # "Author: X   Date: Y" on a single physical line.
    m_author = re.search(r"Author:?\s*(.+?)(?=\s{2,}Date\b|\s+Date:|\n|$)", text)
    m_date = _DATE_RE.search(text)
    return {
        "title": lines[0] if lines else None,
        "author": m_author.group(1).strip() if m_author else None,
        "date": m_date.group(1) if m_date else None,
    }


def extract_expense_csv_fields(csv_path: Path) -> dict:
    """Parse the table directly (this is the "structured format" ingestion
    path -- no OCR, no regex-on-prose, just reading the data)."""
    with Path(csv_path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = 0.0
    by_category: dict[str, float] = {}
    dept = None
    dates = []
    for row in rows:
        amount = float(row["amount"])
        total += amount
        by_category[row["category"]] = round(by_category.get(row["category"], 0.0) + amount, 2)
        dates.append(row["date"])
        m_dept = re.search(r"expense - (.+)$", row["description"])
        if m_dept:
            dept = m_dept.group(1)

    top_category = max(by_category, key=by_category.get) if by_category else None
    month = dates[0][:7] if dates else None
    return {
        "department": dept,
        "month": month,
        "row_count": len(rows),
        "total_amount": round(total, 2),
        "top_category": top_category,
        "by_category": by_category,
    }


def extract_fields(doc_type: str, text: str, raw_path: Path | None = None) -> dict:
    """Dispatch to the right extractor for a document type."""
    if doc_type == "invoice":
        return extract_invoice_fields(text)
    if doc_type == "receipt":
        return extract_receipt_fields(text)
    if doc_type in ("report_image", "text_report"):
        return extract_report_fields(text)
    if doc_type == "expense_csv":
        if raw_path is None:
            # text-only fallback (e.g. re-parsing the flattened text instead
            # of the CSV file) -- used only by tests exercising the regex
            # path directly.
            reader = csv.DictReader(StringIO(text))
            rows = list(reader)
            return {"row_count": len(rows)}
        return extract_expense_csv_fields(raw_path)
    raise ValueError(f"unknown doc_type: {doc_type}")
