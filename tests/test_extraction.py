from __future__ import annotations

import csv

from doc_intelligence import extraction


def test_extract_invoice_fields_from_clean_text() -> None:
    text = (
        "Acme Logistics INVOICE\n"
        "Invoice #: INV-40213\n"
        "Date: 2026-03-14\n"
        "Description Qty Unit Price Line Total\n"
        "Widget Assembly Kit 3 $20.00 $60.00\n"
        "Subtotal: $60.00\n"
        "Tax (8.25%): $4.95\n"
        "Total: $64.95\n"
    )
    fields = extraction.extract_invoice_fields(text)
    assert fields["vendor"] == "Acme Logistics"
    assert fields["invoice_number"] == "INV-40213"
    assert fields["date"] == "2026-03-14"
    assert fields["total"] == 64.95


def test_extract_invoice_total_ignores_subtotal_line() -> None:
    text = "Vendor Co\nSubtotal: $10.00\nTotal: $12.00\n"
    fields = extraction.extract_invoice_fields(text)
    assert fields["total"] == 12.00


def test_extract_receipt_fields() -> None:
    text = (
        "Corner Market\n"
        "Date: 2025-07-02\n"
        "Coffee - Large $4.25\n"
        "Subtotal: $4.25\n"
        "Tax: $0.30\n"
        "Total: $4.55\n"
        "Payment: Credit Card\n"
    )
    fields = extraction.extract_receipt_fields(text)
    assert fields["store"] == "Corner Market"
    assert fields["date"] == "2025-07-02"
    assert fields["total"] == 4.55
    assert fields["payment_method"] == "Credit Card"


def test_extract_report_fields_handles_author_and_date_on_same_line() -> None:
    text = (
        "March Fulfillment Status Report\n"
        "Author: J. Alvarez Date: 2026-03-01\n"
        "\nThe Fulfillment team processed 400 items.\n"
    )
    fields = extraction.extract_report_fields(text)
    assert fields["title"] == "March Fulfillment Status Report"
    assert fields["author"] == "J. Alvarez"
    assert fields["date"] == "2026-03-01"


def test_extract_expense_csv_fields(tmp_path) -> None:
    rows = [
        {"date": "2026-01-05", "category": "Travel", "description": "Travel expense - Regional Sales", "amount": "100.00"},
        {"date": "2026-01-09", "category": "Software", "description": "Software expense - Regional Sales", "amount": "50.00"},
    ]
    csv_path = tmp_path / "expense.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "category", "description", "amount"])
        writer.writeheader()
        writer.writerows(rows)

    fields = extraction.extract_expense_csv_fields(csv_path)
    assert fields["row_count"] == 2
    assert fields["total_amount"] == 150.0
    assert fields["top_category"] == "Travel"
    assert fields["department"] == "Regional Sales"
    assert fields["month"] == "2026-01"


def test_extract_fields_dispatch_unknown_type_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown doc_type"):
        extraction.extract_fields("mystery_type", "some text")
