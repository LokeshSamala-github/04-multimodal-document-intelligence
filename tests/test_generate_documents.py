from __future__ import annotations

from pathlib import Path

from data.generate_documents import generate


def test_generate_writes_all_modalities(tmp_path: Path) -> None:
    records, queries = generate(
        n_invoices=2, n_receipts=2, n_report_images=2, n_expense_csvs=2, n_text_reports=2,
        seed=1, out_dir=tmp_path,
    )
    assert len(records) == 10
    doc_types = {r.doc_type for r in records}
    assert doc_types == {"invoice", "receipt", "report_image", "expense_csv", "text_report"}

    for r in records:
        src = tmp_path / r.source_path
        assert src.exists(), f"{r.source_path} was not written to disk"

    # image files are real JPEGs, not empty stubs
    from PIL import Image

    image_records = [r for r in records if r.modality == "image"]
    assert len(image_records) == 6
    for r in image_records:
        with Image.open(tmp_path / r.source_path) as img:
            assert img.size[0] > 100 and img.size[1] > 100


def test_invoice_fields_are_internally_consistent(tmp_path: Path) -> None:
    records, _ = generate(n_invoices=5, n_receipts=0, n_report_images=0, n_expense_csvs=0,
                           n_text_reports=0, seed=2, out_dir=tmp_path)
    for r in records:
        f = r.fields
        computed_subtotal = round(sum(li["line_total"] for li in f["line_items"]), 2)
        assert abs(computed_subtotal - f["subtotal"]) < 0.01
        assert abs(f["subtotal"] + f["tax"] - f["total"]) < 0.01
        assert f["invoice_number"].startswith("INV-")


def test_expense_csv_matches_ground_truth_fields(tmp_path: Path) -> None:
    import csv

    records, _ = generate(n_invoices=0, n_receipts=0, n_report_images=0, n_expense_csvs=3,
                           n_text_reports=0, seed=3, out_dir=tmp_path)
    for r in records:
        with (tmp_path / r.source_path).open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == r.fields["row_count"]
        total = round(sum(float(row["amount"]) for row in rows), 2)
        assert abs(total - r.fields["total_amount"]) < 0.01


def test_eval_queries_reference_real_documents(tmp_path: Path) -> None:
    records, queries = generate(
        n_invoices=3, n_receipts=3, n_report_images=2, n_expense_csvs=2, n_text_reports=2,
        seed=4, out_dir=tmp_path,
    )
    doc_ids = {r.doc_id for r in records}
    assert len(queries) > 0
    for q in queries:
        assert q["expected_doc_id"] in doc_ids
        assert isinstance(q["query"], str) and len(q["query"]) > 5


def test_generation_is_deterministic_given_a_seed(tmp_path: Path) -> None:
    r1, q1 = generate(n_invoices=3, n_receipts=2, n_report_images=1, n_expense_csvs=1,
                       n_text_reports=1, seed=99, out_dir=tmp_path / "a")
    r2, q2 = generate(n_invoices=3, n_receipts=2, n_report_images=1, n_expense_csvs=1,
                       n_text_reports=1, seed=99, out_dir=tmp_path / "b")
    assert [r.fields for r in r1] == [r.fields for r in r2]
    assert q1 == q2
