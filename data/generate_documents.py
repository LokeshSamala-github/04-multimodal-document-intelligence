"""Synthetic multimodal document corpus generator.

Real scanned business documents can't ship in a public portfolio repo (and
"pre-extracted text pretending to be OCR output" would be dishonest), so this
renders actual RGB document images with PIL -- laid out like real invoices,
receipts and short memos -- and separately writes small CSV expense tables
and plain-text reports. Every document's exact ground-truth text and
structured fields are recorded in `ground_truth.json`, which is what lets
the rest of the pipeline measure *real* OCR accuracy and *real* extraction
accuracy later instead of asserting against itself.

Five document kinds ship, spanning three input modalities:
    invoice        (image)   -- header, line-item table, total
    receipt        (image)   -- store, item list, total, payment method
    report_image   (image)   -- a short "scanned" memo: title/author/date + body
    expense_csv    (csv)     -- a small tabular expense report, no OCR needed
    text_report    (text)    -- a plain-text memo, no OCR needed

Rendered images are deliberately degraded (small rotation, gaussian noise,
JPEG re-compression) so OCR on them is genuinely imperfect -- a 100% "OCR"
accuracy number on synthetic data would be a red flag, not an achievement.

Usage:
    python data/generate_documents.py                  # writes data/raw/
    python data/generate_documents.py --seed 7          # different draw
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_REGULAR = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

IMG_W, IMG_H = 850, 1100
MARGIN = 60

VENDORS = [
    "Northwind Traders", "Acme Logistics", "Blue Ridge Supplies", "Cascade Industrial",
    "Summit Office Co", "Harbor Point Freight", "Meridian Hardware", "Fairview Consulting Group",
    "Redstone Manufacturing", "Pioneer Electrical Supply",
]
INVOICE_ITEMS = [
    ("Widget Assembly Kit", 12.0, 45.0), ("Industrial Fasteners (box)", 5.0, 20.0),
    ("Shipping Pallet", 8.0, 25.0), ("Consulting Hours", 75.0, 150.0),
    ("Office Chair", 60.0, 220.0), ("Network Switch 24-port", 40.0, 180.0),
    ("Safety Gloves (pair)", 3.0, 12.0), ("Steel Bracket", 2.0, 15.0),
    ("Software License - Annual", 100.0, 500.0), ("Warehouse Labor (hrs)", 18.0, 35.0),
    ("Packing Tape (case)", 4.0, 14.0), ("LED Panel Light", 22.0, 60.0),
]

STORES = [
    "Corner Market", "Riverside Coffee Co", "QuickStop Pharmacy", "Green Leaf Grocers",
    "Metro Hardware", "Sunrise Diner", "Pixel Electronics", "Urban Cafe & Bakery",
]
RECEIPT_ITEMS = [
    ("Coffee - Large", 3.25, 5.75), ("Bagel w/ Cream Cheese", 2.5, 4.5),
    ("Notebook", 2.0, 6.0), ("AA Batteries 4-pack", 4.0, 8.0),
    ("Phone Charger Cable", 6.0, 15.0), ("Bottled Water", 1.0, 2.5),
    ("Sandwich - Turkey Club", 6.5, 9.5), ("Multivitamins", 8.0, 18.0),
    ("Greeting Card", 2.5, 5.0), ("Printer Paper Ream", 5.0, 9.0),
]
PAYMENT_METHODS = ["Cash", "Credit Card", "Debit Card"]

DEPARTMENTS = [
    "Fulfillment", "Customer Support", "IT Infrastructure", "Warehouse Safety",
    "Regional Sales", "Vendor Onboarding", "Quality Assurance", "Field Operations",
]
REPORT_AUTHORS = [
    "J. Alvarez", "M. Chen", "R. Okafor", "S. Patel", "T. Nguyen", "K. Brennan",
]
MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


def _degrade(img: Image.Image, rng: random.Random, out_path: Path) -> None:
    """Make a clean rendered page look like a real scanned/photographed
    document: a small rotation, mild gaussian blur, additive noise, then a
    lossy JPEG re-encode. This is what makes the OCR step measure something
    real instead of reading back pixel-perfect rendered text."""
    angle = rng.uniform(-1.6, 1.6)
    img = img.rotate(angle, expand=True, fillcolor="white", resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.6)))

    arr = np.asarray(img).astype(np.int16)
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, 6.0, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path, format="JPEG", quality=rng.randint(72, 88))


@dataclass
class DocRecord:
    doc_id: str
    doc_type: str
    modality: str
    source_path: str
    text: str
    fields: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- invoices
def render_invoice(doc_id: str, rng: random.Random, out_dir: Path) -> DocRecord:
    vendor = rng.choice(VENDORS)
    inv_num = f"INV-{rng.randint(10000, 99999)}"
    year = rng.choice([2025, 2026])
    month = rng.randint(1, 9 if year == 2026 else 12)
    day = rng.randint(1, 28)
    date_str = f"{year:04d}-{month:02d}-{day:02d}"

    n_items = rng.randint(2, 5)
    items = rng.sample(INVOICE_ITEMS, n_items)
    lines = []
    subtotal = 0.0
    for desc, lo, hi in items:
        qty = rng.randint(1, 8)
        unit_price = round(rng.uniform(lo, hi), 2)
        line_total = round(qty * unit_price, 2)
        subtotal += line_total
        lines.append({"description": desc, "qty": qty, "unit_price": unit_price, "line_total": line_total})
    tax = round(subtotal * 0.0825, 2)
    total = round(subtotal + tax, 2)

    img = Image.new("RGB", (IMG_W, IMG_H), "white")
    d = ImageDraw.Draw(img)
    y = MARGIN
    d.text((MARGIN, y), vendor, font=_font(26, bold=True), fill="black")
    d.text((IMG_W - MARGIN - 120, y + 4), "INVOICE", font=_font(22, bold=True), fill="black")
    y += 50
    d.line([(MARGIN, y), (IMG_W - MARGIN, y)], fill="black", width=2)
    y += 20
    d.text((MARGIN, y), f"Invoice #: {inv_num}", font=_font(16), fill="black")
    y += 24
    d.text((MARGIN, y), f"Date: {date_str}", font=_font(16), fill="black")
    y += 40

    col_x = [MARGIN, MARGIN + 380, MARGIN + 500, MARGIN + 620]
    headers = ["Description", "Qty", "Unit Price", "Line Total"]
    for x, h in zip(col_x, headers, strict=True):
        d.text((x, y), h, font=_font(15, bold=True), fill="black")
    y += 22
    d.line([(MARGIN, y), (IMG_W - MARGIN, y)], fill="black", width=1)
    y += 10
    for it in lines:
        d.text((col_x[0], y), it["description"], font=_font(15), fill="black")
        d.text((col_x[1], y), str(it["qty"]), font=_font(15), fill="black")
        d.text((col_x[2], y), f"${it['unit_price']:.2f}", font=_font(15), fill="black")
        d.text((col_x[3], y), f"${it['line_total']:.2f}", font=_font(15), fill="black")
        y += 26
    y += 10
    d.line([(MARGIN, y), (IMG_W - MARGIN, y)], fill="black", width=1)
    y += 16
    d.text((col_x[2], y), "Subtotal:", font=_font(15), fill="black")
    d.text((col_x[3], y), f"${subtotal:.2f}", font=_font(15), fill="black")
    y += 24
    d.text((col_x[2], y), "Tax (8.25%):", font=_font(15), fill="black")
    d.text((col_x[3], y), f"${tax:.2f}", font=_font(15), fill="black")
    y += 24
    d.text((col_x[2], y), "Total:", font=_font(17, bold=True), fill="black")
    d.text((col_x[3], y), f"${total:.2f}", font=_font(17, bold=True), fill="black")

    out_path = out_dir / f"{doc_id}.jpg"
    _degrade(img, rng, out_path)

    text_lines = [vendor, "INVOICE", f"Invoice #: {inv_num}", f"Date: {date_str}", "Description Qty Unit Price Line Total"]
    for it in lines:
        text_lines.append(f"{it['description']} {it['qty']} ${it['unit_price']:.2f} ${it['line_total']:.2f}")
    text_lines += [f"Subtotal: ${subtotal:.2f}", f"Tax (8.25%): ${tax:.2f}", f"Total: ${total:.2f}"]

    return DocRecord(
        doc_id=doc_id, doc_type="invoice", modality="image",
        source_path=f"images/{doc_id}.jpg", text="\n".join(text_lines),
        fields={
            "vendor": vendor, "invoice_number": inv_num, "date": date_str,
            "subtotal": round(subtotal, 2), "tax": tax, "total": total, "line_items": lines,
        },
    )


# --------------------------------------------------------------------------- receipts
def render_receipt(doc_id: str, rng: random.Random, out_dir: Path) -> DocRecord:
    store = rng.choice(STORES)
    year = rng.choice([2025, 2026])
    month = rng.randint(1, 9 if year == 2026 else 12)
    day = rng.randint(1, 28)
    date_str = f"{year:04d}-{month:02d}-{day:02d}"
    payment = rng.choice(PAYMENT_METHODS)

    n_items = rng.randint(2, 5)
    items = rng.sample(RECEIPT_ITEMS, n_items)
    lines = []
    subtotal = 0.0
    for name, lo, hi in items:
        price = round(rng.uniform(lo, hi), 2)
        subtotal += price
        lines.append({"item": name, "price": price})
    tax = round(subtotal * 0.07, 2)
    total = round(subtotal + tax, 2)

    img = Image.new("RGB", (600, 900), "white")
    d = ImageDraw.Draw(img)
    x0 = 40
    y = 40
    d.text((x0, y), store, font=_font(22, bold=True), fill="black")
    y += 34
    d.text((x0, y), f"Date: {date_str}", font=_font(14), fill="black")
    y += 30
    d.line([(x0, y), (560, y)], fill="black", width=1)
    y += 14
    for it in lines:
        d.text((x0, y), it["item"], font=_font(14), fill="black")
        d.text((460, y), f"${it['price']:.2f}", font=_font(14), fill="black")
        y += 24
    y += 6
    d.line([(x0, y), (560, y)], fill="black", width=1)
    y += 14
    d.text((x0, y), "Subtotal:", font=_font(14), fill="black")
    d.text((460, y), f"${subtotal:.2f}", font=_font(14), fill="black")
    y += 22
    d.text((x0, y), "Tax:", font=_font(14), fill="black")
    d.text((460, y), f"${tax:.2f}", font=_font(14), fill="black")
    y += 22
    d.text((x0, y), "Total:", font=_font(16, bold=True), fill="black")
    d.text((460, y), f"${total:.2f}", font=_font(16, bold=True), fill="black")
    y += 30
    d.text((x0, y), f"Payment: {payment}", font=_font(14), fill="black")

    out_path = out_dir / f"{doc_id}.jpg"
    _degrade(img, rng, out_path)

    text_lines = [store, f"Date: {date_str}"]
    for it in lines:
        text_lines.append(f"{it['item']} ${it['price']:.2f}")
    text_lines += [f"Subtotal: ${subtotal:.2f}", f"Tax: ${tax:.2f}", f"Total: ${total:.2f}", f"Payment: {payment}"]

    return DocRecord(
        doc_id=doc_id, doc_type="receipt", modality="image",
        source_path=f"images/{doc_id}.jpg", text="\n".join(text_lines),
        fields={
            "store": store, "date": date_str, "subtotal": round(subtotal, 2),
            "tax": tax, "total": total, "payment_method": payment, "items": lines,
        },
    )


# --------------------------------------------------------------------------- report bodies (shared by image + text reports)
def _make_report_body(rng: random.Random) -> tuple[str, str, str, str, list[str]]:
    dept = rng.choice(DEPARTMENTS)
    author = rng.choice(REPORT_AUTHORS)
    year = rng.choice([2025, 2026])
    month_name = rng.choice(MONTHS)
    title = f"{month_name} {dept} Status Report"

    metric_a = rng.randint(120, 950)
    pct = rng.randint(3, 28)
    direction = rng.choice(["increase", "decrease"])
    n_open = rng.randint(2, 40)
    sentences = [
        f"The {dept} team processed {metric_a} items in {month_name} {year}, "
        f"a {pct}% {direction} compared to the prior month.",
        f"{n_open} items remain open and are expected to close before the next review cycle.",
        f"Report prepared by {author} for the {month_name} {year} operations review.",
    ]
    return title, author, f"{month_name} {year}", dept, sentences


def render_report_image(doc_id: str, rng: random.Random, out_dir: Path) -> DocRecord:
    title, author, period, dept, sentences = _make_report_body(rng)
    year = period.split()[-1]
    month_name = period.split()[0]
    date_str = f"{year}-{MONTHS.index(month_name) + 1:02d}-01"

    img = Image.new("RGB", (IMG_W, 700), "white")
    d = ImageDraw.Draw(img)
    y = MARGIN
    d.text((MARGIN, y), title, font=_font(22, bold=True), fill="black")
    y += 36
    d.text((MARGIN, y), f"Author: {author}    Date: {date_str}", font=_font(14), fill="black")
    y += 34
    d.line([(MARGIN, y), (IMG_W - MARGIN, y)], fill="black", width=1)
    y += 20
    wrap_width = 92
    for sent in sentences:
        for i in range(0, len(sent), wrap_width):
            d.text((MARGIN, y), sent[i:i + wrap_width], font=_font(15), fill="black")
            y += 22
        y += 8

    out_path = out_dir / f"{doc_id}.jpg"
    _degrade(img, rng, out_path)

    text_lines = [title, f"Author: {author}", f"Date: {date_str}", *sentences]
    return DocRecord(
        doc_id=doc_id, doc_type="report_image", modality="image",
        source_path=f"images/{doc_id}.jpg", text="\n".join(text_lines),
        fields={"title": title, "author": author, "date": date_str, "department": dept},
    )


def make_text_report(doc_id: str, rng: random.Random, out_dir: Path) -> DocRecord:
    title, author, period, dept, sentences = _make_report_body(rng)
    year = period.split()[-1]
    month_name = period.split()[0]
    date_str = f"{year}-{MONTHS.index(month_name) + 1:02d}-01"

    body = "\n".join([title, f"Author: {author}", f"Date: {date_str}", "", *sentences])
    out_path = out_dir / f"{doc_id}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")

    return DocRecord(
        doc_id=doc_id, doc_type="text_report", modality="text",
        source_path=f"reports_txt/{doc_id}.txt", text=body,
        fields={"title": title, "author": author, "date": date_str, "department": dept},
    )


# --------------------------------------------------------------------------- expense CSVs
EXPENSE_CATEGORIES = ["Travel", "Office Supplies", "Software", "Meals", "Equipment", "Shipping"]


def make_expense_csv(doc_id: str, rng: random.Random, out_dir: Path) -> DocRecord:
    n_rows = rng.randint(5, 10)
    year = rng.choice([2025, 2026])
    month = rng.randint(1, 9 if year == 2026 else 12)
    dept = rng.choice(DEPARTMENTS)

    rows = []
    total = 0.0
    by_category: dict[str, float] = {}
    for _ in range(n_rows):
        day = rng.randint(1, 28)
        category = rng.choice(EXPENSE_CATEGORIES)
        amount = round(rng.uniform(15, 850), 2)
        desc = f"{category} expense - {dept}"
        rows.append({"date": f"{year:04d}-{month:02d}-{day:02d}", "category": category,
                      "description": desc, "amount": amount})
        total += amount
        by_category[category] = round(by_category.get(category, 0.0) + amount, 2)

    out_path = out_dir / f"{doc_id}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "category", "description", "amount"])
        writer.writeheader()
        writer.writerows(rows)

    top_category = max(by_category, key=by_category.get)
    text = f"Expense report for {dept}, {month:02d}/{year}.\n" + "\n".join(
        f"{r['date']} | {r['category']} | {r['description']} | ${r['amount']:.2f}" for r in rows
    )

    return DocRecord(
        doc_id=doc_id, doc_type="expense_csv", modality="csv",
        source_path=f"tables/{doc_id}.csv", text=text,
        fields={
            "department": dept, "month": f"{year:04d}-{month:02d}", "row_count": n_rows,
            "total_amount": round(total, 2), "top_category": top_category, "by_category": by_category,
        },
    )


# --------------------------------------------------------------------------- eval queries
def build_eval_queries(records: list[DocRecord], rng: random.Random) -> list[dict]:
    queries: list[dict] = []

    invoices = [r for r in records if r.doc_type == "invoice"]
    for r in rng.sample(invoices, min(6, len(invoices))):
        f_ = r.fields
        queries.append({
            "query": f"What is the total amount on invoice {f_['invoice_number']} from {f_['vendor']}?",
            "expected_doc_id": r.doc_id,
        })

    receipts = [r for r in records if r.doc_type == "receipt"]
    for r in rng.sample(receipts, min(5, len(receipts))):
        f_ = r.fields
        queries.append({
            "query": f"Find the {f_['store']} receipt paid by {f_['payment_method']} on {f_['date']}.",
            "expected_doc_id": r.doc_id,
        })

    report_images = [r for r in records if r.doc_type == "report_image"]
    for r in rng.sample(report_images, min(4, len(report_images))):
        f_ = r.fields
        queries.append({
            "query": f"Show me the {f_['department']} status report written by {f_['author']}.",
            "expected_doc_id": r.doc_id,
        })

    csvs = [r for r in records if r.doc_type == "expense_csv"]
    for r in rng.sample(csvs, min(4, len(csvs))):
        f_ = r.fields
        queries.append({
            "query": f"What was the total {f_['department']} expense spend for {f_['month']}?",
            "expected_doc_id": r.doc_id,
        })

    text_reports = [r for r in records if r.doc_type == "text_report"]
    for r in rng.sample(text_reports, min(4, len(text_reports))):
        f_ = r.fields
        queries.append({
            "query": f"What did {f_['author']} report about the {f_['department']} team?",
            "expected_doc_id": r.doc_id,
        })

    rng.shuffle(queries)
    return queries


def generate(
    n_invoices: int = 16, n_receipts: int = 14, n_report_images: int = 10,
    n_expense_csvs: int = 6, n_text_reports: int = 6, seed: int = 42,
    out_dir: Path | None = None,
) -> tuple[list[DocRecord], list[dict]]:
    rng = random.Random(seed)
    out_dir = out_dir or (Path(__file__).resolve().parent / "raw")
    images_dir = out_dir / "images"
    tables_dir = out_dir / "tables"
    reports_txt_dir = out_dir / "reports_txt"

    records: list[DocRecord] = []
    for i in range(n_invoices):
        records.append(render_invoice(f"invoice-{i + 1:03d}", rng, images_dir))
    for i in range(n_receipts):
        records.append(render_receipt(f"receipt-{i + 1:03d}", rng, images_dir))
    for i in range(n_report_images):
        records.append(render_report_image(f"report-img-{i + 1:03d}", rng, images_dir))
    for i in range(n_expense_csvs):
        records.append(make_expense_csv(f"expense-{i + 1:03d}", rng, tables_dir))
    for i in range(n_text_reports):
        records.append(make_text_report(f"text-report-{i + 1:03d}", rng, reports_txt_dir))

    queries = build_eval_queries(records, rng)
    return records, queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-invoices", type=int, default=16)
    parser.add_argument("--n-receipts", type=int, default=14)
    parser.add_argument("--n-report-images", type=int, default=10)
    parser.add_argument("--n-expense-csvs", type=int, default=6)
    parser.add_argument("--n-text-reports", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "raw")
    args = parser.parse_args()

    records, queries = generate(
        n_invoices=args.n_invoices, n_receipts=args.n_receipts, n_report_images=args.n_report_images,
        n_expense_csvs=args.n_expense_csvs, n_text_reports=args.n_text_reports, seed=args.seed,
        out_dir=args.out,
    )

    ground_truth = [
        {"doc_id": r.doc_id, "doc_type": r.doc_type, "modality": r.modality,
         "source_path": r.source_path, "text": r.text, "fields": r.fields}
        for r in records
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    (args.out / "eval_queries.json").write_text(json.dumps(queries, indent=2), encoding="utf-8")

    by_type: dict[str, int] = {}
    for r in records:
        by_type[r.doc_type] = by_type.get(r.doc_type, 0) + 1
    print(f"wrote {len(records)} documents to {args.out}")
    for k, v in by_type.items():
        print(f"  {k}: {v}")
    print(f"wrote {len(queries)} eval queries to {args.out / 'eval_queries.json'}")


if __name__ == "__main__":
    main()
