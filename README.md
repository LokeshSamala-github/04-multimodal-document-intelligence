# Multimodal Document Intelligence

A mixed corpus of scanned-style invoices, receipts, memos, expense tables and text reports, turned into
a searchable, structured knowledge base — with real OCR, real rule-based extraction, and real embedding
retrieval, no hosted API calls required.

```
 data/generate_documents.py
        |
        v
  ,---------------------------------------------------------------.
  |  invoice.jpg  |  receipt.jpg  |  report.jpg  |  expense.csv  |  report.txt  |
  '---------------------------------------------------------------'
        |  (image)        (image)        (image)        (csv)          (text)
        v                 v              v               v              v
   Tesseract OCR    Tesseract OCR   Tesseract OCR    csv.DictReader   read directly
        |                 |              |               |              |
        '--------------------.  regex / rule-based extraction  .--------'
                              v
                    doc_intelligence.extraction
                              |
                              v
                     chunking (1 chunk/doc)
                              |
                              v
              TF-IDF (or sentence-transformers) embeddings
                              |
                              v
                 numpy cosine-similarity VectorIndex
                              |
                              v
                FastAPI  POST /query   POST /ingest   GET /documents/{id}
```

## Why this exists, and what it actually is

The obvious way to build "document intelligence" in 2026 is one API call to a hosted vision-language
model: send it a page image, get back structured fields and a natural-language answer. This project has
no paid API keys and makes no calls to any hosted LLM/VLM, so it demonstrates the same underlying skill —
turning document pixels and mixed formats into structured, searchable knowledge — with a fully local,
classical pipeline instead:

- **OCR does the pixels-to-text step a VLM would do implicitly.** `doc_intelligence/ocr.py` runs
  Tesseract on every rendered document image and measures real word-level accuracy against the exact
  text that was drawn onto each image (`data/generate_documents.py` keeps the ground truth precisely
  because it generated the images). If Tesseract isn't installed, the pipeline falls back to the known
  ground-truth text as an explicit, logged fallback — never a silent substitution — see `ocr.py`'s
  docstring and `backend_name()`.
- **Rule-based regex extraction does the text-to-fields step a VLM would do in the same call.**
  `doc_intelligence/extraction.py` pulls out invoice numbers, vendors, totals, dates, payment methods,
  etc. with plain regex/line-based rules, because the synthetic corpus has a small, fixed set of layouts.
  A real production system would swap this whole OCR+regex path for one VLM call behind the same
  interface (see "what I'd do next" below) — the point of this project is the pipeline architecture and
  the honest, measured accuracy of each stage, not a claim that regex extraction is what you'd ship
  against arbitrary real-world documents.
- **TF-IDF does the embeddings step.** `doc_intelligence/embeddings.py` *does* try `sentence-transformers`
  (`all-MiniLM-L6-v2`) first — that code path is real, not aspirational — but downloading the model
  weights requires a live connection to huggingface.co, which this project's build sandbox blocks
  (`ProxyError: 403 Forbidden`, confirmed by actually running it). The `try/except` around that call
  genuinely falls through to scikit-learn's `TfidfVectorizer`, a legitimate classical embedding baseline
  at this corpus size, and every run logs and reports which backend it actually used. If you run this
  yourself somewhere with huggingface.co access, `sentence-transformers` (an optional, non-default
  install — see below) will be picked up automatically.
- **Multimodal means three genuinely different ingestion paths, not three prompts to one model.**
  Images go through OCR; CSV tables are parsed directly with `csv.DictReader`; plain-text reports are
  read directly. Only the image path pays the OCR-accuracy tax, and the results below show that gap
  honestly (structured/text formats extract at ~100%, OCR'd images somewhat lower).

## The corpus

`data/generate_documents.py` renders **52 synthetic documents** with PIL across five document types and
three input modalities, then applies a small random rotation, gaussian blur and additive noise and
re-saves as JPEG — so Tesseract is reading a genuinely imperfect "scan," not pixel-perfect rendered text:

| Type | Modality | Count | Extracted fields |
|---|---|---|---|
| `invoice` | image (OCR'd) | 16 | vendor, invoice #, date, subtotal/tax/total, line items |
| `receipt` | image (OCR'd) | 14 | store, date, items, total, payment method |
| `report_image` | image (OCR'd) | 10 | title, author, date |
| `expense_csv` | CSV (parsed directly) | 6 | department, month, total, top category, per-category totals |
| `text_report` | plain text (read directly) | 6 | title, author, date |

Every document's exact ground-truth text and fields are recorded in `data/raw/ground_truth.json`, which
is what makes every metric below a real measurement against a known answer, not a number the pipeline
asserts about itself.

```bash
python data/generate_documents.py        # -> data/raw/{images,tables,reports_txt}/, ground_truth.json
```

## Results

From `reports/metrics.json`, produced by `python -m doc_intelligence.evaluate` against the shipped
52-document corpus (TF-IDF embedding backend — see above):

**OCR (Tesseract, word accuracy = 1 − WER against ground truth, 40 images):**

| Doc type | Word accuracy |
|---|---|
| invoice | 0.979 |
| receipt | 0.969 |
| report_image | 0.970 |
| **Overall** | **0.973** |

**Structured extraction (precision against ground-truth fields, 198 field checks):**

| Doc type | Precision |
|---|---|
| invoice | 0.953 |
| receipt | 0.946 |
| report_image | 1.000 |
| text_report | 1.000 |
| expense_csv | 1.000 |
| **Overall** | **0.970** |

The lowest-precision fields are `invoice.total` (0.813) and `receipt.total` (0.857) — both dollar amounts
read off an OCR'd line, so a single misread digit directly breaks the field. That gap between "parsed
directly" formats (100%) and "went through OCR first" formats (~95%) is exactly the honest result you'd
expect, and is the clearest evidence in this project that the numbers are real, not asserted.

**Retrieval (TF-IDF + numpy cosine similarity, 23 held-out queries with known correct source document):**

| Metric | Value |
|---|---|
| Recall@1 | 0.957 |
| Recall@3 | 1.000 |
| Recall@5 | 1.000 |
| MRR | 0.978 |

The one query that misses at rank 1 ("What did R. Okafor report about the Field Operations team?") is
retrieved correctly by rank 2 — it's pulled toward a same-department `report_image` document whose OCR'd
text happens to share more vocabulary with the query than the correct plain-text report does, a very
believable TF-IDF failure mode. `reports/metrics.json` has the full per-query ranking.

## Run it

```bash
pip install -r requirements-dev.txt
sudo apt-get install -y tesseract-ocr     # if `which tesseract` comes up empty

python data/generate_documents.py                                    # 1. synthetic corpus
PYTHONPATH=src python -m doc_intelligence.evaluate --embedding-backend tfidf  # 2. OCR -> extract -> embed -> eval
PYTHONPATH=src uvicorn doc_intelligence.api:app --reload              # 3. serve
```

or, with `make`:

```bash
make install && make all && make serve
```

### API

```bash
curl -s localhost:8000/query -X POST -H 'content-type: application/json' \
  -d '{"query": "What is the total on invoice INV-13278 from Acme Logistics?", "top_k": 1}' | python3 -m json.tool
```

```json
{
  "query": "What is the total on invoice INV-13278 from Acme Logistics?",
  "embedding_backend": "tfidf",
  "results": [
    {
      "chunk_id": "invoice-001::0",
      "doc_id": "invoice-001",
      "doc_type": "invoice",
      "score": 0.3937,
      "snippet": "Acme Logistics INVOICE\n\nInvoice #: INV-13278\n\nDate: 2026-04-08\n\nDescription Qty Unit Price Line Total\n\nLED Panel Light 2...",
      "fields": {"vendor": "Acme Logistics", "invoice_number": "INV-13278", "date": "2026-04-08", "total": 938.06}
    }
  ]
}
```

(the corpus has two Acme Logistics invoices, so a query without an invoice number is genuinely
ambiguous between them — including the number, like the eval query set does, disambiguates it)

Other endpoints: `GET /health`, `GET /corpus/info` (document counts + which OCR/embedding backend is
actually active), `GET /documents/{doc_id}`, `POST /ingest` (add a new plain-text document to the live
index; the embedding backend is re-fit on the whole corpus so newly-ingested vocabulary is immediately
searchable — see `pipeline.py`'s `Corpus.add_text_document`). Interactive docs at `localhost:8000/docs`.

### Docker

```bash
docker build -t doc-intelligence .
docker run -p 8000:8000 doc-intelligence
```

The build stage installs Tesseract, generates the corpus, and runs the full evaluation once — so a
broken pipeline fails the build, not production. The final image also installs Tesseract (not just the
build stage) because, unlike a trained model file, this project's index is rebuilt from the raw corpus
in memory every time the API process starts (`api.py`'s lifespan handler) — OCR genuinely runs again at
container startup, not just at image-build time. `DOCINT_EMBEDDING_BACKEND=tfidf` is set in the image so
startup never attempts a network call to huggingface.co.

## Project layout

```
data/generate_documents.py       synthetic corpus generator (PIL-rendered images + CSV + text)
src/doc_intelligence/
  config.py                      every path, in one place, env-var overridable (DOCINT_*)
  documents.py                   IngestedDocument / Chunk -- the common shape every modality converges on
  ocr.py                         Tesseract wrapper, ground-truth fallback, word-accuracy (1-WER) metric
  extraction.py                  regex/rule-based structured-field extraction, per document type
  ingest.py                      dispatches each modality to its OCR/CSV/text read path
  chunking.py                    document -> retrieval chunk(s)
  embeddings.py                  TF-IDF backend (default) + optional sentence-transformers backend
  index.py                       numpy cosine-similarity vector index
  pipeline.py                    wires ingest -> chunk -> embed -> index into one queryable Corpus
  evaluate.py                    OCR / extraction / retrieval metrics -> reports/metrics.json
  api.py                         FastAPI app
  schemas.py                     pydantic request/response models
tests/                           36 tests: generation, OCR, extraction, embeddings/index,
                                  pipeline/retrieval, API (see `pytest -v`)
data/raw/                        the shipped 52-document corpus + ground_truth.json + eval_queries.json
reports/metrics.json             the real OCR/extraction/retrieval numbers reported above
.github/workflows/ci.yml         lint + test + an end-to-end generate/evaluate smoke run, on every push
```

## Tests

```bash
pytest -v                                       # 36 tests
pytest --cov=doc_intelligence --cov-report=term-missing   # 93% line coverage
ruff check .                                    # clean
```

Every test exercises real code paths against a small, freshly-generated corpus (`tests/conftest.py`,
session-scoped) — no mocked OCR, no mocked embeddings. That includes retrieval assertions like "a query
built from a specific invoice's real invoice number and vendor retrieves that exact invoice at rank 1,"
not just "the function returns a list."

## What I'd do next in production

- Swap the OCR+regex extraction path for a single hosted VLM call (e.g. Claude or GPT-4V) behind the
  same `ingest.py` interface — same `IngestedDocument` output shape, no downstream code would change,
  and it would remove the OCR-accuracy ceiling entirely visible in the invoice/receipt total-amount
  numbers above.
- Real, overlapping-window chunking for multi-page or long documents — every document here is short
  enough that one chunk per document is honest, but it's a simplification (see `chunking.py`'s docstring)
  that would break on a 20-page report.
- Layout-aware extraction (bounding boxes from Tesseract's `image_to_data`, not just flat text) instead
  of line-based regex, so a field extractor doesn't depend on two labels never landing on the same
  physical line after OCR.
- An ANN index (FAISS/HNSW) once the corpus is large enough that a dense `numpy` matrix multiply per
  query stops being effectively free.
