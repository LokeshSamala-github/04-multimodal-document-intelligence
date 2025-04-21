# Multi-stage build. The builder stage installs Tesseract + the Python deps
# and actually runs the pipeline once (generate -> OCR -> extract -> embed ->
# evaluate) so a broken build fails here, not in production, and the image
# ships with a freshly-verified reports/metrics.json. Tesseract is also
# installed in the final stage because unlike project #1's model file, this
# project's index is rebuilt in memory at API startup (see api.py's lifespan
# handler) -- OCR genuinely has to run again there, not just at build time.

FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app/src
RUN python data/generate_documents.py \
    && python -m doc_intelligence.evaluate --embedding-backend tfidf

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home appuser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    DOCINT_EMBEDDING_BACKEND=tfidf

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "doc_intelligence.api:app", "--host", "0.0.0.0", "--port", "8000"]
