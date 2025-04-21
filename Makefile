.PHONY: install data evaluate serve test lint fmt all clean

install:
	pip install -r requirements-dev.txt

data:
	python data/generate_documents.py

evaluate:
	PYTHONPATH=src python -m doc_intelligence.evaluate --embedding-backend tfidf

serve:
	PYTHONPATH=src uvicorn doc_intelligence.api:app --reload --port 8000

test:
	pytest -v

lint:
	ruff check .

fmt:
	ruff check . --fix

# Full pipeline from scratch, in order.
all: data evaluate test

clean:
	rm -rf data/raw/ocr_cache.json .pytest_cache .ruff_cache **/__pycache__
