"""OCR layer.

Honest framing (see README for the full version): a real production system
for this problem would call a hosted vision-language model (GPT-4V, Claude
with vision, Gemini) on the raw document image and get structured fields
back directly. This project has no paid API keys, so it demonstrates the
same *document-intelligence* skill -- turning pixels into structured,
searchable knowledge -- with a classical, fully local pipeline: Tesseract
OCR for pixels-to-text, then rule-based extraction for text-to-fields
(extraction.py). That is a real, working, locally-runnable substitute for
the OCR+understanding half of what a VLM call would do, and it produces
genuine, measurable accuracy numbers instead of a mocked API response.

If the `tesseract` binary is not on PATH (it is not installed by default in
a bare `python:3.11-slim` image, and CI must apt-get it -- see
.github/workflows/ci.yml and Dockerfile), this module falls back to using
the known ground-truth text and flags that fallback explicitly rather than
silently faking OCR output. `backend_name()` reports which path is active
so tests, `evaluate.py`, and the API's `/corpus/info` endpoint can say so
honestly.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_TESSERACT_AVAILABLE = shutil.which("tesseract") is not None
_cache: dict[str, str] = {}
_cache_loaded_from: Path | None = None


def backend_name() -> str:
    return "tesseract" if _TESSERACT_AVAILABLE else "ground_truth_fallback"


def _load_cache(cache_file: Path) -> None:
    global _cache, _cache_loaded_from
    if _cache_loaded_from == cache_file:
        return
    _cache = {}
    if cache_file.exists():
        try:
            _cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _cache = {}
    _cache_loaded_from = cache_file


def _save_cache(cache_file: Path) -> None:
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(_cache, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("could not write OCR cache to %s", cache_file)


def run_ocr(image_path: Path, ground_truth_text: str, cache_file: Path) -> str:
    """Return OCR'd text for `image_path`, caching by absolute path.

    Falls back to `ground_truth_text` -- with a loud warning, not a silent
    substitution -- if the tesseract binary isn't installed at all.
    """
    _load_cache(cache_file)
    key = str(image_path.resolve())
    if key in _cache:
        return _cache[key]

    if not _TESSERACT_AVAILABLE:
        logger.warning(
            "tesseract binary not found on PATH; using ground-truth text as an explicit "
            "OCR fallback for %s (see ocr.py docstring)",
            image_path.name,
        )
        _cache[key] = ground_truth_text
        _save_cache(cache_file)
        return ground_truth_text

    import pytesseract
    from PIL import Image

    with Image.open(image_path) as img:
        text = pytesseract.image_to_string(img)
    text = text.strip()
    _cache[key] = text
    _save_cache(cache_file)
    return text


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Classic O(len(a)*len(b)) edit distance over token sequences, used for
    word error rate. No extra dependency needed at this corpus scale."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def word_accuracy(hypothesis: str, reference: str) -> float:
    """1 - WER (word error rate), clipped to [0, 1]. 1.0 means a perfect
    transcription; this is the headline "OCR accuracy" metric reported in
    reports/metrics.json."""
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not ref_tokens:
        return 1.0 if not hyp_tokens else 0.0
    distance = _levenshtein(hyp_tokens, ref_tokens)
    wer = distance / len(ref_tokens)
    return max(0.0, 1.0 - wer)
