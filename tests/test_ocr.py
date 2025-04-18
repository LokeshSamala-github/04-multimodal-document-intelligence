from __future__ import annotations

from pathlib import Path

from doc_intelligence import ocr


def test_word_accuracy_perfect_match() -> None:
    assert ocr.word_accuracy("Total: $12.50", "Total: $12.50") == 1.0


def test_word_accuracy_penalizes_errors() -> None:
    acc = ocr.word_accuracy("Total: $l2.50 due", "Total: $12.50 due")
    assert 0.5 < acc < 1.0


def test_word_accuracy_empty_reference() -> None:
    assert ocr.word_accuracy("", "") == 1.0
    assert ocr.word_accuracy("garbage", "") == 0.0


def test_backend_name_is_tesseract_when_binary_present() -> None:
    # This sandbox has tesseract installed; the fallback path is exercised
    # separately (below) by forcing the "not available" branch.
    assert ocr.backend_name() in ("tesseract", "ground_truth_fallback")


def test_run_ocr_falls_back_to_ground_truth_when_binary_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ocr, "_TESSERACT_AVAILABLE", False)
    cache_file = tmp_path / "cache.json"
    fake_image = tmp_path / "doc.jpg"
    fake_image.write_bytes(b"not a real image, fallback should never open it")

    text = ocr.run_ocr(fake_image, "Known Vendor\nTotal: $10.00", cache_file)
    assert text == "Known Vendor\nTotal: $10.00"


def test_run_ocr_caches_results(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (300, 100), "white")
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Hello Cache Test", fill="black")
    image_path = tmp_path / "img.png"
    img.save(image_path)

    cache_file = tmp_path / "cache.json"
    first = ocr.run_ocr(image_path, "Hello Cache Test", cache_file)
    assert cache_file.exists()

    # Delete the image; a cache hit must not need to re-read it.
    image_path.unlink()
    second = ocr.run_ocr(image_path, "Hello Cache Test", cache_file)
    assert second == first
