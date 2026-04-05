from io import BytesIO

from PIL import Image

from app.services import extraction


def _make_test_image_bytes() -> bytes:
    image = Image.new("RGB", (60, 30), color="white")
    bio = BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()


def test_detect_file_type_variants():
    assert extraction.detect_file_type("a.pdf", None) == "pdf"
    assert extraction.detect_file_type("a.docx", None) == "docx"
    assert extraction.detect_file_type("a.png", None) == "image"
    assert extraction.detect_file_type("a.txt", None) == "text"


def test_extract_image_text_without_tesseract(monkeypatch):
    monkeypatch.setattr(extraction, "_safe_import_tesseract", lambda: None)
    raw = _make_test_image_bytes()
    result = extraction.extract_image_text(raw)
    assert result.file_type == "image"
    assert "pytesseract_not_available" in result.warnings


def test_extract_plain_text_file():
    raw = b"Quarterly report for Acme Corp."
    result = extraction.extract_text(raw, filename="report.txt", content_type="text/plain")
    assert result.file_type == "text"
    assert "Acme" in result.text
