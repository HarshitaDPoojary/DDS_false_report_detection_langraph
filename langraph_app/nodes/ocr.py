"""
ocr_node — extract text from image and PDF attachments.

Routing by file type for best quality:
  .pdf / typed forms → SmolDocling-256M (docling) — superior for structured documents
  images            → pytesseract — good for handwriting / photos
  fallback          → pytesseract page-by-page if SmolDocling unavailable

Also produces image_data_urls (base64 data: URLs) for vision_node.

Only runs when has_attachments=True.
"""
from __future__ import annotations

from langraph_app.utils.image_utils import ocr_image_bytes, pil_bytes_from_pdf, to_data_url


def _ocr_pdf_smoldocling(pdf_bytes: bytes) -> str:
    """Try SmolDocling (docling) for structured PDF parsing. Returns Markdown string."""
    try:
        from docling.document_converter import DocumentConverter
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            converter = DocumentConverter()
            result = converter.convert(tmp_path)
            return result.document.export_to_markdown()
        finally:
            os.unlink(tmp_path)
    except Exception:
        return ""


def _ocr_single(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """
    OCR a single file. Returns (ocr_text, data_url).
    Routes PDFs to SmolDocling with pytesseract fallback.
    Routes images to pytesseract.
    """
    name_lower = filename.lower()
    data_url = to_data_url(file_bytes)

    if name_lower.endswith(".pdf"):
        text = _ocr_pdf_smoldocling(file_bytes)
        if not text:
            # fallback: render PDF pages and pytesseract each page
            pages = pil_bytes_from_pdf(file_bytes)[:6]
            text = "\n\n".join(ocr_image_bytes(p) for p in pages if p)
        return text, data_url

    # Image file
    return ocr_image_bytes(file_bytes), data_url


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}

    attachments: list = state.get("attachments", [])
    attachment_names: list = state.get("attachment_names", [])
    names = list(attachment_names) + ["unknown"] * len(attachments)

    ocr_texts: list[str] = []
    image_data_urls: list[str] = []

    for idx, file_bytes in enumerate(attachments):
        if not isinstance(file_bytes, (bytes, bytearray)):
            ocr_texts.append("")
            continue
        filename = names[idx]
        ocr_text, data_url = _ocr_single(bytes(file_bytes), filename)
        ocr_texts.append(ocr_text)
        image_data_urls.append(data_url)

    return {
        "ocr_texts": ocr_texts,
        "image_data_urls": image_data_urls,
    }
