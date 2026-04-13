"""
image_utils — pure image helper functions with no heavy dependencies.

Isolated here so new nodes never need to import legacy/gpt_incident_agent.py,
which instantiates an OpenAI client at module level and requires OPENAI_API_KEY.
"""
from __future__ import annotations

import base64
import io


def to_data_url(img_bytes: bytes, fallback_mime: str = "image/png") -> str:
    """Encode raw image bytes as a base64 data: URL, sniffing MIME type."""
    magic = img_bytes[:4]
    mime = fallback_mime
    if magic.startswith(b"\xff\xd8"):
        mime = "image/jpeg"
    elif magic.startswith(b"\x89PNG"):
        mime = "image/png"
    elif magic.startswith(b"GIF8"):
        mime = "image/gif"
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def ocr_image_bytes(img_bytes: bytes) -> str:
    """Run pytesseract OCR on raw image bytes. Returns empty string if unavailable."""
    try:
        from PIL import Image  # type: ignore[import]
        import pytesseract     # type: ignore[import]
        im = Image.open(io.BytesIO(img_bytes))
        return pytesseract.image_to_string(im)
    except Exception:
        return ""


def pil_bytes_from_pdf(pdf_bytes: bytes) -> list[bytes]:
    """Render each PDF page to PNG bytes using pdf2image. Returns [] if unavailable."""
    try:
        from pdf2image import convert_from_bytes  # type: ignore[import]
        pages = convert_from_bytes(pdf_bytes)
        result: list[bytes] = []
        for page in pages:
            buf = io.BytesIO()
            page.save(buf, format="PNG")
            result.append(buf.getvalue())
        return result
    except Exception:
        return []
