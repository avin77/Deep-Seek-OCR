"""Streamlit UI for sending invoices/bills to the DeepSeek OCR FastAPI service."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import httpx
from pdf2image import convert_from_bytes
from PIL import Image
import streamlit as st

DEFAULT_API_BASE = "http://localhost:7861"


def _render_pdf_preview(pages: List[Image.Image], max_pages: int = 3) -> None:
    """Show up to `max_pages` previews of the uploaded PDF."""

    if not pages:
        st.warning("PDF contained no renderable pages.")
        return

    cols = st.columns(min(max_pages, len(pages)))
    for idx, (img, col) in enumerate(zip(pages[:max_pages], cols), start=1):
        with col:
            st.image(img, caption=f"Page {idx}", use_column_width=True)


def _call_ocr_api(
    api_base: str,
    file_name: str,
    payload_bytes: bytes,
    mime_type: str,
) -> Dict[str, Any]:
    """Send the prepared payload to the FastAPI /ocr endpoint."""

    files = {"file": (file_name, payload_bytes, mime_type)}
    with httpx.Client(base_url=api_base, timeout=180) as client:
        response = client.post("/ocr", files=files)
        response.raise_for_status()
        return response.json()


def main() -> None:
    st.set_page_config(page_title="Local DeepSeek Invoice OCR", layout="wide")
    st.title("Local DeepSeek Invoice OCR")

    api_base = st.sidebar.text_input("DeepSeek FastAPI base URL", DEFAULT_API_BASE)
    st.sidebar.caption(
        "Ensure the `deepseek-ocr` Docker container is running and exposes port 7861."
    )

    uploaded_file = st.file_uploader(
        "Upload a scanned invoice or bill (PDF or image)",
        type=["pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff"],
    )

    if not uploaded_file:
        st.info("Upload a document to begin.")
        return

    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name or "upload"
    mime_type = uploaded_file.type or "application/octet-stream"

    payload_bytes = file_bytes
    payload_name = file_name
    payload_mime = mime_type

    is_pdf = file_name.lower().endswith(".pdf") or mime_type == "application/pdf"

    if is_pdf:
        try:
            pages = convert_from_bytes(file_bytes, dpi=180)
        except Exception as exc:  # pragma: no cover - UI feedback only
            st.error(f"Could not render PDF: {exc}")
            return

        if not pages:
            st.error("PDF contained no renderable pages.")
            return

        with st.expander("Preview first pages", expanded=False):
            _render_pdf_preview(pages)

        first_page = pages[0]
        st.image(first_page, caption="Page 1 (sent to DeepSeek)", use_column_width=True)

        buffer = io.BytesIO()
        first_page.save(buffer, format="PNG")
        payload_bytes = buffer.getvalue()
        payload_name = f"{Path(file_name).stem}.png"
        payload_mime = "image/png"
    else:
        st.image(file_bytes, caption=file_name, use_column_width=True)
        if not payload_mime.startswith("image/"):
            payload_mime = "application/octet-stream"

    if st.button("Extract text with DeepSeek-OCR"):
        with st.spinner("Calling FastAPI /ocr ..."):
            try:
                response = _call_ocr_api(
                    api_base=api_base,
                    file_name=payload_name,
                    payload_bytes=payload_bytes,
                    mime_type=payload_mime,
                )
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text
                st.error(f"OCR request failed: {exc}\n{detail}")
            except httpx.HTTPError as exc:
                st.error(f"Network error: {exc}")
            else:
                text = (response.get("text") or "").strip()
                if text:
                    st.success("OCR completed")
                    st.text_area("Extracted Markdown/Text", text, height=400)
                else:
                    st.warning("OCR completed but returned empty text.")
                st.caption("Raw response")
                st.json(response)


if __name__ == "__main__":
    main()
