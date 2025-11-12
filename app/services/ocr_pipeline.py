"""High-level pipeline that converts PDFs to JSON invoices."""

from __future__ import annotations

import logging

from app.core.config import Settings, get_settings
from app.schemas.invoice import InvoiceOCRResponse
from app.services.ocr_client import DeepSeekOCRClient, OCRClientError
from utils.pdf_rasterizer import RasterizedPDF, pdf_bytes_to_images

logger = logging.getLogger(__name__)


class InvoiceOCRPipeline:
    """Glue layer between PDF rasterization and the DeepSeek OCR client."""

    def __init__(
        self,
        ocr_client: DeepSeekOCRClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.ocr_client = ocr_client or DeepSeekOCRClient(settings=self.settings)

    async def run(self, pdf_bytes: bytes) -> InvoiceOCRResponse:
        rasterized: RasterizedPDF | None = None
        try:
            rasterized = pdf_bytes_to_images(
                pdf_bytes,
                dpi=self.settings.pdf_raster_dpi,
            )
            result = await self.ocr_client.extract_invoice(rasterized.image_paths)
            return result
        finally:
            if rasterized:
                try:
                    rasterized.cleanup()
                except Exception:  # pragma: no cover - best effort cleanup
                    logger.debug("Failed to cleanup temp files", exc_info=True)
