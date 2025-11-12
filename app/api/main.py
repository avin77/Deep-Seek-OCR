
"""FastAPI entry point for the local invoice OCR system."""

from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.schemas.invoice import InvoiceOCRResponse
from app.services.ocr_pipeline import InvoiceOCRPipeline
from faiss_service.indexer import InvoiceFaissIndex

logger = logging.getLogger(__name__)

app: FastAPI | None = None


class FaissQuery(BaseModel):
    query_text: str
    top_k: int = Field(default=3, ge=1, le=20)


class FaissIndexPayload(BaseModel):
    invoice: InvoiceOCRResponse


def get_app() -> FastAPI:
    global app
    if app is not None:
        return app

    settings = get_settings()
    pipeline = InvoiceOCRPipeline(settings=settings)
    faiss_index = InvoiceFaissIndex()

    api = FastAPI(title="Local Invoice OCR", version="0.1.0")

    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @api.post("/ocr/invoice")
    async def ocr_invoice(
        index_result: bool = False,
        file: UploadFile = File(...),
    ) -> dict:
        if file.content_type not in {"application/pdf", "application/octet-stream"}:
            raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        try:
            result = await pipeline.run(pdf_bytes)
        except Exception as exc:  # pragma: no cover - surfaced to client
            logger.exception("OCR processing failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if index_result:
            faiss_index.add_invoices([result])

        return result.model_dump()

    @api.post("/faiss/index")
    async def faiss_index_endpoint(payload: FaissIndexPayload) -> dict:
        faiss_index.add_invoices([payload.invoice])
        return {"status": "indexed", "count": len(faiss_index._store)}

    @api.post("/faiss/query")
    async def faiss_query_endpoint(query: FaissQuery) -> dict:
        results = faiss_index.query(query.query_text, top_k=query.top_k)
        return {"results": results}

    @api.delete("/faiss/index")
    async def faiss_drop_endpoint() -> dict:
        faiss_index.drop()
        return {"status": "cleared"}

    globals()["app"] = api
    return api


app = get_app()
