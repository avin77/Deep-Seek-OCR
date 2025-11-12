"""Client wrapper for calling DeepSeek-OCR via a local vLLM endpoint."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import httpx
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.schemas.invoice import InvoiceOCRResponse

logger = logging.getLogger(__name__)


class OCRClientError(RuntimeError):
    """Raised when the OCR model response cannot be parsed."""


class DeepSeekOCRClient:
    """Lightweight async client that enforces strict JSON responses."""

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=self.settings.deepseek_base_url,
            timeout=self.settings.request_timeout_seconds,
            headers={"Authorization": f"Bearer {self.settings.deepseek_api_key}"},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def extract_invoice(self, image_paths: Sequence[Path]) -> InvoiceOCRResponse:
        """Send rasterized invoice images to DeepSeek-OCR and parse JSON."""

        if not image_paths:
            raise ValueError("Expected at least one image for OCR")

        encoded_images = [self._encode_image(path) for path in image_paths]

        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_request_retries + 1):
            try:
                completion = await self._post_completion(encoded_images)
                payload = self._parse_completion(completion)
                return payload
            except (httpx.HTTPError, OCRClientError, ValidationError) as exc:  # pragma: no cover - retried
                last_error = exc
                logger.warning("DeepSeek OCR attempt %s failed: %s", attempt, exc)

        raise OCRClientError("OCR failed after retries") from last_error

    async def _post_completion(self, encoded_images: Sequence[str]) -> dict:
        body = {
            "model": self.settings.deepseek_model,
            "messages": self._build_messages(encoded_images),
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

        response = await self._client.post("chat/completions", json=body)
        response.raise_for_status()
        return response.json()

    def _build_messages(self, encoded_images: Sequence[str]) -> List[dict]:
        schema_hint = json.dumps(
            {
                "schema_version": "invoice_v1",
                "status": "success",
                "model": self.settings.deepseek_model,
                "data": {
                    "vendor": {"name": "", "address": "", "tax_id": ""},
                    "customer": {"name": ""},
                    "invoice_number": "",
                    "purchase_order": "",
                    "invoice_date": "YYYY-MM-DD",
                    "due_date": "YYYY-MM-DD",
                    "terms": "",
                    "notes": "",
                    "line_items": [
                        {
                            "description": "",
                            "quantity": 0,
                            "unit_price": 0,
                            "total": 0,
                        }
                    ],
                    "totals": {
                        "subtotal": 0,
                        "tax": 0,
                        "discounts": 0,
                        "total": 0,
                        "currency": "USD",
                    },
                },
                "warnings": [],
            }
        )

        system_prompt = (
            "You are DeepSeek-OCR fine-tuned for invoices. Always output valid JSON "
            "matching the provided schema and NEVER include prose."
        )
        user_content: List[dict] = [
            {
                "type": "text",
                "text": (
                    "Extract structured invoice data. Respond with STRICT JSON only. "
                    f"Schema example: {schema_hint}."
                ),
            }
        ]

        for encoded in encoded_images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _parse_completion(self, completion: dict) -> InvoiceOCRResponse:
        try:
            content = completion["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
            raise OCRClientError("Malformed completion payload") from exc

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise OCRClientError("Model did not return valid JSON") from exc

        data.setdefault("schema_version", "invoice_v1")
        data.setdefault("status", "success")
        data.setdefault("model", self.settings.deepseek_model)
        data.setdefault("warnings", [])
        data.setdefault("data", {})
        data["data"].setdefault("totals", {})

        return InvoiceOCRResponse.model_validate(data)

    @staticmethod
    def _encode_image(path: Path) -> str:
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return b64
