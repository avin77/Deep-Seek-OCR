"""Optional FAISS helper for semantic search over OCR results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.schemas.invoice import InvoiceOCRResponse


@dataclass
class IndexedInvoice:
    """Stores serialized invoice data alongside metadata."""

    payload: Dict
    raw_text: str


class InvoiceFaissIndex:
    """Thin wrapper around FAISS IndexFlatIP with sentence-transformer embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._embedder: SentenceTransformer | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._dimension: int | None = None
        self._store: List[IndexedInvoice] = []

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
            self._dimension = dim

    def add_invoices(self, invoices: Sequence[InvoiceOCRResponse]) -> None:
        if not invoices:
            return

        texts = [json.dumps(invoice.model_dump(), ensure_ascii=False) for invoice in invoices]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        dim = embeddings.shape[1]
        self._ensure_index(dim)

        self._index.add(embeddings)
        for invoice, blob in zip(invoices, texts):
            self._store.append(IndexedInvoice(payload=invoice.model_dump(), raw_text=blob))

    def query(self, query_text: str, top_k: int = 3) -> List[Dict]:
        if self._index is None:
            return []

        embedding = self.embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self._index.search(embedding, top_k)

        results: List[Dict] = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self._store):
                continue
            entry = self._store[idx]
            results.append({"score": float(score), "invoice": entry.payload})
        return results

    def drop(self) -> None:
        self._index = None
        self._dimension = None
        self._store.clear()
