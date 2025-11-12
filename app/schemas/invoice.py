"""Pydantic schemas describing invoice OCR payloads."""

from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class InvoiceParty(BaseModel):
    """Basic vendor/customer metadata extracted from the invoice."""

    name: Optional[str] = None
    address: Optional[str] = None
    tax_id: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class InvoiceLineItem(BaseModel):
    """Single line item in the invoice."""

    description: str
    quantity: float = Field(ge=0)
    unit_price: float = Field(ge=0)
    total: Optional[float] = Field(default=None, ge=0)


class InvoiceTotals(BaseModel):
    """Summaries extracted from the invoice."""

    subtotal: Optional[float] = Field(default=None, ge=0)
    tax: Optional[float] = Field(default=None, ge=0)
    discounts: Optional[float] = Field(default=None)
    total: Optional[float] = Field(default=None, ge=0)
    currency: Optional[str] = "USD"


class InvoiceData(BaseModel):
    """Structured OCR payload."""

    vendor: InvoiceParty = Field(default_factory=InvoiceParty)
    customer: InvoiceParty = Field(default_factory=InvoiceParty)
    invoice_number: Optional[str] = None
    purchase_order: Optional[str] = None
    invoice_date: Optional[date] = None
    due_date: Optional[date] = None
    terms: Optional[str] = None
    notes: Optional[str] = None
    line_items: List[InvoiceLineItem] = Field(default_factory=list)
    totals: InvoiceTotals = Field(default_factory=InvoiceTotals)


class InvoiceOCRResponse(BaseModel):
    """Response returned from the OCR pipeline."""

    schema_version: str = "invoice_v1"
    status: Literal["success", "error"] = "success"
    model: str
    data: InvoiceData
    warnings: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None

