"""
Pydantic v2 data models and LangChain PydanticOutputParser.
InvoiceExtraction is the canonical output model for all LLM extractions.
All chains return this model — no raw dicts downstream.
"""
from pydantic import BaseModel, Field
try:
    from langchain_core.output_parsers import PydanticOutputParser
except ImportError:
    from langchain.output_parsers import PydanticOutputParser  # type: ignore

from typing import Optional, List, Dict


class LineItem(BaseModel):
    """Single line item extracted from an invoice."""
    line_no:     Optional[int]   = None
    description: Optional[str]   = None
    hsn_code:    Optional[str]   = None
    quantity:    Optional[float] = None
    unit_price:  Optional[float] = None
    gst_rate:    Optional[float] = None
    cgst:        Optional[float] = None
    sgst:        Optional[float] = None
    amount:      Optional[float] = None


class Flag(BaseModel):
    """Validation flag produced by field validator or 3-way matcher."""
    rule:     str
    field:    str
    severity: str    # "error" | "warning" | "info"
    message:  str


class InvoiceExtraction(BaseModel):
    """
    Complete invoice extraction result.
    Fields extracted by LLM via LCEL chain from invoice text.
    Updated by validation chain with match status and flags.
    """
    # ── Identity (set by system) ──────────────────────────────────────────────
    invoice_id: str = ""
    filename:   str = ""

    # ── Vendor ───────────────────────────────────────────────────────────────
    vendor_name:    Optional[str] = Field(None, description="Vendor company name")
    vendor_address: Optional[str] = Field(None, description="Vendor full address")
    vendor_gstin:   Optional[str] = Field(None, description="15-char GSTIN of vendor")
    vendor_pan:     Optional[str] = Field(None, description="PAN of vendor")
    vendor_phone:   Optional[str] = Field(None, description="Phone number")
    vendor_email:   Optional[str] = Field(None, description="Email address")

    # ── Invoice header ────────────────────────────────────────────────────────
    invoice_number:  Optional[str] = Field(None, description="Invoice number e.g. INV-2026-0001")
    invoice_date:    Optional[str] = Field(None, description="Invoice date YYYY-MM-DD")
    due_date:        Optional[str] = Field(None, description="Due date YYYY-MM-DD")
    po_number:       Optional[str] = Field(None, description="PO reference number")
    place_of_supply: Optional[str] = Field(None, description="Place of supply state")
    currency:        Optional[str] = Field("INR", description="Currency code")

    # ── Buyer ─────────────────────────────────────────────────────────────────
    buyer_name:  Optional[str] = Field(None, description="Buyer company name")
    buyer_gstin: Optional[str] = Field(None, description="Buyer GSTIN")

    # ── Financial ─────────────────────────────────────────────────────────────
    subtotal:        Optional[float] = Field(None, description="Subtotal before tax")
    cgst_amount:     Optional[float] = Field(None, description="Central GST")
    sgst_amount:     Optional[float] = Field(None, description="State GST")
    igst_amount:     Optional[float] = Field(None, description="Integrated GST")
    tax_amount:      Optional[float] = Field(None, description="Total tax")
    total_amount:    Optional[float] = Field(None, description="Grand total")
    amount_in_words: Optional[str]   = Field(None, description="Amount in words")

    # ── Line items ────────────────────────────────────────────────────────────
    line_items: List[LineItem] = Field(default_factory=list)

    # ── Bank ──────────────────────────────────────────────────────────────────
    bank_name:    Optional[str] = None
    account_no:   Optional[str] = None
    ifsc_code:    Optional[str] = None
    payment_terms: Optional[str] = None

    # ── Confidence scores (per field, 0.0–1.0) ────────────────────────────────
    confidence: Dict[str, float] = Field(default_factory=dict)

    # ── Validation results (populated by validation chain) ────────────────────
    flags:               List[Flag]    = Field(default_factory=list)
    status:              str           = "pending"     # auto_approved | needs_review | failed
    match_status:        str           = "not_checked"
    matched_po_number:   Optional[str] = None
    po_match_score:      float         = 0.0
    grn_number:          Optional[str] = None
    variance_details:    dict          = Field(default_factory=dict)
    match_recommendation: Optional[str] = None

    # ── Duplicate detection ───────────────────────────────────────────────────
    duplicate_of:         Optional[str]   = None
    duplicate_similarity: Optional[float] = None


def get_invoice_parser() -> PydanticOutputParser:
    """
    Return LangChain PydanticOutputParser for InvoiceExtraction.

    Returns:
        PydanticOutputParser that validates and parses LLM output into InvoiceExtraction
    """
    return PydanticOutputParser(pydantic_object=InvoiceExtraction)
