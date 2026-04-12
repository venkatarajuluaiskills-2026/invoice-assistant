"""
Multi-strategy PO matching engine.
4-strategy cascade: exact PO number → GSTIN → fuzzy name → amount proximity.
Handles OCR errors and name variations using rapidfuzz.
"""
import re
import json
from pathlib import Path
from typing import Optional, Tuple
from rapidfuzz import fuzz, process
from config import PO_MASTER_PATH, PO_FUZZY_MATCH_THRESHOLD
from guardrails.output_parser import InvoiceExtraction


def load_po_database() -> list:
    """
    Load PO master JSON from disk.

    Returns:
        List of PO dicts; empty list if file not found
    """
    path = Path(PO_MASTER_PATH)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def extract_po_number_from_text(raw_text: str) -> Optional[str]:
    """
    Extract PO reference number from invoice text using regex patterns.
    Handles all common PO reference formats in Indian invoices.

    Args:
        raw_text: Full invoice text (may be redacted)

    Returns:
        First matching PO number string, or None if not found
    """
    patterns = [
        r"(PO-\d{4}-\d{4})",
        r"(?:PO|P\.O\.|Purchase\s+Order|Our\s+Ref)[\s:#-]*([A-Z0-9-]{5,20})",
        r"(?:Ref\.?\s*No\.?)[\s:]+([A-Z0-9-]{5,20})",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def find_po(
    invoice: InvoiceExtraction,
    po_database: list,
) -> Tuple[Optional[dict], float]:
    """
    Find matching PO using 4-strategy cascade.

    Strategy 1: Exact PO number match         → score 1.00
    Strategy 2: Exact GSTIN match             → score 0.90–0.95
    Strategy 3: Fuzzy vendor name match       → score 0.70–0.90
    Strategy 4: Amount proximity (±5%)        → score 0.50

    Args:
        invoice:     InvoiceExtraction with extracted fields
        po_database: List of PO dicts from po_master.json

    Returns:
        (matched_po_dict, confidence_score) or (None, 0.0) if no match
    """
    if not po_database:
        return (None, 0.0)

    # Strategy 1: Exact PO number match
    if invoice.po_number:
        for po in po_database:
            if po["po_number"].upper() == invoice.po_number.upper():
                return (po, 1.0)

    # Strategy 2: Exact GSTIN match
    if invoice.vendor_gstin:
        gstin_matches = [
            p for p in po_database
            if p.get("vendor_gstin", "").upper() == invoice.vendor_gstin.upper()
        ]
        if len(gstin_matches) == 1:
            return (gstin_matches[0], 0.95)
        if len(gstin_matches) > 1 and invoice.vendor_name:
            best = max(
                gstin_matches,
                key=lambda p: fuzz.token_sort_ratio(invoice.vendor_name, p["vendor_name"]),
            )
            return (best, 0.90)

    # Strategy 3: Fuzzy vendor name match
    if invoice.vendor_name:
        po_names = [p["vendor_name"] for p in po_database]
        result = process.extractOne(
            invoice.vendor_name,
            po_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=PO_FUZZY_MATCH_THRESHOLD,
        )
        if result:
            matched = next(
                (p for p in po_database if p["vendor_name"] == result[0]), None
            )
            if matched:
                return (matched, result[1] / 100 * 0.85)

    # Strategy 4: Amount proximity (±5%)
    if invoice.total_amount:
        for po in po_database:
            po_total = po.get("total_amount", 0)
            if po_total > 0:
                variance = abs(invoice.total_amount - po_total) / po_total
                if variance <= 0.05:
                    return (po, 0.50)

    return (None, 0.0)
