"""
3-Way Invoice Matching: Invoice ↔ PO ↔ GRN.
This is the core AP automation logic.
Matches invoice amounts/quantities to PO and confirms goods receipt via GRN.
Generates detailed flags for every variance found.
"""
import json
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel
from config import (
    GRN_MASTER_PATH, PRICE_TOLERANCE_PCT, AMOUNT_ROUNDING_INR,
)
from guardrails.output_parser import InvoiceExtraction
from matching.po_matcher import find_po, load_po_database


class MatchResult(BaseModel):
    """Result of 3-way matching for one invoice."""
    po_number:        Optional[str] = None
    po_match_score:   float         = 0.0
    grn_number:       Optional[str] = None
    match_status:     str           = "not_checked"
    variance_details: dict          = {}
    flags:            list          = []
    recommendation:   Optional[str] = None


def load_grn_database() -> list:
    """
    Load GRN master JSON from disk.

    Returns:
        List of GRN dicts; empty list if file not found
    """
    path = Path(GRN_MASTER_PATH)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def find_grn(po_number: str, grn_database: list) -> Optional[dict]:
    """
    Find GRN for a given PO number.

    Args:
        po_number:    PO number to search for
        grn_database: List of GRN dicts from grn_master.json

    Returns:
        First matching GRN dict, or None if not found
    """
    if not po_number:
        return None
    for grn in grn_database:
        if grn.get("po_number", "").upper() == po_number.upper():
            return grn
    return None


def run_three_way_match(
    invoice: InvoiceExtraction,
    po_database: list,
    grn_database: list,
) -> MatchResult:
    """
    Perform 3-way matching: Invoice ↔ PO ↔ GRN.

    Checks (in order):
    1. Find matching PO (4-strategy cascade)
    2. Check PO status (reject if fully_invoiced)
    3. Validate total amount within 3% tolerance
    4. Validate line-item prices and quantities (zero tolerance on overbilling)
    5. Find matching GRN and check receipt status
    6. Flag partial receipt, rejection, or missing GRN
    7. Return final match_status and recommendation

    Args:
        invoice:      InvoiceExtraction with extracted fields
        po_database:  List of PO dicts from po_master.json
        grn_database: List of GRN dicts from grn_master.json

    Returns:
        MatchResult with match_status, variance_details, flags, recommendation
    """
    result = MatchResult()
    flags: List[dict] = []

    # ── Step 1: Find PO ───────────────────────────────────────────────────────
    matched_po, po_score = find_po(invoice, po_database)
    result.po_match_score = po_score

    if not matched_po:
        result.match_status = "no_po_match"
        flags.append({
            "rule":     "po_not_found",
            "field":    "po_number",
            "severity": "error",
            "message":  (
                f"No matching PO found for invoice {invoice.invoice_number}. "
                f"PO reference in invoice: {invoice.po_number or 'not detected'}. "
                f"Vendor: {invoice.vendor_name or 'unknown'}."
            ),
        })
        result.flags = flags
        result.recommendation = (
            "Hold invoice — no matching PO found. "
            "Contact vendor for the correct PO reference number."
        )
        return result

    result.po_number = matched_po["po_number"]

    # ── Step 2: PO status check ───────────────────────────────────────────────
    if matched_po.get("status") == "fully_invoiced":
        result.match_status = "po_closed"
        flags.append({
            "rule":     "po_already_closed",
            "field":    "po_number",
            "severity": "error",
            "message":  (
                f"PO {matched_po['po_number']} is already fully invoiced. "
                f"This invoice may be a duplicate billing attempt."
            ),
        })
        result.flags = flags
        result.recommendation = (
            "Reject invoice — PO is closed. "
            "Investigate for duplicate billing before any payment."
        )
        return result

    # ── Step 3: Total amount variance ─────────────────────────────────────────
    variance_details: dict = {}
    po_total   = matched_po.get("total_amount", 0)
    inv_total  = invoice.total_amount or 0
    tolerance  = po_total * PRICE_TOLERANCE_PCT

    amount_variance     = abs(inv_total - po_total)
    amount_variance_pct = (amount_variance / po_total * 100) if po_total else 0

    variance_details["total_amount"] = {
        "po_value":         po_total,
        "invoice_value":    inv_total,
        "variance_amount":  amount_variance,
        "variance_pct":     round(amount_variance_pct, 2),
        "within_tolerance": amount_variance <= tolerance,
    }

    if amount_variance > tolerance:
        flags.append({
            "rule":     "amount_exceeds_po_tolerance",
            "field":    "total_amount",
            "severity": "error",
            "message":  (
                f"Invoice total ₹{inv_total:,.2f} vs PO total ₹{po_total:,.2f}. "
                f"Variance: ₹{amount_variance:,.2f} ({amount_variance_pct:.1f}%). "
                f"Allowed tolerance: 3%."
            ),
        })

    # ── Step 4: Line item validation ──────────────────────────────────────────
    if invoice.line_items and matched_po.get("line_items"):
        for i, inv_item in enumerate(invoice.line_items):
            if i >= len(matched_po["line_items"]):
                break
            po_item = matched_po["line_items"][i]

            # Price variance check
            if inv_item.unit_price and po_item.get("unit_price"):
                price_var     = abs(inv_item.unit_price - po_item["unit_price"])
                price_var_pct = price_var / po_item["unit_price"] * 100
                within        = price_var / po_item["unit_price"] <= PRICE_TOLERANCE_PCT
                variance_details[f"line_{i+1}_unit_price"] = {
                    "po_value":         po_item["unit_price"],
                    "invoice_value":    inv_item.unit_price,
                    "variance_pct":     round(price_var_pct, 2),
                    "within_tolerance": within,
                }
                if not within:
                    flags.append({
                        "rule":     "unit_price_variance",
                        "field":    f"line_item_{i+1}",
                        "severity": "warning",
                        "message":  (
                            f"Line {i+1} unit price ₹{inv_item.unit_price:,.2f} "
                            f"vs PO ₹{po_item['unit_price']:,.2f} "
                            f"(+{price_var_pct:.1f}% over 3% tolerance)"
                        ),
                    })

            # Quantity overbilling — zero tolerance
            if inv_item.quantity and po_item.get("quantity"):
                qty_var_pct = (
                    (inv_item.quantity - po_item["quantity"]) / po_item["quantity"] * 100
                )
                overbilled = inv_item.quantity > po_item["quantity"]
                variance_details[f"line_{i+1}_quantity"] = {
                    "po_value":      po_item["quantity"],
                    "invoice_value": inv_item.quantity,
                    "variance_pct":  round(qty_var_pct, 2),
                    "overbilled":    overbilled,
                }
                if overbilled:
                    flags.append({
                        "rule":     "quantity_overbilled",
                        "field":    f"line_item_{i+1}",
                        "severity": "error",
                        "message":  (
                            f"Line {i+1} quantity overbilled: "
                            f"Invoice qty {inv_item.quantity} > PO qty {po_item['quantity']} "
                            f"(+{qty_var_pct:.1f}%). Zero tolerance on overbilling."
                        ),
                    })

    # ── Step 5: GRN lookup ────────────────────────────────────────────────────
    grn = find_grn(result.po_number, grn_database)

    if not grn:
        result.match_status = "grn_pending"
        flags.append({
            "rule":     "grn_not_found",
            "field":    "grn_number",
            "severity": "warning",
            "message":  (
                f"No GRN found for PO {result.po_number}. "
                f"Goods may not yet be received or GRN not yet entered in the system."
            ),
        })
        result.recommendation = (
            "Hold invoice — await GRN confirmation from the warehouse team."
        )
    else:
        result.grn_number = grn["grn_number"]
        grn_status        = grn.get("status", "")

        if grn_status == "rejected":
            reason = grn.get("rejection_reason", "Goods were not accepted by the warehouse.")
            result.match_status = "grn_rejected"
            flags.append({
                "rule":     "grn_rejected",
                "field":    "grn_number",
                "severity": "error",
                "message":  (
                    f"GRN {grn['grn_number']} for PO {result.po_number} is REJECTED. "
                    f"Reason: {reason} Invoice cannot be paid."
                ),
            })
            result.recommendation = (
                f"Reject invoice — GRN shows goods were rejected. {reason}"
            )

        elif grn_status == "partially_received":
            result.match_status = "partial_grn"
            flags.append({
                "rule":     "partial_goods_receipt",
                "field":    "grn_number",
                "severity": "warning",
                "message":  (
                    f"GRN {grn['grn_number']} shows only partial goods receipt. "
                    f"Invoice should only be for the quantity actually received."
                ),
            })
            result.recommendation = (
                "Review invoice — only partial goods received. "
                "Pay proportional amount for received quantity only."
            )

        elif grn_status == "fully_received":
            error_flags = [f for f in flags if f["severity"] == "error"]
            if not error_flags:
                result.match_status = "3way_matched"
                result.recommendation = (
                    "Auto-approve eligible — 3-way match successful. "
                    "Invoice, PO, and GRN are all aligned."
                )
            else:
                result.match_status  = "matched_with_errors"
                result.recommendation = (
                    "Needs review — 3-way match has errors. Review all flags before approving."
                )
        else:
            result.match_status  = "grn_status_unknown"
            result.recommendation = (
                "Manual review required — GRN status not recognised."
            )

    result.variance_details = variance_details
    result.flags            = flags
    return result
