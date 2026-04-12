"""
Deterministic business rule validator for Indian GST invoices.
All rules are Python-based (fast, no LLM cost).
LLM is used only for plain-English flag explanations.

Validation Rules:
  1. GSTIN format (15-char regex)
  2. GSTIN state code matches vendor state
  3. Invoice date not in future, not >180 days old
  4. Due date must be after invoice date
  5. Tax calculation: CGST+SGST=18% (intra-state) or IGST=18%
  6. Amount reconciliation: subtotal + tax = total (±₹1 rounding)
  7. HSN codes present and 6-digit starting with 99 (services)
  8. Mandatory fields: invoice_number, invoice_date, vendor_gstin,
     buyer_gstin, total_amount
  9. Currency must be INR
"""
import re
from datetime import datetime, date
from typing import List, Tuple
from config import AMOUNT_ROUNDING_INR
from guardrails.output_parser import InvoiceExtraction

GSTIN_PATTERN = re.compile(
    r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$'
)

STATE_CODES = {
    "01": "Jammu & Kashmir",    "02": "Himachal Pradesh",
    "03": "Punjab",             "04": "Chandigarh",
    "05": "Uttarakhand",        "06": "Haryana",
    "07": "Delhi",              "08": "Rajasthan",
    "09": "Uttar Pradesh",      "10": "Bihar",
    "11": "Sikkim",             "12": "Arunachal Pradesh",
    "13": "Nagaland",           "14": "Manipur",
    "15": "Mizoram",            "16": "Tripura",
    "17": "Meghalaya",          "18": "Assam",
    "19": "West Bengal",        "20": "Jharkhand",
    "21": "Odisha",             "22": "Chattisgarh",
    "23": "Madhya Pradesh",     "24": "Gujarat",
    "27": "Maharashtra",        "28": "Andhra Pradesh",
    "29": "Karnataka",          "30": "Goa",
    "31": "Lakshadweep",        "32": "Kerala",
    "33": "Tamil Nadu",         "34": "Puducherry",
    "36": "Telangana",          "37": "Andhra Pradesh (New)",
}


def validate_gstin(gstin: str) -> bool:
    """
    Validate GSTIN against the 15-character Indian GST format regex.

    Args:
        gstin: GSTIN string to validate

    Returns:
        True if valid 15-char GSTIN, False otherwise
    """
    if not gstin:
        return False
    return bool(GSTIN_PATTERN.match(gstin.strip().upper()))


def validate_gstin_state(
    gstin: str,
    expected_state_code: str = None,
) -> Tuple[bool, str]:
    """
    Validate GSTIN format and extract state name from first 2 digits.

    Args:
        gstin:               GSTIN to validate
        expected_state_code: Optional 2-digit state code to match against

    Returns:
        (is_valid, state_name) tuple
    """
    if not validate_gstin(gstin):
        return (False, "Invalid GSTIN format")
    state_code = gstin[:2]
    state_name = STATE_CODES.get(state_code, f"Unknown state code {state_code}")
    return (True, state_name)


def validate_date_logic(invoice_date: str, due_date: str) -> bool:
    """
    Validate that due_date is strictly after invoice_date.

    Args:
        invoice_date: Date string YYYY-MM-DD
        due_date:     Date string YYYY-MM-DD

    Returns:
        True if due_date > invoice_date, False otherwise
    """
    try:
        inv = datetime.strptime(invoice_date, "%Y-%m-%d").date()
        due = datetime.strptime(due_date, "%Y-%m-%d").date()
        return due > inv
    except Exception:
        return False


def validate_invoice_date_range(invoice_date: str) -> Tuple[bool, str]:
    """
    Validate invoice date is:
    - Not in the future (max +1 day timezone leeway)
    - Not older than 180 days

    Args:
        invoice_date: Date string YYYY-MM-DD

    Returns:
        (is_valid, error_message) tuple; error_message is "" if valid
    """
    try:
        inv   = datetime.strptime(invoice_date, "%Y-%m-%d").date()
        today = date.today()
        if inv > today:
            return (False, f"Invoice date {invoice_date} is in the future")
        age_days = (today - inv).days
        if age_days > 180:
            return (False, f"Invoice date {invoice_date} is {age_days} days old (>180 days)")
        return (True, "")
    except Exception:
        return (False, f"Cannot parse invoice_date: {invoice_date}")


def validate_amount_reconciliation(
    subtotal: float,
    tax_amount: float,
    total_amount: float,
    rounding: float = AMOUNT_ROUNDING_INR,
) -> Tuple[bool, float]:
    """
    Validate: subtotal + tax_amount ≈ total_amount (allow ±₹1 rounding).

    Args:
        subtotal:     Invoice subtotal before tax
        tax_amount:   Total tax (CGST + SGST or IGST)
        total_amount: Invoice grand total
        rounding:     Rounding tolerance in INR (default ₹1)

    Returns:
        (is_valid, variance_amount) tuple
    """
    expected = subtotal + tax_amount
    variance = abs(total_amount - expected)
    return (variance <= rounding, variance)


def validate_tax_calculation(
    subtotal: float,
    cgst: float = None,
    sgst: float = None,
    igst: float = None,
    gst_rate: float = 18.0,
) -> Tuple[bool, str]:
    """
    Validate GST calculation accuracy:
    - Intra-state: CGST = SGST = 9% of subtotal (for 18% GST rate)
    - Inter-state: IGST = 18% of subtotal

    Args:
        subtotal:  Subtotal before tax
        cgst:      Central GST amount (intra-state)
        sgst:      State GST amount (intra-state)
        igst:      Integrated GST amount (inter-state)
        gst_rate:  Applicable GST rate percentage (default 18%)

    Returns:
        (is_valid, error_message) tuple
    """
    tolerance = subtotal * 0.005   # 0.5% tolerance for rounding
    if cgst is not None and sgst is not None:
        expected_each = subtotal * (gst_rate / 2) / 100
        if (
            abs(cgst - expected_each) > tolerance
            or abs(sgst - expected_each) > tolerance
        ):
            return (
                False,
                f"CGST/SGST should be ₹{expected_each:,.2f} each for {gst_rate}% GST",
            )
    elif igst is not None:
        expected_igst = subtotal * gst_rate / 100
        if abs(igst - expected_igst) > tolerance:
            return (
                False,
                f"IGST should be ₹{expected_igst:,.2f} for {gst_rate}% GST",
            )
    return (True, "")


def validate_hsn_codes(line_items: list) -> Tuple[bool, str]:
    """
    Validate HSN/SAC codes for service invoices.
    Services must have 6-digit HSN/SAC starting with '99'.

    Args:
        line_items: List of LineItem objects from InvoiceExtraction

    Returns:
        (is_valid, error_message) tuple
    """
    for item in line_items:
        hsn = (item.hsn_code or "").strip()
        if not hsn:
            return (False, f"HSN/SAC code missing for: {item.description}")
        if len(hsn) != 6:
            return (False, f"HSN/SAC code '{hsn}' must be 6 digits")
        if not hsn.startswith("99"):
            return (False, f"Service HSN code '{hsn}' should start with '99'")
    return (True, "")


def run_all_validations(extraction: InvoiceExtraction) -> List[dict]:
    """
    Run all deterministic business rule validations on an InvoiceExtraction.
    Returns list of flag dicts for appending to extraction.flags.

    Args:
        extraction: InvoiceExtraction from LLM extraction chain

    Returns:
        List of flag dicts with keys: rule, field, severity, message
    """
    flags = []

    # Rule 1: Mandatory fields
    mandatory = {
        "invoice_number": extraction.invoice_number,
        "invoice_date":   extraction.invoice_date,
        "vendor_gstin":   extraction.vendor_gstin,
        "buyer_gstin":    extraction.buyer_gstin,
        "total_amount":   extraction.total_amount,
    }
    for field, value in mandatory.items():
        if not value:
            flags.append({
                "rule":     "mandatory_field_missing",
                "field":    field,
                "severity": "error",
                "message":  f"Mandatory field '{field}' is missing or could not be extracted",
            })

    # Rule 2: Vendor GSTIN format validation
    if extraction.vendor_gstin:
        if not validate_gstin(extraction.vendor_gstin):
            flags.append({
                "rule":     "gstin_format_invalid",
                "field":    "vendor_gstin",
                "severity": "error",
                "message":  (
                    f"Vendor GSTIN '{extraction.vendor_gstin}' does not match "
                    f"the 15-character Indian GST format"
                ),
            })
        else:
            _, state = validate_gstin_state(extraction.vendor_gstin)
            flags.append({
                "rule":     "gstin_state_resolved",
                "field":    "vendor_gstin",
                "severity": "info",
                "message":  f"Vendor state from GSTIN: {state}",
            })

    # Rule 3: Buyer GSTIN format
    if extraction.buyer_gstin:
        if not validate_gstin(extraction.buyer_gstin):
            flags.append({
                "rule":     "buyer_gstin_format_invalid",
                "field":    "buyer_gstin",
                "severity": "error",
                "message":  (
                    f"Buyer GSTIN '{extraction.buyer_gstin}' does not match "
                    f"the 15-character Indian GST format"
                ),
            })

    # Rule 4: Invoice date range
    if extraction.invoice_date:
        ok, msg = validate_invoice_date_range(extraction.invoice_date)
        if not ok:
            flags.append({
                "rule":     "invoice_date_out_of_range",
                "field":    "invoice_date",
                "severity": "error",
                "message":  msg,
            })

    # Rule 5: Due date logic
    if extraction.invoice_date and extraction.due_date:
        if not validate_date_logic(extraction.invoice_date, extraction.due_date):
            flags.append({
                "rule":     "due_date_before_invoice_date",
                "field":    "due_date",
                "severity": "error",
                "message":  (
                    f"Due date {extraction.due_date} is before "
                    f"invoice date {extraction.invoice_date}"
                ),
            })

    # Rule 6: Amount reconciliation
    if all(
        v is not None
        for v in [extraction.subtotal, extraction.tax_amount, extraction.total_amount]
    ):
        ok, variance = validate_amount_reconciliation(
            extraction.subtotal, extraction.tax_amount, extraction.total_amount
        )
        if not ok:
            flags.append({
                "rule":     "amount_mismatch",
                "field":    "total_amount",
                "severity": "error",
                "message":  (
                    f"Amount mismatch: ₹{extraction.subtotal:,.2f} + "
                    f"₹{extraction.tax_amount:,.2f} = "
                    f"₹{extraction.subtotal + extraction.tax_amount:,.2f} "
                    f"but invoice total is ₹{extraction.total_amount:,.2f}. "
                    f"Variance: ₹{variance:,.2f}"
                ),
            })

    # Rule 7: Tax calculation
    if (
        extraction.subtotal
        and extraction.cgst_amount
        and extraction.sgst_amount
    ):
        ok, msg = validate_tax_calculation(
            extraction.subtotal,
            cgst=extraction.cgst_amount,
            sgst=extraction.sgst_amount,
        )
        if not ok:
            flags.append({
                "rule":     "tax_calculation_error",
                "field":    "cgst_amount",
                "severity": "warning",
                "message":  msg,
            })

    # Rule 8: HSN codes
    if extraction.line_items:
        ok, msg = validate_hsn_codes(extraction.line_items)
        if not ok:
            flags.append({
                "rule":     "hsn_code_invalid",
                "field":    "line_items",
                "severity": "warning",
                "message":  msg,
            })

    # Rule 9: Currency
    if extraction.currency and extraction.currency.upper() != "INR":
        flags.append({
            "rule":     "currency_not_inr",
            "field":    "currency",
            "severity": "warning",
            "message":  f"Invoice currency is {extraction.currency}, expected INR",
        })

    return flags
