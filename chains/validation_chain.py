"""
LangChain LCEL validation chain.
Combines deterministic Python rule checks + 3-way PO/GRN matching.
Sets final approval status on the InvoiceExtraction object.
"""
import logging
from config import AUTO_APPROVE_THRESHOLD, NEEDS_REVIEW_THRESHOLD
from guardrails.output_parser import InvoiceExtraction, Flag
from guardrails.field_validator import run_all_validations
from matching.po_matcher import extract_po_number_from_text, load_po_database
from matching.three_way_matcher import run_three_way_match, load_grn_database

logger = logging.getLogger(__name__)


def run_validation(
    extraction: InvoiceExtraction,
    raw_text: str = "",
) -> InvoiceExtraction:
    """
    Run complete validation pipeline on an extracted invoice.

    Steps:
    1. Regex fallback — extract PO number from text if LLM missed it
    2. Business rule validation (GSTIN, dates, amounts, HSN)
    3. 3-way matching (Invoice ↔ PO ↔ GRN)
    4. Determine final approval status

    Args:
        extraction: InvoiceExtraction from run_extraction()
        raw_text:   Redacted invoice text (for regex PO fallback)

    Returns:
        Updated InvoiceExtraction with flags, match_status, and final status
    """
    # Step 1: Regex PO number fallback
    if not extraction.po_number and raw_text:
        po_from_text = extract_po_number_from_text(raw_text)
        if po_from_text:
            extraction.po_number = po_from_text
            logger.info(f"PO number extracted via regex: {po_from_text}")

    # Step 2: Business rule validation
    field_flags = run_all_validations(extraction)
    for f in field_flags:
        extraction.flags.append(Flag(**f))

    # Step 3: 3-way matching
    po_database  = load_po_database()
    grn_database = load_grn_database()
    match_result = run_three_way_match(extraction, po_database, grn_database)

    extraction.matched_po_number    = match_result.po_number
    extraction.po_match_score       = match_result.po_match_score
    extraction.grn_number           = match_result.grn_number
    extraction.match_status         = match_result.match_status
    extraction.variance_details     = match_result.variance_details
    extraction.match_recommendation = match_result.recommendation

    for f in match_result.flags:
        extraction.flags.append(Flag(**f))

    # Step 4: Final status determination
    if extraction.status == "failed":
        return extraction

    error_flags = [f for f in extraction.flags if f.severity == "error"]
    all_conf    = list(extraction.confidence.values())
    avg_conf    = sum(all_conf) / len(all_conf) if all_conf else 0.5
    low_conf    = [v for v in all_conf if v < NEEDS_REVIEW_THRESHOLD]

    if match_result.match_status in ("grn_rejected", "po_closed"):
        extraction.status = "failed"
    elif not error_flags and match_result.match_status == "3way_matched":
        # Auto-approve if no errors and PO/GRN matched.
        # Confidence scores are optional — only penalize when actually returned by LLM.
        if all_conf:
            if avg_conf >= AUTO_APPROVE_THRESHOLD and not low_conf:
                extraction.status = "auto_approved"
            else:
                extraction.status = "needs_review"
        else:
            # No confidence scores (Groq/small model) — approve on rule checks alone
            extraction.status = "auto_approved"
    else:
        extraction.status = "needs_review"

    return extraction
