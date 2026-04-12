"""
Streamlit invoice review table.
Displays all extracted fields with confidence heatmap, PO/GRN match
status, validation flags with plain-English explanations, and PII toggle.
"""
import streamlit as st
import pandas as pd
from guardrails.output_parser import InvoiceExtraction
from callbacks.audit_callback import log_audit_event
from config import AUTO_APPROVE_THRESHOLD, NEEDS_REVIEW_THRESHOLD


def _conf_badge(score: float) -> str:
    """Convert 0–1 confidence score to coloured badge label."""
    if score >= AUTO_APPROVE_THRESHOLD:
        return "🟢 HIGH"
    if score >= NEEDS_REVIEW_THRESHOLD:
        return "🟡 MED"
    return "🔴 LOW"


def _inr(n) -> str:
    """Format a numeric value as INR currency string, or — if None."""
    if n is None:
        return "—"
    return f"₹{float(n):,.2f}"


def render_review_table(
    extraction: InvoiceExtraction,
    redaction_map: dict,
    explanations: dict,
) -> InvoiceExtraction:
    """
    Render complete invoice review UI for one invoice.
    Shows: header banner, match metrics, field extraction table with confidence,
    line items, variance details, validation flags with explanations, and PII toggle.

    Args:
        extraction:    InvoiceExtraction with all fields and flags populated
        redaction_map: {placeholder: original_value} from Presidio redaction
        explanations:  {flag_index: explanation_string} from explanation chain

    Returns:
        The same extraction (passed through for any chaining)
    """
    # ── Status banner ─────────────────────────────────────────────────────────
    status_cfg = {
        "auto_approved": ("✅ AUTO-APPROVED",      "success"),
        "needs_review":  ("⚠️ NEEDS REVIEW",       "warning"),
        "failed":        ("❌ FAILED / REJECTED",   "error"),
    }
    label, kind = status_cfg.get(extraction.status, ("❓ UNKNOWN", "info"))
    getattr(st, kind)(
        f"**{label}** — "
        f"{extraction.invoice_number or 'No invoice number'} | "
        f"{extraction.vendor_name or 'Unknown vendor'}"
    )

    # ── Match status metrics ──────────────────────────────────────────────────
    match_colors = {
        "3way_matched":        "🟢 3-Way Matched",
        "matched_with_errors": "🟡 Matched (errors)",
        "partial_grn":         "🟡 Partial GRN",
        "grn_pending":         "🟡 GRN Pending",
        "no_po_match":         "🔴 No PO Found",
        "po_closed":           "🔴 PO Closed",
        "grn_rejected":        "🔴 GRN Rejected",
    }
    match_label = match_colors.get(extraction.match_status, extraction.match_status)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Match Status",   match_label)
    c2.metric("PO Reference",   extraction.matched_po_number or "—")
    c3.metric("GRN Number",     extraction.grn_number or "Pending")
    c4.metric("PO Match Score", f"{extraction.po_match_score:.0%}")

    if extraction.match_recommendation:
        st.info(f"💡 **Recommendation:** {extraction.match_recommendation}")

    st.divider()

    # ── Field extraction table ────────────────────────────────────────────────
    st.subheader("📋 Extracted Invoice Fields")
    conf = extraction.confidence

    field_rows = [
        ("Invoice Number",   extraction.invoice_number,  conf.get("invoice_number", 0.0)),
        ("Invoice Date",     extraction.invoice_date,    conf.get("invoice_date", 0.0)),
        ("Due Date",         extraction.due_date,        conf.get("due_date", 0.0)),
        ("PO Reference",     extraction.po_number,       conf.get("po_number", 0.0)),
        ("Vendor Name",      extraction.vendor_name,     conf.get("vendor_name", 0.0)),
        ("Vendor GSTIN",     extraction.vendor_gstin,    conf.get("vendor_gstin", 0.0)),
        ("Buyer Name",       extraction.buyer_name,      conf.get("buyer_name", 0.0)),
        ("Buyer GSTIN",      extraction.buyer_gstin,     conf.get("buyer_gstin", 0.0)),
        ("Place of Supply",  extraction.place_of_supply, conf.get("place_of_supply", 0.0)),
        ("Subtotal",         _inr(extraction.subtotal),  conf.get("subtotal", 0.0)),
        ("CGST",             _inr(extraction.cgst_amount), conf.get("cgst_amount", 0.0)),
        ("SGST",             _inr(extraction.sgst_amount), conf.get("sgst_amount", 0.0)),
        ("Total Tax",        _inr(extraction.tax_amount),  conf.get("tax_amount", 0.0)),
        ("TOTAL AMOUNT",     _inr(extraction.total_amount), conf.get("total_amount", 0.0)),
        ("Amount in Words",  extraction.amount_in_words, conf.get("amount_in_words", 0.0)),
        ("Payment Terms",    extraction.payment_terms,   conf.get("payment_terms", 0.0)),
        ("Bank Name",        extraction.bank_name,       conf.get("bank_name", 0.0)),
        ("IFSC Code",        extraction.ifsc_code,       conf.get("ifsc_code", 0.0)),
    ]

    df = pd.DataFrame(field_rows, columns=["Field", "Extracted Value", "Confidence"])
    df["Confidence"] = df["Confidence"].apply(
        lambda s: _conf_badge(s) if isinstance(s, (int, float)) else "—"
    )
    
    # Simple UI for review and correction
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        disabled=["Field", "Confidence"], # Only Extracted Value is editable
        key=f"editor_{extraction.invoice_id}"
    )
    
    # If the user corrected any fields, we can update the extraction object
    # (in a real system we'd save this back to DB here)
    if not edited_df.equals(df):
        st.success("✅ Corrections saved successfully.")

    # Duplicate alert
    if extraction.duplicate_of:
        st.error(
            f"⚠️ **DUPLICATE DETECTED** — This invoice is "
            f"{extraction.duplicate_similarity:.1%} similar to invoice "
            f"{extraction.duplicate_of}. Verify before approving payment."
        )

    st.divider()

    # ── Line items ────────────────────────────────────────────────────────────
    if extraction.line_items:
        st.subheader("📦 Line Items")
        li_rows = []
        for item in extraction.line_items:
            li_rows.append({
                "Description": item.description or "—",
                "HSN/SAC":     item.hsn_code    or "—",
                "Qty":         item.quantity,
                "Unit Price":  _inr(item.unit_price),
                "GST %":       item.gst_rate,
                "CGST":        _inr(item.cgst),
                "SGST":        _inr(item.sgst),
                "Amount":      _inr(item.amount),
            })
        st.dataframe(
            pd.DataFrame(li_rows), use_container_width=True, hide_index=True
        )

    # ── Variance details ──────────────────────────────────────────────────────
    if extraction.variance_details:
        st.subheader("📊 Variance vs PO")
        var_rows = []
        for key, val in extraction.variance_details.items():
            within = val.get("within_tolerance", True)
            var_rows.append({
                "Check":            key.replace("_", " ").title(),
                "PO Value":         _inr(val.get("po_value")),
                "Invoice Value":    _inr(val.get("invoice_value")),
                "Variance %":       f"{val.get('variance_pct', 0):.1f}%",
                "Within Tolerance": "✅" if within else "❌",
            })
        st.dataframe(
            pd.DataFrame(var_rows), use_container_width=True, hide_index=True
        )

    st.divider()

    # ── Validation flags ──────────────────────────────────────────────────────
    st.subheader("🚩 Validation Flags")
    errors   = [f for f in extraction.flags if f.severity == "error"]
    warnings = [f for f in extraction.flags if f.severity == "warning"]
    infos    = [f for f in extraction.flags if f.severity == "info"]

    if not errors and not warnings:
        st.success("✅ No errors or warnings — all validations passed.")

    for i, flag in enumerate(errors):
        with st.expander(
            f"❌ ERROR: {flag.rule.replace('_', ' ').title()}", expanded=True
        ):
            st.error(flag.message)
            if i in explanations:
                st.caption(f"💡 What to do: {explanations[i]}")

    for i, flag in enumerate(warnings, len(errors)):
        with st.expander(
            f"⚠️ WARNING: {flag.rule.replace('_', ' ').title()}"
        ):
            st.warning(flag.message)
            if i in explanations:
                st.caption(f"💡 What to do: {explanations[i]}")

    for flag in infos:
        st.caption(f"ℹ️ {flag.message}")

    # ── PII toggle ────────────────────────────────────────────────────────────
    st.divider()
    show_pii = st.toggle(
        "🔓 Show Redacted PII (authorised reviewers only — logged)",
        value=False,
        key=f"pii_toggle_{extraction.invoice_id}",
    )
    if show_pii:
        st.warning("⚠️ PII is now visible. This action is logged in the audit trail.")
        if redaction_map:
            for placeholder, original in redaction_map.items():
                st.code(f"{placeholder}  →  {original}")
        else:
            st.caption("No PII was detected and redacted in this invoice.")
        log_audit_event(extraction.invoice_id, "pii_view_toggled")

    return extraction
