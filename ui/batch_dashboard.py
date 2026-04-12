"""
Streamlit batch dashboard.
Shows metrics, confidence chart, executive summary, duplicate alerts,
match status distribution, and CSV/JSON export buttons.
"""
import streamlit as st
import pandas as pd
from collections import Counter
from guardrails.output_parser import InvoiceExtraction
from exports.exporter import export_csv, export_json


def render_batch_dashboard(
    results: list,
    filenames: list,
    batch_summary: str,
) -> None:
    """
    Render the complete batch processing dashboard.

    Args:
        results:       List of InvoiceExtraction objects
        filenames:     Corresponding list of original filenames
        batch_summary: Executive summary string from summary_chain
    """
    total       = len(results)
    auto_app    = sum(1 for r in results if r.status == "auto_approved")
    review      = sum(1 for r in results if r.status == "needs_review")
    failed      = sum(1 for r in results if r.status == "failed")
    three_way   = sum(1 for r in results if r.match_status == "3way_matched")
    total_value = sum(r.total_amount for r in results if r.total_amount)
    all_conf    = [v for r in results for v in r.confidence.values()]
    avg_conf    = sum(all_conf) / len(all_conf) if all_conf else 0.0
    total_flags = sum(
        len([f for f in r.flags if f.severity == "error"]) for r in results
    )

    # ── Top-level metrics ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Invoices",       total)
    c2.metric("Auto-Approved ✅",     auto_app)
    c3.metric("Needs Review ⚠️",      review)
    c4.metric("Failed / Rejected ❌", failed)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Value (₹)",  f"{total_value:,.0f}")
    c6.metric("3-Way Matched",    three_way)
    c7.metric("Avg Confidence",   f"{avg_conf:.0%}")
    c8.metric("Total Error Flags", total_flags)

    # ── Executive summary ─────────────────────────────────────────────────────
    if batch_summary:
        st.markdown("### 📝 Executive Summary")
        st.info(batch_summary)

    # ── Duplicate alerts ──────────────────────────────────────────────────────
    duplicates = [r for r in results if r.duplicate_of]
    if duplicates:
        st.markdown("### ⚠️ Duplicate Invoice Alerts")
        for r in duplicates:
            st.warning(
                f"**{r.filename}** is {(r.duplicate_similarity or 0):.1%} similar to "
                f"invoice {r.duplicate_of}. Hold until verified."
            )

    # ── Invoice summary table ─────────────────────────────────────────────────
    st.markdown("### 📋 All Invoices")
    rows = []
    for r, fn in zip(results, filenames):
        avg_c = (
            sum(r.confidence.values()) / len(r.confidence)
            if r.confidence else 0.0
        )
        rows.append({
            "File":       fn[:35],
            "Vendor":     r.vendor_name      or "—",
            "Invoice No": r.invoice_number   or "—",
            "Date":       r.invoice_date      or "—",
            "Total (₹)":  f"{r.total_amount:,.2f}" if r.total_amount else "—",
            "PO Match":   r.match_status.replace("_", " ").title(),
            "GRN":        r.grn_number        or "Pending",
            "Status":     r.status.replace("_", " ").title(),
            "Errors":     sum(1 for f in r.flags if f.severity == "error"),
            "Confidence": f"{avg_c:.0%}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Confidence bar chart ──────────────────────────────────────────────────
    st.markdown("### 📊 Confidence by Invoice")
    conf_data = {
        fn[:20]: (sum(r.confidence.values()) / len(r.confidence) if r.confidence else 0.0)
        for r, fn in zip(results, filenames)
    }
    if conf_data:
        st.bar_chart(conf_data)

    # ── Match status distribution ─────────────────────────────────────────────
    st.markdown("### 🔄 Match Status Distribution")
    ms_counts = dict(Counter(r.match_status for r in results))
    if ms_counts:
        st.bar_chart(ms_counts)

    # ── Approval status breakdown ─────────────────────────────────────────────
    st.markdown("### ✅ Approval Status Breakdown")
    status_counts = dict(Counter(r.status for r in results))
    if status_counts:
        st.bar_chart(status_counts)

    # ── Export section ────────────────────────────────────────────────────────
    st.markdown("### 📥 Export Results")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export CSV", use_container_width=True, key="export_csv_btn"):
            path = export_csv(results, filenames)
            st.success(f"✅ Saved: {path}")
    with col2:
        if st.button("📥 Export JSON", use_container_width=True, key="export_json_btn"):
            path = export_json(results)
            st.success(f"✅ Saved: {path}")
