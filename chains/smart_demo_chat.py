"""
Smart local chat engine for Demo Mode.
Reads the real extracted InvoiceExtraction data from session state
and answers any user question intelligently — zero latency, no LLM needed.
"""
from typing import List
import re


PII_FIELDS = ["pan", "aadhaar", "phone", "email", "address", "account", "ifsc", "pii", "sensitive", "personal", "contact"]


def _fmt_amt(val):
    if val is None:
        return "Not found"
    return f"₹{val:,.2f}"


def smart_demo_answer(question: str, results: list) -> str:
    """
    Intelligently answer any question about the processed invoices
    using only the extracted InvoiceExtraction data. No LLM needed.
    """
    if not results:
        return "No invoices have been processed yet. Please upload and process invoices first."

    q = question.lower().strip()

    # ── PII questions ─────────────────────────────────────────────────────────
    if any(k in q for k in PII_FIELDS):
        return "🔒 Redacted — sensitive PII data is not accessible for privacy and security."

    # ── "which / list / show" style questions ─────────────────────────────────

    # Which invoices don't have a PO number?
    if ("po" in q or "purchase order" in q) and any(k in q for k in ["no", "not", "without", "missing", "don't", "dont", "lack"]):
        missing = [r for r in results if not r.po_number]
        if not missing:
            return "All processed invoices have a PO number."
        names = "\n".join(f"• {r.filename} ({r.vendor_name or 'Unknown'})" for r in missing)
        return f"Invoices without a PO number:\n{names}"

    # Which invoices failed / have errors?
    if any(k in q for k in ["fail", "error", "reject"]):
        failed = [r for r in results if r.status == "failed"]
        if not failed:
            return "No invoices have failed."
        return "\n".join(f"• {r.filename} — {r.vendor_name or 'Unknown'}" for r in failed)

    # Which invoices need review?
    if any(k in q for k in ["review", "pending", "warning", "flag"]):
        review = [r for r in results if r.status == "needs_review"]
        if not review:
            return "No invoices need review — all are auto-approved."
        return "\n".join(f"• {r.filename} ({r.vendor_name or 'Unknown'})" for r in review)

    # Which invoices are approved?
    if any(k in q for k in ["approv", "pass", "ok", "clear"]):
        approved = [r for r in results if r.status == "auto_approved"]
        if not approved:
            return "No invoices have been auto-approved yet."
        return "\n".join(f"• {r.filename} ({r.vendor_name or 'Unknown'})" for r in approved)

    # Which / all vendors?
    if "vendor" in q and any(k in q for k in ["list", "all", "which", "what"]):
        vendors = [f"• {r.vendor_name or 'Unknown'} ({r.filename})" for r in results]
        return "Vendors processed:\n" + "\n".join(vendors)

    # ── Single field lookups (for single or multiple invoices) ────────────────
    r = results[0]  # Default to first invoice for single-invoice answers
    multi = len(results) > 1

    # Vendor name
    if "vendor" in q and "name" in q:
        if multi:
            return "\n".join(f"• {x.filename}: {x.vendor_name or 'Not found'}" for x in results)
        return r.vendor_name or "Not found"

    # Invoice number
    if "invoice" in q and any(k in q for k in ["no", "number", "id"]):
        if multi:
            return "\n".join(f"• {x.filename}: {x.invoice_number or 'Not found'}" for x in results)
        return r.invoice_number or "Not found"

    # Invoice date / order date
    if any(k in q for k in ["invoice date", "order date", "date of invoice", "invoice day"]):
        if multi:
            return "\n".join(f"• {x.filename}: {x.invoice_date or 'Not found'}" for x in results)
        return r.invoice_date or "Not found"

    if "due" in q and "date" in q:
        if multi:
            return "\n".join(f"• {x.filename}: {x.due_date or 'Not found'}" for x in results)
        return r.due_date or "Not found"

    if "date" in q:
        if multi:
            return "\n".join(f"• {x.filename}: Invoice Date: {x.invoice_date or 'N/A'}, Due Date: {x.due_date or 'N/A'}" for x in results)
        return f"Invoice Date: {r.invoice_date or 'N/A'} | Due Date: {r.due_date or 'N/A'}"

    # PO Number
    if "po" in q or "purchase order" in q:
        if multi:
            return "\n".join(f"• {x.filename}: {x.po_number or 'None'}" for x in results)
        return r.po_number or "No PO number found"

    # Total amount
    if any(k in q for k in ["total", "grand total", "invoice amount"]):
        if multi:
            total_sum = sum(x.total_amount or 0 for x in results)
            lines = "\n".join(f"• {x.filename}: {_fmt_amt(x.total_amount)}" for x in results)
            return f"{lines}\n\n**Combined Total: {_fmt_amt(total_sum)}**"
        return _fmt_amt(r.total_amount)

    # Tax / GST
    if any(k in q for k in ["tax", "gst", "cgst", "sgst", "igst"]):
        if multi:
            return "\n".join(f"• {x.filename}: Tax: {_fmt_amt(x.tax_amount)}, CGST: {_fmt_amt(x.cgst_amount)}, SGST: {_fmt_amt(x.sgst_amount)}" for x in results)
        return f"Tax: {_fmt_amt(r.tax_amount)} | CGST: {_fmt_amt(r.cgst_amount)} | SGST: {_fmt_amt(r.sgst_amount)}"

    # Subtotal
    if "subtotal" in q or "sub total" in q:
        if multi:
            return "\n".join(f"• {x.filename}: {_fmt_amt(x.subtotal)}" for x in results)
        return _fmt_amt(r.subtotal)

    # Status
    if "status" in q:
        if multi:
            return "\n".join(f"• {x.filename}: {x.status.replace('_', ' ').title()}" for x in results)
        return r.status.replace("_", " ").title()

    # Flags / issues / problems
    if any(k in q for k in ["flag", "issue", "problem", "mismatch", "discrepan"]):
        lines = []
        for x in results:
            if x.flags:
                flag_msgs = "; ".join(f.message for f in x.flags[:3])
                lines.append(f"• {x.filename}: {flag_msgs}")
            else:
                lines.append(f"• {x.filename}: No issues found ✅")
        return "\n".join(lines)

    # GSTIN
    if "gstin" in q:
        if multi:
            return "\n".join(f"• {x.filename}: {x.vendor_gstin or 'Not found'}" for x in results)
        return r.vendor_gstin or "Not found"

    # Summary / all fields / display all
    if any(k in q for k in ["summarize", "summary", "all detail", "all field", "display", "show all", "extract", "full detail", "overview"]):
        summaries = []
        for x in results:
            summaries.append(
                f"**{x.filename}**\n"
                f"• Vendor: {x.vendor_name or 'N/A'}\n"
                f"• Invoice No: {x.invoice_number or 'N/A'}\n"
                f"• Invoice Date: {x.invoice_date or 'N/A'}\n"
                f"• PO Number: {x.po_number or 'N/A'}\n"
                f"• Total Amount: {_fmt_amt(x.total_amount)}\n"
                f"• Tax: {_fmt_amt(x.tax_amount)}\n"
                f"• Status: {x.status.replace('_', ' ').title()}\n"
                f"• GSTIN: {x.vendor_gstin or 'Redacted/N/A'}"
            )
        return "\n\n".join(summaries)

    # How many invoices
    if any(k in q for k in ["how many", "count", "number of invoice"]):
        return f"{len(results)} invoice(s) processed."

    # Highest amount
    if any(k in q for k in ["highest", "largest", "maximum", "most expensive"]):
        top = max(results, key=lambda x: x.total_amount or 0)
        return f"{top.filename} — {top.vendor_name or 'Unknown'} ({_fmt_amt(top.total_amount)})"

    # Lowest amount
    if any(k in q for k in ["lowest", "smallest", "minimum", "least"]):
        bot = min(results, key=lambda x: x.total_amount or 0)
        return f"{bot.filename} — {bot.vendor_name or 'Unknown'} ({_fmt_amt(bot.total_amount)})"

    # Duplicate
    if "duplicate" in q:
        dups = [r for r in results if r.duplicate_of]
        if not dups:
            return "No duplicate invoices detected."
        return "\n".join(f"• {x.filename} is a duplicate of invoice ID: {x.duplicate_of}" for x in dups)

    # Fallback — show summary of all
    return (
        f"{len(results)} invoice(s) processed. You can ask me about:\n"
        "• Vendor name, Invoice number, Date, PO number\n"
        "• Total amount, Tax, Subtotal, GSTIN\n"
        "• Status, Flags/issues, Duplicates\n"
        "• Which invoices are missing PO / failed / need review\n"
        "• Summary / display all fields"
    )
