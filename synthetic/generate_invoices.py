"""
Generates 20 realistic Indian B2B GST invoice PDFs and PNG images.
Uses reportlab for PDF generation. All invoices reference real POs.
Tesseract OCR is preinstalled on TCS lab machines — no install needed.

Run this ONCE after setup:
    python synthetic/generate_invoices.py
"""
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle,
    Paragraph, Spacer, HRFlowable,
)
from reportlab.lib.enums import TA_RIGHT, TA_CENTER, TA_LEFT
from PIL import Image

PO_MASTER_PATH = Path(__file__).parent / "po_master.json"
OUTPUT_DIR     = Path(__file__).parent / "invoices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

po_database = json.loads(PO_MASTER_PATH.read_text(encoding="utf-8"))

INVOICE_SCENARIOS = [
    ("INV-2026-0001", "PO-2026-0001", "normal"),
    ("INV-2026-0002", "PO-2026-0002", "normal"),
    ("INV-2026-0003", "PO-2026-0003", "normal"),       # GRN rejected
    ("INV-2026-0004", "PO-2026-0004", "normal"),
    ("INV-2026-0005", "PO-2026-0005", "normal"),
    ("INV-2026-0006", "PO-2026-0006", "mixed_content"),
    ("INV-2026-0007", "PO-2026-0007", "mixed_content"),
    ("INV-2026-0008", "PO-2026-0008", "mixed_content"),
    ("INV-2026-0009", "PO-2026-0009", "mixed_content"),
    ("INV-2026-0010", "PO-2026-0010", "mixed_content"),
    ("INV-2026-0011", "PO-2026-0011", "amount_mismatch"),  # subtotal+tax != total
    ("INV-2026-0012", "PO-2026-0012", "date_error"),        # due_date < invoice_date
    ("INV-2026-0013", "PO-2026-0013", "missing_gstin"),     # GSTIN blank
    ("INV-2026-0014", "PO-2026-0014", "duplicate_a"),       # duplicate pair A
    ("INV-2026-0015", "PO-2026-0014", "duplicate_b"),       # duplicate pair B (same PO!)
    ("INV-2026-0016", "PO-2026-0018", "closed_po"),         # PO already fully_invoiced
    ("INV-2026-0017", "PO-2026-0019", "closed_po"),         # PO already fully_invoiced
    ("INV-2026-0018", "PO-2026-0007", "qty_overbill"),      # qty 1.5x PO qty
    ("INV-2026-0019", "PO-2026-0008", "qty_overbill"),      # qty 1.5x PO qty
    ("INV-2026-0020", "PO-2026-0004", "price_variance"),    # unit_price +6% above PO
]

# These 3 invoices are saved as PNG instead of PDF
PNG_INVOICES = {"INV-2026-0001", "INV-2026-0005", "INV-2026-0010"}

BANK_DETAILS = [
    {
        "bank":    "HDFC Bank",
        "account": "50200012345678",
        "ifsc":    "HDFC0001234",
        "branch":  "Whitefield, Bengaluru",
    },
    {
        "bank":    "ICICI Bank",
        "account": "001005009876",
        "ifsc":    "ICIC0000010",
        "branch":  "Bandra Kurla Complex, Mumbai",
    },
    {
        "bank":    "State Bank of India",
        "account": "32109876543210",
        "ifsc":    "SBIN0001234",
        "branch":  "Sector 62, Noida",
    },
]

NAVY  = colors.HexColor("#1a237e")
LNAVY = colors.HexColor("#e8eaf6")
WHITE = colors.white
GREY  = colors.HexColor("#f5f5f5")


def get_po(po_number: str) -> dict:
    """Find a PO by number in the loaded po_database."""
    for po in po_database:
        if po["po_number"] == po_number:
            return po
    return {}


def inr(amount: float) -> str:
    """Format amount in Indian number format with ₹ symbol."""
    return f"₹{amount:,.2f}"


def num_to_words(n: float) -> str:
    """
    Convert a float amount to Indian number words.

    Args:
        n: Amount to convert (e.g. 708000.00)

    Returns:
        e.g. "Seven Lakh Eight Thousand Rupees Only"
    """
    ones = [
        "", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
        "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
        "Seventeen", "Eighteen", "Nineteen",
    ]
    tens = [
        "", "", "Twenty", "Thirty", "Forty", "Fifty",
        "Sixty", "Seventy", "Eighty", "Ninety",
    ]

    def _under_1000(n: int) -> str:
        if n < 20:
            return ones[n]
        elif n < 100:
            return tens[n // 10] + (" " + ones[n % 10] if n % 10 else "")
        else:
            return (
                ones[n // 100] + " Hundred"
                + (" " + _under_1000(n % 100) if n % 100 else "")
            )

    n = int(n)
    if n == 0:
        return "Zero Rupees Only"
    parts = []
    if n >= 10000000:
        parts.append(_under_1000(n // 10000000) + " Crore")
        n %= 10000000
    if n >= 100000:
        parts.append(_under_1000(n // 100000) + " Lakh")
        n %= 100000
    if n >= 1000:
        parts.append(_under_1000(n // 1000) + " Thousand")
        n %= 1000
    if n > 0:
        parts.append(_under_1000(n))
    return " ".join(parts) + " Rupees Only"


def generate_invoice_pdf(inv_num: str, po_num: str, scenario: str) -> Path:
    """
    Generate a single realistic Indian GST invoice PDF using reportlab.

    Args:
        inv_num:  Invoice number e.g. "INV-2026-0001"
        po_num:   PO reference e.g. "PO-2026-0001"
        scenario: Scenario type affecting data (normal, date_error, etc.)

    Returns:
        Path to the generated PDF file
    """
    po = get_po(po_num)
    if not po:
        raise ValueError(f"PO {po_num} not found in po_master.json")

    vendor       = po["vendor_name"]
    vendor_gstin = po.get("vendor_gstin", "")
    bank         = random.choice(BANK_DETAILS)

    inv_date = datetime.today() - timedelta(days=random.randint(5, 30))
    due_date = inv_date + timedelta(days=30)

    # Scenario-specific modifications
    if scenario == "date_error":
        due_date = inv_date - timedelta(days=5)
    if scenario == "missing_gstin":
        vendor_gstin = ""

    subtotal   = po["subtotal"]
    tax_amount = po["tax_amount"]
    total      = po["total_amount"]

    if scenario == "amount_mismatch":
        total = subtotal + tax_amount + 500.00   # deliberate error

    line_items = []
    for item in po.get("line_items", []):
        qty   = item["quantity"]
        price = item["unit_price"]

        if scenario == "qty_overbill":
            qty = int(qty * 1.5)
        if scenario == "price_variance":
            price = round(price * 1.06, 2)

        base_amt = qty * price
        cgst     = round(base_amt * 0.09, 2)
        sgst     = round(base_amt * 0.09, 2)
        line_items.append({
            "description": item["description"],
            "hsn_code":    item["hsn_code"],
            "quantity":    qty,
            "unit_price":  price,
            "gst_rate":    item["gst_rate"],
            "cgst":        cgst,
            "sgst":        sgst,
            "amount":      base_amt + cgst + sgst,
        })

    # Safe filename
    safe_vendor = vendor.replace(" ", "_").replace("&", "and")[:20]
    out_file    = OUTPUT_DIR / f"{inv_num.lower().replace('-', '_')}_{safe_vendor}.pdf"

    doc = SimpleDocTemplate(
        str(out_file),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles    = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title", fontSize=16, textColor=NAVY,
        fontName="Helvetica-Bold", spaceAfter=2,
    )
    sub_style = ParagraphStyle(
        "sub", fontSize=9, textColor=colors.grey, fontName="Helvetica",
    )
    inv_title = ParagraphStyle(
        "invt", fontSize=18, textColor=NAVY,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )
    right_style = ParagraphStyle("right", fontSize=9, alignment=TA_RIGHT)
    val_style   = ParagraphStyle(
        "val", fontSize=9, fontName="Helvetica-Bold",
    )

    story = []

    # ── Address / header block ────────────────────────────────────────────────
    gstin_display = (
        vendor_gstin
        if vendor_gstin
        else "[GSTIN MISSING]"
    )
    vendor_addr = (
        f"<b>{vendor}</b><br/>"
        f"{'404, Tower A, Prestige Tech Park' if 'Infosys' in vendor else '3rd Floor, DLF Cyber City'}<br/>"
        f"{'Whitefield, Bengaluru - 560066' if 'Infosys' in vendor else 'Gurugram, Haryana - 122002'}<br/>"
        f"GSTIN: {gstin_display}<br/>"
        f"Phone: +91-80-{random.randint(10000000, 99999999)}<br/>"
        f"Email: accounts@{vendor.split()[0].lower()}.com"
    )
    bill_to = (
        f"<b>Bill To:</b><br/>"
        f"Tata Consultancy Services Ltd<br/>"
        f"TCS House, Raveline Street, Fort<br/>"
        f"Mumbai - 400001, Maharashtra<br/>"
        f"GSTIN: {po.get('buyer_gstin', '27AAACT2727Q1ZW')}<br/>"
        f"State: Maharashtra (27)"
    )
    inv_details = (
        f"<b>Invoice No:</b> {inv_num}<br/>"
        f"<b>Invoice Date:</b> {inv_date.strftime('%d/%m/%Y')}<br/>"
        f"<b>Due Date:</b> {due_date.strftime('%d/%m/%Y')}<br/>"
        f"<b>PO Reference:</b> {po_num}<br/>"
        f"<b>Place of Supply:</b> {'Karnataka (29)' if 'Infosys' in vendor else 'Maharashtra (27)'}"
    )

    addr_table = Table(
        [[
            Paragraph(vendor_addr, sub_style),
            Paragraph(inv_details, sub_style),
            Paragraph(bill_to, sub_style),
        ]],
        colWidths=[62 * mm, 65 * mm, 58 * mm],
    )
    addr_table.setStyle(TableStyle([
        ("VALIGN",    (0, 0), (-1, -1), "TOP"),
        ("LINEAFTER", (0, 0), (1, 0),   0.5, colors.lightgrey),
    ]))

    story.append(Paragraph("TAX INVOICE", inv_title))
    story.append(Spacer(1, 4 * mm))
    story.append(addr_table)
    story.append(Spacer(1, 6 * mm))

    # ── Line items table ──────────────────────────────────────────────────────
    col_headers = [
        "Sr", "Description", "HSN/SAC", "Qty",
        "Unit Price (₹)", "GST %", "CGST (₹)", "SGST (₹)", "Total (₹)",
    ]
    table_data = [col_headers]
    for i, item in enumerate(line_items, 1):
        table_data.append([
            str(i),
            item["description"],
            item["hsn_code"],
            str(item["quantity"]),
            f"{item['unit_price']:,.2f}",
            f"{item['gst_rate']}%",
            f"{item['cgst']:,.2f}",
            f"{item['sgst']:,.2f}",
            f"{item['amount']:,.2f}",
        ])

    col_widths = [
        10 * mm, 55 * mm, 20 * mm, 12 * mm,
        22 * mm, 12 * mm, 20 * mm, 20 * mm, 22 * mm,
    ]
    items_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    items_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0),  (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0),  (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0),  (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0),  (-1, -1), 8),
        ("ALIGN",         (3, 0),  (-1, -1), "RIGHT"),
        ("ALIGN",         (0, 0),  (2, -1),  "LEFT"),
        ("ROWBACKGROUNDS",(0, 1),  (-1, -1), [WHITE, LNAVY]),
        ("GRID",          (0, 0),  (-1, -1), 0.3, colors.lightgrey),
        ("VALIGN",        (0, 0),  (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0),  (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0),  (-1, -1), 4),
    ]))
    story.append(items_table)
    story.append(Spacer(1, 6 * mm))

    # ── Summary box ───────────────────────────────────────────────────────────
    summary_data = [
        ["Subtotal:",         f"₹{subtotal:,.2f}"],
        ["CGST (9%):",        f"₹{tax_amount / 2:,.2f}"],
        ["SGST (9%):",        f"₹{tax_amount / 2:,.2f}"],
        ["Total Tax Amount:", f"₹{tax_amount:,.2f}"],
        ["", ""],
        ["TOTAL AMOUNT:",     f"₹{total:,.2f}"],
        ["Amount in Words:",  num_to_words(total)],
    ]
    summary_table = Table(summary_data, colWidths=[55 * mm, 55 * mm], hAlign="RIGHT")
    summary_table.setStyle(TableStyle([
        ("FONTNAME",        (0, 5),  (-1, 5),  "Helvetica-Bold"),
        ("FONTSIZE",        (0, 5),  (-1, 5),  11),
        ("TEXTCOLOR",       (0, 5),  (-1, 5),  NAVY),
        ("FONTSIZE",        (0, 0),  (-1, -1), 9),
        ("ALIGN",           (1, 0),  (1, -1),  "RIGHT"),
        ("LINEABOVE",       (0, 5),  (-1, 5),  1, NAVY),
        ("BOX",             (0, 0),  (-1, -1), 0.5, colors.grey),
        ("BACKGROUND",      (0, 5),  (-1, 5),  LNAVY),
        ("TOPPADDING",      (0, 0),  (-1, -1), 4),
        ("BOTTOMPADDING",   (0, 0),  (-1, -1), 4),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 6 * mm))

    # ── Payment terms (mixed_content scenario adds extra paragraphs) ──────────
    if scenario == "mixed_content":
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph("<b>Payment Terms &amp; Conditions:</b>", val_style))
        payment_text = (
            "Payment is due within 30 days of invoice date. Please transfer the amount "
            "to the bank account mentioned below. In case of any discrepancy in the "
            "invoice amount or description, please contact our accounts team within 7 "
            "days of receipt. Late payment will attract interest at 18% per annum as "
            "per our standard terms and conditions. This invoice is subject to the "
            "jurisdiction of courts in Bengaluru."
        )
        story.append(Paragraph(payment_text, sub_style))
        story.append(Spacer(1, 3 * mm))

    # ── Bank details ──────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 3 * mm))
    bank_text = (
        f"<b>Bank Details:</b> {bank['bank']} | "
        f"Account: {bank['account']} | "
        f"IFSC: {bank['ifsc']} | "
        f"Branch: {bank['branch']}"
    )
    story.append(Paragraph(bank_text, sub_style))
    story.append(Spacer(1, 5 * mm))

    # ── Footer ────────────────────────────────────────────────────────────────
    footer_data = [[
        Paragraph("This is a computer generated invoice.", sub_style),
        Paragraph(
            f"<b>Authorised Signatory:</b> ________________________<br/>"
            f"{vendor}<br/>Chief Financial Officer",
            right_style,
        ),
    ]]
    footer_table = Table(footer_data, colWidths=[95 * mm, 80 * mm])
    story.append(footer_table)

    doc.build(story)
    return out_file


def main() -> None:
    """Generate all 20 invoice PDFs/PNGs and print progress."""
    print("=" * 60)
    print("Generating 20 Indian GST Invoice PDFs/PNGs...")
    print("=" * 60)

    for inv_num, po_num, scenario in INVOICE_SCENARIOS:
        try:
            pdf_path = generate_invoice_pdf(inv_num, po_num, scenario)

            if inv_num in PNG_INVOICES:
                # Convert first PDF page to PNG and remove PDF
                try:
                    import fitz
                    doc = fitz.open(pdf_path)
                    page = doc[0]
                    pix = page.get_pixmap(dpi=200)
                    png_path = pdf_path.with_suffix(".png")
                    pix.save(str(png_path))
                    doc.close()
                    pdf_path.unlink()
                    out_path = png_path
                except Exception as e:
                    print(f"  [!] PDF->PNG conversion failed for {inv_num}: {e}. Keeping PDF.")
                    out_path = pdf_path
            else:
                out_path = pdf_path

            print(f"  [OK] {inv_num} ({scenario:20s}) -> {out_path.name}")

        except Exception as e:
            print(f"  [ERROR] {inv_num} ({scenario}) -> ERROR: {e}")

    print(f"\nDone! Files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
