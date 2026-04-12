"""
LangChain LCEL conversational chain for Q&A over processed invoices.
Uses the extracted structured data directly in the context for accurate answers.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import json

from llm_factory import get_llm
from guardrails.output_parser import InvoiceExtraction

CHAT_SYSTEM_PROMPT = """You are a precise AI Finance Assistant. Answer questions based ONLY on the invoice data below.

PROCESSED INVOICE DATA:
{invoice_context}

RULES:
1. Give exact, direct answers. No filler phrases like "Based on the data" or "I am an AI".
2. If asked for "full details" or "all details" of an invoice, list EVERY non-null field clearly.
3. If asked WHY an invoice "needs_review" or has a certain status, explain using the "flags" list — each flag has a rule, field, severity, and message. Summarise each flag clearly.
4. If asked about PII (phone, aadhaar, pan, bank account, email), say: "Redacted — sensitive data."
5. If a field is null / not found in the data, say "Not available."
6. Use ₹ for currency. Format amounts with commas.
7. For follow-up questions referring to "it" or "this invoice", use the last discussed invoice.
"""

def generate_invoice_context(results: list[InvoiceExtraction]) -> str:
    """Format ALL extracted fields into a rich context string for the LLM."""
    if not results:
        return "No invoices processed yet."

    context_blocks = []
    for r in results:
        flags_detail = []
        for f in r.flags:
            flags_detail.append({
                "rule":     f.rule,
                "field":    f.field,
                "severity": f.severity,
                "message":  f.message,
            })

        data = {
            "filename":           r.filename,
            "invoice_id":         r.invoice_id,
            "invoice_number":     r.invoice_number,
            "invoice_date":       r.invoice_date,
            "due_date":           r.due_date,
            "vendor_name":        r.vendor_name,
            "vendor_gstin":       r.vendor_gstin,
            "po_number":          r.po_number,
            "buyer_name":         r.buyer_name,
            "buyer_gstin":        r.buyer_gstin,
            "place_of_supply":    r.place_of_supply,
            "payment_terms":      r.payment_terms,
            "subtotal":           f"₹{r.subtotal:,.2f}" if r.subtotal else None,
            "cgst_amount":        f"₹{r.cgst_amount:,.2f}" if r.cgst_amount else None,
            "sgst_amount":        f"₹{r.sgst_amount:,.2f}" if r.sgst_amount else None,
            "igst_amount":        f"₹{r.igst_amount:,.2f}" if r.igst_amount else None,
            "tax_amount":         f"₹{r.tax_amount:,.2f}" if r.tax_amount else None,
            "total_amount":       f"₹{r.total_amount:,.2f}" if r.total_amount else None,
            "amount_in_words":    r.amount_in_words,
            "bank_name":          r.bank_name,
            "status":             r.status,
            "match_status":       r.match_status,
            "matched_po_number":  r.matched_po_number,
            "po_match_score":     r.po_match_score,
            "match_recommendation": r.match_recommendation,
            "duplicate_of":       r.duplicate_of,
            "flags":              flags_detail,
            "flag_count":         len(r.flags),
            "line_items":         [
                {
                    "description": li.description,
                    "hsn_code":    li.hsn_code,
                    "quantity":    li.quantity,
                    "unit_price":  li.unit_price,
                    "gst_rate":    li.gst_rate,
                    "amount":      li.amount,
                }
                for li in (r.line_items or [])
            ],
        }
        context_blocks.append(json.dumps(data, indent=2, default=str))

    return "\n\n---\n\n".join(context_blocks)

def run_chat(user_question: str, chat_history: list, results_context: list[InvoiceExtraction]) -> str:
    """
    Run the conversational chain.
    """
    llm = get_llm()
    invoice_context_str = generate_invoice_context(results_context)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", CHAT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "invoice_context": invoice_context_str,
        "history": chat_history,
        "question": user_question,
    })
