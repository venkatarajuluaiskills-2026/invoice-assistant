"""
LangChain LCEL batch summary chain.
Generates an executive summary for the AP finance manager
after processing a batch of invoices.
Chain: prompt | llm | StrOutputParser
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from collections import Counter
import logging

from guardrails.output_parser import InvoiceExtraction
from llm_factory import get_llm

logger = logging.getLogger(__name__)


def run_batch_summary(results: list) -> str:
    """
    Generate executive summary for processed invoice batch.

    Args:
        results: List of InvoiceExtraction objects from the batch

    Returns:
        100–120 word professional executive summary string;
        falls back to a plain-text summary if LLM fails
    """
    if not results:
        return "No invoices processed yet."

    total         = len(results)
    auto_approved = sum(1 for r in results if r.status == "auto_approved")
    needs_review  = sum(1 for r in results if r.status == "needs_review")
    failed        = sum(1 for r in results if r.status == "failed")
    three_way     = sum(1 for r in results if r.match_status == "3way_matched")
    total_value   = sum(r.total_amount for r in results if r.total_amount)
    top_vendors   = Counter(
        r.vendor_name for r in results if r.vendor_name
    ).most_common(3)
    common_flags = Counter(
        f.rule for r in results for f in r.flags
    ).most_common(3)

    llm    = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a finance operations analyst at TCS. "
            "Write concise, professional executive summaries for AP batch reports. "
            "Use the actual numbers provided. Flag what needs immediate attention. "
            "100-120 words. Plain English. No bullet points."
        )),
        ("human", (
            "Batch statistics:\n"
            "- Total invoices: {total}\n"
            "- Auto-approved (3-way matched): {auto_approved}\n"
            "- Needs review: {needs_review}\n"
            "- Failed/Rejected: {failed}\n"
            "- 3-way matched: {three_way}\n"
            "- Total invoice value: ₹{total_value}\n"
            "- Top vendors: {top_vendors}\n"
            "- Most common issues: {common_flags}\n\n"
            "Write the executive summary:"
        )),
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        return chain.invoke({
            "total":         total,
            "auto_approved": auto_approved,
            "needs_review":  needs_review,
            "failed":        failed,
            "three_way":     three_way,
            "total_value":   f"{total_value:,.0f}",
            "top_vendors":   ", ".join(v[0] for v in top_vendors if v[0]) or "N/A",
            "common_flags":  ", ".join(r[0] for r in common_flags) or "None",
        })
    except Exception as e:
        logger.error(f"Batch summary LLM call failed: {e}")
        return (
            f"Processed {total} invoices. "
            f"{auto_approved} auto-approved, {needs_review} need review, "
            f"{failed} failed/rejected. "
            f"Total invoice value: ₹{total_value:,.2f}. "
            f"Top vendors: {', '.join(v[0] for v in top_vendors if v[0]) or 'N/A'}."
        )
