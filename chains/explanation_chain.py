"""
LangChain LCEL chain for plain-English flag explanations.
Uses Azure GPT-4o (or Ollama fallback) to explain each validation flag
in plain English for a non-technical finance team reviewer.
Chain: prompt | llm | StrOutputParser
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

from guardrails.output_parser import InvoiceExtraction, Flag
from llm_factory import get_llm

logger = logging.getLogger(__name__)

_explanation_chain = None


def get_explanation_chain():
    """
    Build and cache LCEL explanation chain (prompt | llm | str_parser).

    Returns:
        Cached LCEL chain for generating flag explanations
    """
    global _explanation_chain
    if _explanation_chain is None:
        llm    = get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a finance audit assistant helping a non-technical AP team. "
                "Explain invoice validation issues in plain English. "
                "Be specific with numbers and amounts where mentioned. "
                "State exactly what the reviewer must check or do. "
                "Keep under 40 words. No preamble."
            )),
            ("human", (
                "Invoice: {invoice_number} | Vendor: {vendor_name} | Total: ₹{total_amount}\n"
                "Flag rule: {flag_rule}\n"
                "Flag message: {flag_message}\n\n"
                "Plain English explanation and action required:"
            )),
        ])
        _explanation_chain = prompt | llm | StrOutputParser()
    return _explanation_chain


def explain_flag(flag: Flag, extraction: InvoiceExtraction) -> str:
    """
    Generate plain-English explanation for one validation flag.

    Args:
        flag:       The validation Flag to explain
        extraction: Invoice context (number, vendor, total) for the prompt

    Returns:
        Short explanation string under 40 words; falls back to raw message on error
    """
    try:
        chain = get_explanation_chain()
        return chain.invoke({
            "invoice_number": extraction.invoice_number or "Unknown",
            "vendor_name":    extraction.vendor_name    or "Unknown vendor",
            "total_amount":   (
                f"{extraction.total_amount:,.2f}"
                if extraction.total_amount else "Unknown"
            ),
            "flag_rule":    flag.rule,
            "flag_message": flag.message,
        })
    except Exception as e:
        logger.warning(f"Explanation generation failed: {e}")
        return flag.message   # Graceful fallback — return raw message
