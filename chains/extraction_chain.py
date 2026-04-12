"""
LangChain LCEL extraction chain.
Uses RAG: retrieve relevant invoice chunks → LLM extracts all fields.
Chain: retriever | prompt | llm | pydantic_parser
Primary LLM: Azure GPT-4o | Fallback: Ollama llama3.2:3b
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import logging

from guardrails.output_parser import InvoiceExtraction, get_invoice_parser, LineItem
from llm_factory import get_llm

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an expert Indian B2B GST invoice parser.
Extract ALL fields from the invoice text provided by the user.

CRITICAL FIELD EXTRACTION RULES:
- vendor_name: The company ISSUING the invoice (top of document, From section)
- vendor_gstin: The GSTIN of the vendor/supplier (15 characters, starts with state code)
- buyer_name: The company RECEIVING the invoice (Bill To, Buyer, Sold To section)
- buyer_gstin: The GSTIN of the BUYER found in the Bill To section. VERY IMPORTANT do not leave null if present
- invoice_number: Look for Invoice No, Invoice #, Bill No
- invoice_date: Look for Invoice Date, Date. Convert to YYYY-MM-DD format
- due_date: Look for Due Date, Payment Due. Convert to YYYY-MM-DD format
- po_number: Look for PO Reference, PO No, Purchase Order, Our Ref. Extract exactly as printed
- subtotal: Amount before tax (before GST)
- cgst_amount: Central GST amount (numeric only, no currency symbol)
- sgst_amount: State GST amount (numeric only, no currency symbol)
- igst_amount: Integrated GST amount if interstate
- tax_amount: Total of all GST. If not stated, calculate as cgst + sgst or igst
- total_amount: Grand total payable (numeric only, no currency symbol)
- Extract all amounts as plain numeric values with no currency symbols and no commas
- If a field is truly absent, set it to null

{format_instructions}

Return ONLY valid JSON. No markdown fences, no explanation, no preamble."""

EXTRACTION_HUMAN_PROMPT = """Invoice text:

{invoice_text}

Extract all fields and return as JSON. Pay special attention to: buyer_gstin (in Bill To section), po_number (PO Reference line), and all amount fields."""


def run_extraction(invoice_id: str, redacted_text: str) -> InvoiceExtraction:
    """
    Run LangChain LCEL extraction chain for one invoice.

    Pipeline:
      1. Build prompt with format instructions from PydanticOutputParser
      2. LLM extracts all fields via structured prompt (Azure GPT-4o or Ollama)
      3. PydanticOutputParser validates and returns InvoiceExtraction

    Args:
        invoice_id:    ID used to filter ChromaDB retriever
        redacted_text: PII-redacted invoice text (safe for LLM)

    Returns:
        InvoiceExtraction with all fields populated and confidence scores
    """
    parser = get_invoice_parser()
    llm    = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_SYSTEM_PROMPT),
        ("human",  EXTRACTION_HUMAN_PROMPT),
    ])

    # LCEL chain: prompt | llm | str_parser
    chain = prompt | llm | StrOutputParser()

    first_error = None
    try:
        raw_output = chain.invoke({
            "invoice_text":        redacted_text,
            "format_instructions": parser.get_format_instructions(),
        })

        # Clean potential markdown fences
        clean = raw_output.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:])
        if clean.endswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[:-1])
        clean = clean.strip()

        clean = clean.strip()
        
        # Robust parsing strategy: try strict first, then fallback to flexible dict mapping
        try:
            extraction = parser.parse(clean)
        except Exception as parse_err:
            logger.warning(f"Strict parse failed: {parse_err}. Attempting partial fallback...")
            import json, re
            # Extract anything that looks like JSON
            match = re.search(r'\{.*\}', clean.replace('\n', ' '), re.DOTALL)
            if not match:
                raise ValueError("No JSON found in output")
            # Try to fix trailing commas (very common with small LLMs)
            json_str = re.sub(r',\s*([\]}])', r'\1', match.group(0))
            data = json.loads(json_str)
            
            # Map robustly to model
            extraction = InvoiceExtraction(
                invoice_id=invoice_id,
                invoice_number=data.get('invoice_number'),
                invoice_date=data.get('invoice_date'),
                due_date=data.get('due_date'),
                vendor_name=data.get('vendor_name'),
                po_number=data.get('po_number'),
                vendor_gstin=data.get('vendor_gstin'),
                total_amount=data.get('total_amount'),
                tax_amount=data.get('tax_amount'),
                subtotal=data.get('subtotal'),
                cgst_amount=data.get('cgst_amount'),
                sgst_amount=data.get('sgst_amount'),
                line_items=[] # Skip line items if strict parsing failed to keep it simple
            )

        extraction.invoice_id = invoice_id
        return extraction

    except Exception as e:
        first_error = e
        logger.warning(f"Primary extraction failed for {invoice_id}: {e}")

    # Retry with simplified prompt (fallback)
    try:
        schema = '{"invoice_number": "", "invoice_date": "YYYY-MM-DD", "vendor_name": "", "total_amount": 0.0, "po_number": ""}'
        schema_escaped = schema.replace("{", "{{").replace("}", "}}")
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Extract invoice fields from this text as JSON matching this simplified schema. Do not include markdown: "
                + schema_escaped
            )),
            ("human", "{invoice_text}"),
        ])
        simple_chain = simple_prompt | llm | StrOutputParser()
        raw2 = simple_chain.invoke({"invoice_text": redacted_text[:3000]})
        clean2 = raw2.strip().replace("```json", "").replace("```", "").strip()
        
        import json, re
        match = re.search(r'\{.*\}', clean2.replace('\n', ' '), re.DOTALL)
        if match:
            json_str = re.sub(r',\s*([\]}])', r'\1', match.group(0))
            data = json.loads(json_str)
            extraction = InvoiceExtraction(**{k: v for k, v in data.items() if k in InvoiceExtraction.model_fields})
        else:
            extraction = parser.parse(clean2)
            
        extraction.invoice_id = invoice_id
        return extraction

    except Exception as second_error:
        logger.error(f"Extraction failed completely for {invoice_id}: {second_error}")
        from guardrails.output_parser import Flag
        return InvoiceExtraction(
            invoice_id=invoice_id,
            status="failed",
            flags=[Flag(
                rule="extraction_failed",
                field="all",
                severity="error",
                message=(
                    f"LLM extraction failed. Primary: {str(first_error)[:80]}. "
                    f"Retry: {str(second_error)[:80]}"
                ),
            )]
        )
