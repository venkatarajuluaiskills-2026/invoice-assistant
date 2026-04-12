"""
LangChain custom Document Loaders for invoice files.
InvoicePDFLoader  — extracts text from PDF via PyMuPDF + OCR fallback
InvoiceImageLoader — extracts text from PNG/JPG/TIFF via Tesseract OCR
Both return LangChain Documents with rich metadata for downstream RAG.
"""
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from typing import List
from PIL import Image
from io import BytesIO
import uuid
import logging

from ingest.ocr_engine import preprocess_image, extract_text_from_image

logger = logging.getLogger(__name__)


def _detect_content_type(text: str) -> str:
    """
    Detect whether invoice content is structured, unstructured, or mixed.
    Structured = tables, pipe-delimited data, numeric columns.
    Unstructured = paragraphs, payment terms, notes.

    Args:
        text: Extracted invoice text

    Returns:
        "structured" | "unstructured" | "mixed"
    """
    lines = text.split('\n')
    structured_score   = 0
    unstructured_score = 0

    for line in lines:
        if '|' in line or '\t' in line:
            structured_score += 1
        if any(k in line.upper() for k in [
            'TOTAL', 'AMOUNT', 'QTY', 'PRICE',
            'GSTIN', 'HSN', 'SAC', 'CGST', 'SGST',
            'SUBTOTAL', 'TAX', 'INVOICE',
        ]):
            structured_score += 1
        words = line.split()
        if len(words) > 8:
            unstructured_score += 1
        if any(p in line for p in ['. ', ', ', '; ']):
            unstructured_score += 1

    if structured_score >= 3 and unstructured_score >= 3:
        return "mixed"
    elif structured_score > unstructured_score:
        return "structured"
    else:
        return "unstructured"


class InvoicePDFLoader(BaseLoader):
    """
    LangChain Document Loader for PDF invoice files.
    Strategy:
      1. Try PyMuPDF native text extraction (fast, accurate for digital PDFs)
      2. Fallback to page-to-image + Tesseract OCR (for scanned/image PDFs)
    Each page becomes one LangChain Document with metadata.
    """

    def __init__(self, file_bytes: bytes, filename: str) -> None:
        """
        Args:
            file_bytes: Raw bytes of the PDF file
            filename:   Original filename for metadata
        """
        self.file_bytes = file_bytes
        self.filename   = filename
        self.invoice_id = str(uuid.uuid4())[:8]

    def load(self) -> List[Document]:
        """Load PDF and return list of LangChain Documents (one per page)."""
        import fitz  # PyMuPDF
        docs = []
        pdf  = fitz.open(stream=self.file_bytes, filetype="pdf")

        for page_num, page in enumerate(pdf):
            # Try native text extraction first
            text = page.get_text("text").strip()

            if len(text) < 100:
                # Fallback: render page as image, run Tesseract OCR
                mat = fitz.Matrix(2.0, 2.0)   # 2x zoom for better OCR quality
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img = preprocess_image(img)
                text = extract_text_from_image(img)

            content_type = _detect_content_type(text)

            docs.append(Document(
                page_content=text,
                metadata={
                    "invoice_id":  self.invoice_id,
                    "filename":    self.filename,
                    "page":        page_num + 1,
                    "total_pages": len(pdf),
                    "source":      "pdf",
                    "data_type":   content_type,
                    "loader":      "InvoicePDFLoader",
                }
            ))

        pdf.close()
        return docs


class InvoiceImageLoader(BaseLoader):
    """
    LangChain Document Loader for image invoice files (PNG/JPG/TIFF).
    Preprocesses image for optimal OCR using Tesseract.
    Returns a single LangChain Document with full page text.
    """

    def __init__(self, file_bytes: bytes, filename: str) -> None:
        """
        Args:
            file_bytes: Raw bytes of the image file
            filename:   Original filename for metadata
        """
        self.file_bytes = file_bytes
        self.filename   = filename
        self.invoice_id = str(uuid.uuid4())[:8]

    def load(self) -> List[Document]:
        """Load image and return list with one LangChain Document."""
        img  = Image.open(BytesIO(self.file_bytes))
        img  = preprocess_image(img)
        text = extract_text_from_image(img)

        content_type = _detect_content_type(text)

        return [Document(
            page_content=text,
            metadata={
                "invoice_id":  self.invoice_id,
                "filename":    self.filename,
                "page":        1,
                "total_pages": 1,
                "source":      "image",
                "data_type":   content_type,
                "loader":      "InvoiceImageLoader",
            }
        )]


def load_invoice(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Route file to correct loader based on file extension.

    Args:
        file_bytes: Raw file bytes from Streamlit uploader
        filename:   Original filename (used to detect file type)

    Returns:
        List of LangChain Documents from the appropriate loader

    Raises:
        ValueError: If file type is not supported
    """
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return InvoicePDFLoader(file_bytes, filename).load()
    elif ext in ("png", "jpg", "jpeg", "tiff", "tif"):
        return InvoiceImageLoader(file_bytes, filename).load()
    else:
        raise ValueError(
            f"Unsupported file type: .{ext}. Supported formats: PDF, PNG, JPG, TIFF"
        )
