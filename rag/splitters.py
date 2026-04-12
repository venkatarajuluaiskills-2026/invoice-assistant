"""
LangChain Text Splitters for invoice RAG pipeline.
Structured content (tables, line items): smaller chunks, preserve rows.
Unstructured content (paragraphs, payment terms): larger semantic chunks.
Mixed content: both splitters applied and chunks merged.
"""
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from langchain_core.documents import Document
from typing import List
from config import (
    CHUNK_SIZE_STRUCTURED, CHUNK_SIZE_UNSTRUCTURED, CHUNK_OVERLAP,
)


def get_structured_splitter() -> RecursiveCharacterTextSplitter:
    """
    Splitter for structured invoice content (tables, line items).
    Smaller chunks with table-aware separators to preserve rows.

    Returns:
        RecursiveCharacterTextSplitter configured for structured invoice data
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_STRUCTURED,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " | ", "  ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def get_unstructured_splitter() -> RecursiveCharacterTextSplitter:
    """
    Splitter for unstructured content (paragraphs, terms, notes).
    Larger chunks with sentence-aware separators.

    Returns:
        RecursiveCharacterTextSplitter configured for unstructured invoice text
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_UNSTRUCTURED,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Split LangChain Documents using the correct strategy for each doc's
    data_type metadata (set by InvoicePDFLoader / InvoiceImageLoader).

    Strategy selection:
      - "structured"   → structured splitter (smaller chunks, table-aware)
      - "unstructured" → unstructured splitter (larger semantic chunks)
      - "mixed"        → both splitters applied; chunks merged together

    Args:
        docs: Documents from a LangChain Invoice Loader

    Returns:
        Chunked Documents with preserved metadata + chunk_type and chunk_index
    """
    structured_splitter   = get_structured_splitter()
    unstructured_splitter = get_unstructured_splitter()
    all_chunks            = []

    for doc in docs:
        data_type = doc.metadata.get("data_type", "unstructured")

        if data_type == "structured":
            chunks = structured_splitter.split_documents([doc])
            for i, c in enumerate(chunks):
                c.metadata["chunk_type"]  = "structured"
                c.metadata["chunk_index"] = i

        elif data_type == "unstructured":
            chunks = unstructured_splitter.split_documents([doc])
            for i, c in enumerate(chunks):
                c.metadata["chunk_type"]  = "unstructured"
                c.metadata["chunk_index"] = i

        else:  # mixed — apply both splitters and combine results
            s_chunks = structured_splitter.split_documents([doc])
            u_chunks = unstructured_splitter.split_documents([doc])
            for i, c in enumerate(s_chunks):
                c.metadata["chunk_type"]  = "structured"
                c.metadata["chunk_index"] = i
            for i, c in enumerate(u_chunks):
                c.metadata["chunk_type"]  = "unstructured"
                c.metadata["chunk_index"] = len(s_chunks) + i
            chunks = s_chunks + u_chunks

        all_chunks.extend(chunks)

    return all_chunks
