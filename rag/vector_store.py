"""
ChromaDB vector store via LangChain wrapper.
Uses Ollama gte-large embeddings (local — zero Azure token cost).
Handles document upsert, similarity retrieval, and duplicate detection.
All vectors persisted to local disk at ./chroma_db
"""
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Tuple, Optional
from datetime import datetime
import streamlit as st
from config import CHROMA_DB_PATH, COLLECTION_NAME, DUPLICATE_SIMILARITY_THRESHOLD
from llm_factory import get_embeddings


@st.cache_resource
def get_vectorstore() -> Chroma:
    """
    Singleton ChromaDB instance persisted to ./chroma_db.
    Uses Ollama gte-large embeddings — local inference, no API cost.

    Returns:
        Chroma vector store instance
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DB_PATH,
    )


def upsert_invoice_chunks(
    invoice_id: str,
    filename: str,
    chunks: List[Document],
) -> None:
    """
    Embed and store all invoice chunks in ChromaDB.
    Adds invoice_id, filename, and timestamp to each chunk's metadata.

    Args:
        invoice_id: Unique ID for this invoice (uuid prefix)
        filename:   Original uploaded filename
        chunks:     Text chunks from split_documents()
    """
    vs    = get_vectorstore()
    texts = [c.page_content for c in chunks]

    metadatas = []
    for c in chunks:
        meta = dict(c.metadata)
        meta["invoice_id"] = invoice_id
        meta["filename"]   = filename
        meta["timestamp"]  = datetime.utcnow().isoformat()
        metadatas.append(meta)

    ids = [f"{invoice_id}_chunk_{i}" for i in range(len(chunks))]
    vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)


def get_invoice_retriever(invoice_id: str):
    """
    Get LangChain Retriever filtered to a specific invoice_id.
    Used by the extraction chain to retrieve only relevant chunks.

    Args:
        invoice_id: Invoice ID to filter results to

    Returns:
        LangChain VectorStoreRetriever for this invoice only
    """
    vs = get_vectorstore()
    return vs.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k":      5,
            "filter": {"invoice_id": invoice_id},
        },
    )


def check_duplicate(
    invoice_id: str,
    query_text: str,
) -> Tuple[bool, Optional[str], float]:
    """
    Check if a new invoice is a near-duplicate of an existing one.
    Uses cosine similarity of first chunk embedding against all
    existing invoices (excluding current invoice_id).

    Args:
        invoice_id:  ID of the current (new) invoice being checked
        query_text:  Text to use as similarity query (first chunk)

    Returns:
        Tuple of (is_duplicate, matching_invoice_id, similarity_score)
    """
    vs = get_vectorstore()
    try:
        results = vs.similarity_search_with_relevance_scores(
            query=query_text,
            k=2,
            filter={"invoice_id": {"$ne": invoice_id}},
        )
        if not results:
            return (False, None, 0.0)

        top_doc, score = results[0]
        if score >= DUPLICATE_SIMILARITY_THRESHOLD:
            return (True, top_doc.metadata.get("invoice_id"), float(score))
        return (False, None, float(score))

    except Exception:
        return (False, None, 0.0)
