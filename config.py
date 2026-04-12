"""
Central configuration for Invoice Processing Assistant.
Supports both Azure GenAI Lab MaaS and Ollama local SLM modes.
Primary: Azure GPT-4o | Fallback: Ollama llama3.2 | Embed: Ollama gte-large
"""
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(override=True)

# ── LLM Mode ─────────────────────────────────────────────────────────────────
USE_AZURE = os.getenv("USE_AZURE", "true").lower() == "true"

# ── Azure GenAI Lab MaaS ──────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
LLM_MODEL                = os.getenv("LLM_MODEL", "azure/genailab-maas-gpt-4o")
REASONING_MODEL          = os.getenv("REASONING_MODEL", "azure_ai/genailab-maas-DeepSeek-R1")

# ── Ollama Local SLMs ─────────────────────────────────────────────────────────
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_LLM_MODEL   = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ── LLM parameters ───────────────────────────────────────────────────────────
LLM_TEMPERATURE = 0
LLM_TIMEOUT     = 60      # Ollama local can be slower
LLM_MAX_RETRIES = 2

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "invoices"

# ── RAG ──────────────────────────────────────────────────────────────────────
CHUNK_SIZE_STRUCTURED   = 300
CHUNK_SIZE_UNSTRUCTURED = 500
CHUNK_OVERLAP           = 50
TOP_K_RETRIEVAL         = 5

# ── Thresholds ────────────────────────────────────────────────────────────────
AUTO_APPROVE_THRESHOLD         = 0.92
NEEDS_REVIEW_THRESHOLD         = 0.60
DUPLICATE_SIMILARITY_THRESHOLD = 0.95

# ── PO matching ───────────────────────────────────────────────────────────────
PO_FUZZY_MATCH_THRESHOLD = 85
PRICE_TOLERANCE_PCT      = 0.03
AMOUNT_ROUNDING_INR      = 1.00

# ── Paths ─────────────────────────────────────────────────────────────────────
EXPORT_DIR      = "./exports"
LOG_DIR         = "./logs"
INVOICE_DIR     = "./synthetic/invoices"
PO_MASTER_PATH  = "./synthetic/po_master.json"
GRN_MASTER_PATH = "./synthetic/grn_master.json"

for d in [EXPORT_DIR, LOG_DIR, INVOICE_DIR, CHROMA_DB_PATH]:
    Path(d).mkdir(parents=True, exist_ok=True)
