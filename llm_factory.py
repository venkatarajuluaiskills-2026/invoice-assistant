import os
import logging
import streamlit as st

logger = logging.getLogger(__name__)


@st.cache_resource
def get_llm(reasoning: bool = False):
    """
    Get primary LLM as LangChain singleton.
    Priority: Azure → Groq (free cloud, fast) → Ollama (local, slow).
    """
    from config import (
        USE_AZURE, LLM_TEMPERATURE, LLM_TIMEOUT, LLM_MAX_RETRIES,
        AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
        LLM_MODEL, REASONING_MODEL,
        OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL,
    )

    # 1. Try to get keys from st.secrets (Cloud) then os.environ (Local)
    groq_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY", "")
    azure_key = st.secrets.get("AZURE_OPENAI_API_KEY") or AZURE_OPENAI_API_KEY

    # 1. Azure (enterprise)
    if USE_AZURE and azure_key and AZURE_OPENAI_ENDPOINT:
        try:
            from langchain_openai import AzureChatOpenAI
            model      = REASONING_MODEL if reasoning else LLM_MODEL
            deployment = model.split("/")[-1]
            llm = AzureChatOpenAI(
                azure_deployment=deployment,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=azure_key,
                api_version=AZURE_OPENAI_API_VERSION,
                temperature=LLM_TEMPERATURE,
                request_timeout=LLM_TIMEOUT,
                max_retries=LLM_MAX_RETRIES,
            )
            logger.info(f"Azure LLM initialised: {deployment}")
            return llm
        except Exception as e:
            logger.warning(f"Azure LLM failed ({e}), trying Groq...")

    # 2. Groq — free cloud inference, very fast
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            groq_model = st.secrets.get("GROQ_MODEL") or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            llm = ChatGroq(
                api_key=groq_key,
                model=groq_model,
                temperature=LLM_TEMPERATURE,
                max_retries=LLM_MAX_RETRIES,
            )
            logger.info(f"Groq LLM initialised: {groq_model}")
            return llm
        except Exception as e:
            logger.warning(f"Groq LLM failed ({e}), falling back to Ollama")

    # 3. Ollama — local fallback (slow on CPU)
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            timeout=LLM_TIMEOUT,
            num_ctx=2048,
        )
        logger.info(f"Ollama LLM initialised: {OLLAMA_LLM_MODEL}")
        return llm
    except Exception as e:
        logger.error(f"Ollama fallback failed: {e}")
        return None


@st.cache_resource
def get_embeddings():
    """
    Get embedding model as LangChain singleton.
    Priority: Ollama gte-large (local) → HuggingFace MiniLM (cloud/CPU).
    """
    from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL

    # Try Ollama first (local development)
    try:
        from langchain_ollama import OllamaEmbeddings
        import requests
        # Quick health check to see if Ollama is running
        requests.get(OLLAMA_BASE_URL, timeout=1)
        
        embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        logger.info(f"Embeddings initialised: {OLLAMA_EMBED_MODEL} (local Ollama)")
        return embeddings
    except Exception:
        # Cloud Fallback: Use HuggingFace (runs on CPU, no API key needed)
        logger.info("Ollama unreachable. Initialising HuggingFace Embeddings (Cloud Mode)...")
        from langchain_huggingface import HuggingFaceEmbeddings
        # all-MiniLM-L6-v2 is small, fast, and high quality
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embeddings
