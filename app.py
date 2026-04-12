# ── DIAGNOSTIC HEARTBEAT ───────────────────────────────────────────────────
print("\n" + "="*50)
print("🚀 TCS INVOICE ASSISTANT STARTING...")
print("="*50 + "\n")

# ── CHROMADB SQLITE FIX (FOR CLOUD DEPLOYMENT) ────────────────────────────────
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import json
from pathlib import Path

st.set_page_config(
    page_title="AI Invoice Assistant",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium dark-mode styling ─────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  .main .block-container {
    padding-top: 1rem;
    max-width: 1400px;
  }

  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(26,35,126,0.08) 0%, rgba(26,35,126,0.02) 100%);
    padding: 14px 18px;
    border-radius: 10px;
    border: 1px solid rgba(26,35,126,0.15);
    transition: transform 0.15s ease;
  }

  div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    border-color: rgba(26,35,126,0.35);
  }

  div[data-testid="metric-container"] [data-testid="metric-label"] {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    opacity: 0.7;
  }

  div[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 1.5rem;
    font-weight: 700;
  }

  .stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
  }

  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a237e, #283593);
    border: none;
    color: white;
    box-shadow: 0 4px 15px rgba(26,35,126,0.3);
  }

  .stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(26,35,126,0.4);
  }

  .stSelectbox > div > div {
    border-radius: 8px;
  }

  .stExpander {
    border-radius: 8px;
  }

  .stTabs [data-baseweb="tab"] {
    font-weight: 600;
  }

  /* Invoice status badge colors */
  div[class="stAlert"] {
    border-radius: 10px;
  }
</style>
""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from config import (
    PO_MASTER_PATH, GRN_MASTER_PATH,
    USE_AZURE, LLM_MODEL, OLLAMA_LLM_MODEL,
)
from ingest.loaders import load_invoice
from ingest.pii_redactor import redact, log_pii_event
from rag.splitters import split_documents
from rag.vector_store import upsert_invoice_chunks, check_duplicate
from chains.extraction_chain import run_extraction
from chains.validation_chain import run_validation
from chains.explanation_chain import explain_flag
from chains.summary_chain import run_batch_summary
from guardrails.output_parser import InvoiceExtraction
from callbacks.audit_callback import log_audit_event
from chains.chat_chain import run_chat
from chains.smart_demo_chat import smart_demo_answer


@st.cache_data
def load_databases():
    """
    Load PO and GRN master data once at startup.

    Returns:
        (po_database, grn_database) as lists of dicts
    """
    po_path  = Path(PO_MASTER_PATH)
    grn_path = Path(GRN_MASTER_PATH)
    po_db    = json.loads(po_path.read_text(encoding="utf-8"))  if po_path.exists()  else []
    grn_db   = json.loads(grn_path.read_text(encoding="utf-8")) if grn_path.exists() else []
    return po_db, grn_db


po_database, grn_database = load_databases()

# ── Session state defaults ────────────────────────────────────────────────────
defaults = {
    "results":        [],
    "redaction_maps": {},
    "explanations":   {},
    "batch_summary":  "",
    "filenames":      [],
    "chat_history":   [
        ("ai", "👋 **Hi! I am your AI Finance Assistant.** \n\nI can instantly extract data from your PDFs, perform 3-way matching against your master PO records, and answer any complex audit questions in natural language.\n\n👈 **Please upload your invoices in the sidebar to begin!**")
    ],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0;">
        <div style="font-size:3rem; font-weight:900; color:#1a237e; letter-spacing:-1px;">Invoice <span style="color:#0080ff;">AI</span></div>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("### 📂 Upload Invoices")
    st.caption("Supports: **PDF**, **PNG**, **JPG**, **TIFF**")

    uploaded_files = st.file_uploader(
        label="Drop invoice files here",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
        accept_multiple_files=True,
        key="invoice_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) ready")
        for f in uploaded_files:
            size_kb = len(f.getvalue()) / 1024
            st.caption(f"• {f.name[:30]} — {size_kb:.0f} KB")

    st.divider()
    
    st.session_state["demo_mode"] = st.checkbox("🚀 Enable Fast Demo Mode", value=True, help="Bypasses the slow heavy Local AI model and uses simulated extraction data. Perfect for live Hackathon presentations on standard laptops!")
    st.divider()

    # Stats snapshot
    if st.session_state.results:
        n       = len(st.session_state.results)
        approved = sum(1 for r in st.session_state.results if r.status == "auto_approved")
        review   = sum(1 for r in st.session_state.results if r.status == "needs_review")
        failed   = sum(1 for r in st.session_state.results if r.status == "failed")
        st.markdown(
            f"**Processed:** {n}  \n"
            f"✅ Approved: {approved} | ⚠️ Review: {review} | ❌ Failed: {failed}"
        )
        st.divider()

    process_btn = st.button(
        "⚡ Process All Invoices",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

# ── Processing pipeline ───────────────────────────────────────────────────────
if process_btn and uploaded_files:
    # Clear previous results
    st.session_state.results        = []
    st.session_state.redaction_maps = {}
    st.session_state.explanations   = {}
    st.session_state.filenames      = []

    progress_bar = st.sidebar.progress(0)
    status_text  = st.sidebar.empty()

    for idx, file in enumerate(uploaded_files):
        status_text.text(f"⏳ {idx+1}/{len(uploaded_files)}: {file.name[:30]}")

        try:
            # Step 1: LangChain Document Loader ────────────────────────────────
            docs       = load_invoice(file.getvalue(), file.name)
            invoice_id = docs[0].metadata["invoice_id"]
            full_text  = "\n\n".join(d.page_content for d in docs)

            # Step 2: PII Redaction (Presidio — fully local) ───────────────────
            redacted_text, redaction_map = redact(full_text)

            if redaction_map:
                entity_types = [p.split("_")[1] for p in redaction_map if "_" in p]
                log_pii_event(invoice_id, entity_types)
            st.session_state.redaction_maps[invoice_id] = redaction_map

            # Update doc content with redacted text for embedding
            for doc in docs:
                doc.page_content = redacted_text[:len(doc.page_content)]

            # Step 3: LangChain Text Splitter ─────────────────────────────────
            chunks = split_documents(docs)
            if not chunks:
                chunks = docs   # Fallback: use full docs if splitting produced nothing

            # Step 4: Duplicate detection (vector similarity) ─────────────────
            query_text              = chunks[0].page_content if chunks else redacted_text[:500]
            is_dup, dup_id, dup_score = check_duplicate(invoice_id, query_text)

            # Step 5: ChromaDB upsert ──────────────────────────────────────────
            upsert_invoice_chunks(invoice_id, file.name, chunks)

            # Step 6: LCEL Extraction Chain ────────────────────────────────────
            if st.session_state.get("demo_mode"):
                import time
                time.sleep(1.5) # Simulate processing
                from guardrails.output_parser import InvoiceExtraction
                extraction = InvoiceExtraction(
                    invoice_id=invoice_id,
                    invoice_number="INV-" + invoice_id[:4].upper(),
                    invoice_date="2026-04-01",
                    due_date="2026-05-01",
                    vendor_name="Demo India Pvt Ltd",
                    vendor_gstin="27AADCB2230M1Z2",
                    po_number="PO-2026-1002",
                    total_amount=54900.00,
                    tax_amount=4900.00,
                    subtotal=50000.00,
                    cgst_amount=2450.00,
                    sgst_amount=2450.00,
                    line_items=[]
                )
            else:
                extraction = run_extraction(invoice_id, redacted_text)
                
            extraction.filename = file.name

            if is_dup:
                extraction.duplicate_of         = dup_id
                extraction.duplicate_similarity = dup_score

            if st.session_state.get("demo_mode"):
                # Demo Mode: skip heavy chains, mark as clean approved
                extraction.status             = "auto_approved"
                extraction.match_status       = "3way_matched"
                extraction.flags              = []
                extraction.match_recommendation = "Auto-approved via Demo Mode."
            else:
                # Step 7: LCEL Validation Chain ──────────────────────────────
                extraction = run_validation(extraction, redacted_text)
                # Step 8: LCEL Explanation Chain ─────────────────────────────
                flag_explanations = {}
                for i, flag in enumerate(extraction.flags[:5]):
                    if flag.severity in ("error", "warning"):
                        flag_explanations[i] = explain_flag(flag, extraction)
                st.session_state.explanations[invoice_id] = flag_explanations

            # Step 9: Store results ─────────────────────────────────────────
            st.session_state.results.append(extraction)
            st.session_state.filenames.append(file.name)

            # Step 10: Audit log ────────────────────────────────────────────
            log_audit_event(
                invoice_id,
                "invoice_processed",
                {
                    "filename":     file.name,
                    "status":       extraction.status,
                    "match_status": extraction.match_status,
                    "flag_count":   len(extraction.flags),
                    "llm_mode":     "demo" if st.session_state.get("demo_mode") else ("azure" if USE_AZURE else "ollama"),
                },
            )

        except Exception as e:
            import traceback
            err_str = str(e)
            
            if "tesseract is not installed" in err_str.lower() or "tesseract" in err_str.lower():
                st.sidebar.error(f"❌ {file.name}: Tesseract OCR is not installed on this PC.")
                st.sidebar.caption("💡 **Fix:** Please upload PDF files instead of Images (PNG/JPG), OR install Tesseract-OCR on your Windows machine.")
            elif "10061" in err_str or "connection refused" in err_str.lower() or "failed to connect to ollama" in err_str.lower():
                st.sidebar.error(f"❌ {file.name}: Could not connect to Ollama.")
                st.sidebar.caption("💡 **Fix:** Ollama is not running on this PC. Please start Ollama locally, OR switch to Azure mode by adding your API key in the `.env` file.")
            else:
                st.sidebar.error(f"❌ {file.name}: {err_str[:80]}")
                with st.sidebar.expander("Error details"):
                    st.code(traceback.format_exc())

        progress_bar.progress((idx + 1) / len(uploaded_files))

    # Step 11: Auto-generate AI Response ───────────────────────────────────────
    if st.session_state.results:
        # Pushing a brief prompt to invite the user to chat about the visuals
        st.session_state.chat_history.append(("ai", "✨ I have successfully extracted and verified all documents against the PO master! My visual analysis is populated above. What specific details would you like me to dive into?"))

    n = len(st.session_state.results)
    status_text.text(f"✅ Done — {n} invoice(s) processed")
    progress_bar.progress(1.0)
    st.rerun()

# ── Main page ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Premium Dark Mode Glassmorphism Theme */
  .stApp {
      background-color: #0F172A;
  }
  .glass-card {
      background: rgba(30, 41, 59, 0.7);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
      transition: all 0.3s ease;
  }
  .glass-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 32px rgba(79, 70, 229, 0.15);
      border: 1px solid rgba(79, 70, 229, 0.3);
  }
  .metric-value {
      font-size: 2.2rem;
      font-weight: 800;
      background: linear-gradient(135deg, #818cf8, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
  }
  .metric-label {
      color: #94a3b8;
      font-size: 0.85rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 8px;
  }
  .error-text { color: #f87171; font-weight: 600; }
  .success-text { color: #34d399; font-weight: 600; }
  
  .stChatMessage {
      background: rgba(30, 41, 59, 0.4) !important;
      border: 1px solid rgba(255, 255, 255, 0.05) !important;
      border-radius: 16px;
  }
  [data-testid="stChatMessage"]:nth-child(even) {
      background: rgba(79, 70, 229, 0.05) !important;
      border-left: 3px solid #818cf8 !important;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display: flex; align-items: center; gap: 16px; margin-bottom: 0.5rem;">
  <span style="font-size: 2.5rem; background: linear-gradient(135deg, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">⚡</span>
  <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #f8fafc;">AI Invoice Assistant</h1>
</div>
""", unsafe_allow_html=True)

if not st.session_state.results:
    st.markdown("""
    <div style="margin-top: 4rem; text-align: center; animation: fadeIn 1s ease;">
      <div style="font-size: 4rem; opacity: 0.5; margin-bottom: 1rem;">🌌</div>
      <h3 style="color: #cbd5e1; font-weight: 500;">Awaiting Documents</h3>
      <p style="color: #64748b; font-size: 1.1rem;">Upload invoices in the sidebar to initialize the AI extraction pipeline.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # --- VISUAL DASHBOARD ---
    st.markdown("### 📊 Extraction Overview")
    cols = st.columns(3)
    
    total_val = sum(r.total_amount for r in st.session_state.results if r.total_amount)
    err_count = sum(1 for r in st.session_state.results if r.status != "auto_approved")
    
    with cols[0]:
        st.markdown(f"""
        <div class="glass-card">
          <div class="metric-label">Files Processed</div>
          <div class="metric-value" style="background: none; -webkit-text-fill-color: #f8fafc;">{len(st.session_state.results)}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="glass-card">
          <div class="metric-label">Total Volume (INR)</div>
          <div class="metric-value">₹{total_val:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        color = "#f87171" if err_count > 0 else "#34d399"
        st.markdown(f"""
        <div class="glass-card">
          <div class="metric-label">Exceptions Found</div>
          <div class="metric-value" style="background: none; -webkit-text-fill-color: {color};">{err_count}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # --- VALIDATION BREAKDOWN ---
    st.markdown("### 🔍 Rule Validations & Extracted Data")
    for r in st.session_state.results:
        icon = "✅" if r.status == "auto_approved" else "⚠️" if r.status == "needs_review" else "❌"
        with st.expander(f"{icon} {r.filename} — {r.vendor_name or 'Unknown'} (₹{r.total_amount or 0:,.2f})"):
            st.markdown("#### 📄 Extracted Fields")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Invoice No:** {r.invoice_number}")
            c1.markdown(f"**Date:** {r.invoice_date}")
            c2.markdown(f"**Vendor:** {r.vendor_name}")
            c2.markdown(f"**PO Number:** {r.po_number}")
            c3.markdown(f"**Subtotal:** ₹{r.subtotal or 0:,.2f}")
            c3.markdown(f"**Tax:** ₹{r.tax_amount or 0:,.2f}")
            
            st.divider()
            st.markdown("#### 🤖 Rules Check")
            if not r.flags:
                st.markdown("<span class='success-text'>✓ All 3-way matching rules passed successfully.</span>", unsafe_allow_html=True)
            else:
                for f in r.flags:
                    st.markdown(f"- <span class='error-text'>[{f.severity.upper()}]</span> {f.message}", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 💬 Ask AI")

# Render existing chat history
for msg in st.session_state.chat_history:
    role = msg[0]
    content = msg[1]
    st.chat_message(role).write(content)

# Chat Input
if user_q := st.chat_input("Ask a question about your processed invoices..."):
    # Render user question immediately
    st.chat_message("human").write(user_q)
    
    # Show a spinner while the LLM thinks
    with st.spinner("Analyzing invoices..."):
        try:
            if st.session_state.get("demo_mode"):
                # Instant smart chat — reads real extracted data, zero LLM latency
                answer = smart_demo_answer(user_q, st.session_state.results)
            else:
                # Full LCEL chain with local Ollama (slow on CPU)
                answer = run_chat(
                    user_question=user_q,
                    chat_history=st.session_state.chat_history,
                    results_context=st.session_state.results
                )
            
            # Add to session history
            st.session_state.chat_history.append(("human", user_q))
            st.session_state.chat_history.append(("ai", answer))
            
            # Render AI response
            st.chat_message("ai").write(answer)
            
        except Exception as e:
            import traceback
            st.error(f"Chat failed: {e}")
            with st.expander("Details"):
                st.code(traceback.format_exc())
