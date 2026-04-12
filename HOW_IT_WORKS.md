# 🤖 AI Invoice Assistant: How It Works

Welcome to the **AI Invoice Processing Assistant**. This document explains the sophisticated "Intelligent Pipeline" that powers this application, designed for speed, accuracy, and enterprise-grade compliance.

---

## 🏗️ The Intelligent Pipeline
The system processes every invoice through a strictly orchestrated 3-phase pipeline.

### Phase 1: Neural Extraction (The "Reader")
Instead of rigid templates, we use a **Large Language Model (LLM)** via the **Groq Cloud Infrastructure**.
- **OCR Engine**: Converts PDF/Image pixels into raw text.
- **PII Redactor**: Automatically identifies and masks sensitive data (Phone numbers, Personal IDs) *before* it leaves the local machine.
- **AI Parsing**: The AI reads the invoice like a human accountant, extracting Vendor details, GSTINs, PO numbers, and Line Items into a structured digital format.

### Phase 2: Deterministic Audit (The "Accountant")
AI can sometimes "hallucinate." To prevent this, we pass the data through a set of **hard-coded Python Validation Rules**.
- **Math Verification**: Cross-checks `Subtotal + Tax = Total` (±₹1 rounding).
- **Compliance Check**: Validates Indian GST formats and ensures HSN/SAC codes are valid.
- **Date Logic**: Ensures the invoice isn't in the future or expired (>180 days old).

### Phase 3: 3-Way Matching (The "Gatekeeper")
This is the heart of Procurement automation. The system cross-references the invoice against your internal databases:
1. **Invoice**: What the vendor claims.
2. **Purchase Order (PO)**: What your company authorized (Price/Quantity check).
3. **Goods Receipt Note (GRN)**: What was actually delivered to the warehouse.

> [!IMPORTANT]
> **Match Status**:
> - ✅ **Auto-Approved**: Perfect 3-way match within 3% price tolerance.
> - ⚠️ **Needs Review**: Variance detected (e.g., price changed) or mandatory fields missing.
> - ❌ **Failed**: Critical issue (e.g., GRN was rejected by the warehouse due to "damaged goods").

---

## 💬 Smart Conversational AI
The implementation features a **Context-Aware Chat Assistant**. Unlike generic chatbots:
- **Zero Hallucination**: It only answers based on the *actual* processed data of the current invoices.
- **Deep Context**: It knows every single flag, variance, and rejection reason (down to "Damaged box" notes in the GRN).
- **Privacy First**: It is programmed to block and redact sensitive PII even if the user asks for it.

---

## 🛠️ Technology Stack
- **Frontend**: Streamlit (Premium Dark Mode Dashboard)
- **Engine**: LangChain (LCEL - LangChain Expression Language)
- **AI Infrastructure**: Groq (Llama-3.1-8B-Instant) for sub-second cloud inference.
- **Data Security**: Microsoft Presidio (PII Identification).
- **OCR**: PyMuPDF & Tesseract.

---

## 📊 Summary of Value
| Feature | Benefit |
| :--- | :--- |
| **3-Way Matching** | Prevents overpayment and duplicate billing. |
| **Cloud-Fast AI** | Processes complex documents in <10 seconds. |
| **Rule-Based Validation** | 100% mathematical accuracy on tax and totals. |
| **Dynamic Chat** | Eliminates manual searching for "Why was this rejected?" |

---
*Created for the 2026 AI Innovation Hackathon*
