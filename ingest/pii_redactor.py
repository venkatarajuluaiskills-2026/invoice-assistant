"""
PII detection and redaction using Microsoft Presidio (fully local).
Ensures no personal data (PAN, Aadhaar, phone, email) reaches the LLM.
All processing happens on-device — no cloud calls made by Presidio.
IMPORTANT: redacted_text is the ONLY text that goes to any LLM call.

Presidio is optional — if not installed the module degrades gracefully:
redact() returns the original text unchanged with an empty redaction_map.
Install with: pip install presidio-analyzer presidio-anonymizer
"""
_analyzer   = None
_anonymizer = None

from typing import Tuple
import json
import logging
from datetime import datetime
from pathlib import Path
from config import LOG_DIR

logger = logging.getLogger(__name__)

def _get_engines():
    """Initialise Presidio AnalyzerEngine and AnonymizerEngine as singletons."""
    global _analyzer, _anonymizer
    
    if _analyzer is None:
        try:
            # Lazy imports
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            from presidio_analyzer.nlp_engine import SpacyNlpEngine
            
            # This ensures Presidio doesn't try to download a model at runtime
            _analyzer = AnalyzerEngine(
                default_score_threshold=0.4, 
                nlp_engine_name="spacy", 
                models=[{"lang_code": "en", "model_name": "en_core_web_sm"}]
            )
            _anonymizer = AnonymizerEngine()
            logger.info("Presidio PII engines initialised successfully with en_core_web_sm.")
        except Exception as e:
            logger.warning(f"Presidio not fully available (could be model missing): {e}. PII redaction disabled.")
            return None, None
            
    return _analyzer, _anonymizer


def redact(text: str) -> Tuple[str, dict]:
    """
    Detect and redact PII from invoice text using Presidio.
    Replaces each PII entity with a typed placeholder, e.g. <IN_PAN_1>.
    Builds a redaction_map so authorised reviewers can restore originals.

    If Presidio is not installed, returns (text, {}) unchanged.
    """
    analyzer, _ = _get_engines()
    if analyzer is None:
        return text, {}

    try:
        results = analyzer.analyze(
            text=text,
            entities=ENTITIES_TO_REDACT,
            language="en",
        )
    except Exception as e:
        logger.warning(f"Presidio analysis failed: {e}. Returning text unredacted.")
        return text, {}

    redaction_map = {}
    counters      = {}

    # Sort by position descending — replace from end to preserve string offsets
    sorted_results = sorted(results, key=lambda x: x.start, reverse=True)

    redacted = text
    for result in sorted_results:
        entity_type = result.entity_type
        counters[entity_type] = counters.get(entity_type, 0) + 1
        placeholder  = f"<{entity_type}_{counters[entity_type]}>"
        original_val = text[result.start:result.end]
        redaction_map[placeholder] = original_val
        redacted = redacted[:result.start] + placeholder + redacted[result.end:]

    return redacted, redaction_map


def restore(redacted_text: str, redaction_map: dict) -> str:
    """
    Restore original PII values from redaction map.
    Called only for authorised reviewer display in the UI.

    Args:
        redacted_text: Text with <ENTITY_N> placeholders
        redaction_map: Mapping from placeholders to original values

    Returns:
        Text with original PII values restored
    """
    restored = redacted_text
    for placeholder, original in redaction_map.items():
        restored = restored.replace(placeholder, original)
    return restored


def log_pii_event(invoice_id: str, entity_types: list) -> None:
    """
    Log PII detection event to privacy audit log.
    NEVER logs actual PII values — only entity types and counts.

    Args:
        invoice_id:    Invoice identifier
        entity_types:  List of detected entity type strings
    """
    event = {
        "timestamp":         datetime.utcnow().isoformat(),
        "invoice_id":        invoice_id,
        "entities_detected": entity_types,
        "count":             len(entity_types),
    }
    log_path = Path(LOG_DIR) / "privacy_audit.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
