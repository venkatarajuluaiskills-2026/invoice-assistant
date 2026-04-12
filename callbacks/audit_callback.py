"""
LangChain BaseCallbackHandler for audit trail logging.
Logs every LLM call, chain start/end, and errors to audit_trail.jsonl.
Never logs actual prompt content — only metadata (duration, token counts).
"""
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List, Union
from datetime import datetime
from pathlib import Path
import json
import logging

from config import LOG_DIR, LLM_MODEL

logger = logging.getLogger(__name__)


class AuditCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that writes every LLM and chain event
    to an append-only JSONL audit trail at logs/audit_trail.jsonl.
    Captures: chain_start, chain_end, llm_start, llm_end, chain_error.
    """

    def __init__(self, invoice_id: str, action: str) -> None:
        """
        Args:
            invoice_id: Invoice this callback is tracking
            action:     Human-readable action label (e.g. "extraction")
        """
        super().__init__()
        self.invoice_id  = invoice_id
        self.action      = action
        self._start_time = None

    def on_chain_start(
        self, serialized: Dict, inputs: Dict, **kwargs: Any
    ) -> None:
        """Log chain start event."""
        self._start_time = datetime.utcnow()
        self._write({
            "event":  "chain_start",
            "chain":  serialized.get("name", "?"),
            "action": self.action,
        })

    def on_chain_end(self, outputs: Dict, **kwargs: Any) -> None:
        """Log chain end event with duration_ms."""
        duration_ms = None
        if self._start_time:
            duration_ms = int(
                (datetime.utcnow() - self._start_time).total_seconds() * 1000
            )
        self._write({
            "event":       "chain_end",
            "action":      self.action,
            "duration_ms": duration_ms,
        })

    def on_llm_start(
        self, serialized: Dict, prompts: List[str], **kwargs: Any
    ) -> None:
        """Log LLM call start — records prompt size, not content."""
        self._write({
            "event":        "llm_start",
            "model":        serialized.get("name", LLM_MODEL),
            "prompt_chars": sum(len(p) for p in prompts),
        })

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log LLM call end with token usage metadata."""
        token_usage = {}
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
        self._write({"event": "llm_end", "token_usage": token_usage})

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Log chain errors — records type and truncated message."""
        self._write({
            "event":         "chain_error",
            "error_type":    type(error).__name__,
            "error_message": str(error)[:200],
        })

    def _write(self, data: dict) -> None:
        """Append a timestamped event to the JSONL audit trail."""
        event = {
            "timestamp":  datetime.utcnow().isoformat(),
            "invoice_id": self.invoice_id,
            **data,
        }
        log_path = Path(LOG_DIR) / "audit_trail.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")


def log_audit_event(
    invoice_id: str,
    action: str,
    details: dict | None = None,
    user_note: str = "",
) -> None:
    """
    Convenience function — write a direct audit event without a callback.

    Args:
        invoice_id: Invoice identifier (use "batch" for batch-level events)
        action:     Action label (e.g. "invoice_processed", "export_csv")
        details:    Optional dict of additional metadata
        user_note:  Optional free-text note from the reviewer
    """
    event = {
        "timestamp":  datetime.utcnow().isoformat(),
        "invoice_id": invoice_id,
        "action":     action,
        "details":    details or {},
        "user_note":  user_note,
    }
    log_path = Path(LOG_DIR) / "audit_trail.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
