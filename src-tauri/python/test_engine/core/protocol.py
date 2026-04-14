"""
core/protocol.py
================
JSON-Kommunikations-Protokoll zwischen Python (Test-Engine) und Rust (stdout).

Alle Messages folgen dem Schema:
  {"type": "<typ>", "timestamp": "<iso>", "data": {...}}

Rust (test_manager.rs) liest stdout zeilenweise und parsed jede Zeile als JSON.

  "progress"  → emit "test-progress"
  "status"    → emit "test-status"
  "complete"  → emit "test-complete"
  "error"     → emit "test-error"
  "warning"   → emit "test-status" (als Warnung)
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional


class MessageProtocol:
    """Sendet JSON-Messages über stdout an das Rust-Backend."""

    @staticmethod
    def _send(msg_type: str, data: Dict[str, Any]) -> None:
        msg = {
            "type": msg_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        print(json.dumps(msg, ensure_ascii=False, default=str), flush=True)

    # ── Fortschritt ────────────────────────────────────────────────────────
    @staticmethod
    def progress(current: int, total: int, samples_per_sec: float = 0.0) -> None:
        remaining = total - current
        eta = remaining / samples_per_sec if samples_per_sec > 0 else None
        MessageProtocol._send("progress", {
            "current_sample": current,
            "total_samples": total,
            "progress_percent": round((current / max(total, 1)) * 100, 2),
            "samples_per_second": round(samples_per_sec, 3),
            "estimated_time_remaining": eta,
        })

    # ── Status ─────────────────────────────────────────────────────────────
    @staticmethod
    def status(status: str, message: str = "") -> None:
        MessageProtocol._send("status", {"status": status, "message": message})

    # ── Fehler ─────────────────────────────────────────────────────────────
    @staticmethod
    def error(error: str, details: str = "") -> None:
        MessageProtocol._send("error", {"error": error, "details": details})

    # ── Warnung ────────────────────────────────────────────────────────────
    @staticmethod
    def warning(message: str) -> None:
        MessageProtocol._send("warning", {"message": message})

    # ── Debug ──────────────────────────────────────────────────────────────
    @staticmethod
    def debug(message: str, data: Optional[Dict[str, Any]] = None) -> None:
        import os
        if os.getenv("FRAMETRAIN_DEBUG"):
            MessageProtocol._send("debug", {"message": message, "data": data or {}})

    # ── Abgeschlossen ──────────────────────────────────────────────────────
    @staticmethod
    def complete(results: Dict[str, Any]) -> None:
        MessageProtocol._send("complete", results)
